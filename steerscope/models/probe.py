from .model import Model
import torch, transformers, datasets
from tqdm.auto import tqdm
import os
import pandas as pd
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Any
from torch.utils.data import DataLoader
from .interventions import (
    AdditionIntervention,
    SubspaceIntervention,
    ProbeIntervention,
    SparseProbeIntervention
)
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm, 
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr,
    calculate_l1_losses
)
from transformers import get_scheduler
from ..utils.training import (
    GRADIENT_ACCUMULATION_SEMANTICS,
    is_optimizer_step,
    normalize_loss_for_accumulation,
    optimizer_steps_per_epoch,
)

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


@dataclass
class DataCollator(object):
    """Collate examples for ReFT."""
    
    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_intervention_len = max([len(inst["intervention_locations"][0]) for inst in instances])
        max_seq_len = max([len(inst["input_ids"]) for inst in instances])
        
        for inst in instances:
            non_pad_len = len(inst["input_ids"])

            _intervention_mask = torch.ones_like(inst["intervention_locations"][0])
            _intervention_location_paddings = torch.tensor(
                [[len(inst["input_ids"]) for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))]])
            _intervention_mask_paddings = torch.tensor(
                [0 for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))])
            inst["intervention_locations"] = torch.cat([inst["intervention_locations"], _intervention_location_paddings], dim=-1).int()
            inst["intervention_masks"] = torch.cat([_intervention_mask, _intervention_mask_paddings], dim=-1).int()

            _input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)])
            inst["input_ids"] = torch.cat((inst["input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _input_id_paddings)).int()
            inst["attention_mask"] = (inst["input_ids"] != self.tokenizer.pad_token_id).int()
            inst["labels"] = inst["labels"].int()
        batch_inputs = self.data_collator(instances)
        return batch_inputs


def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer, model, df, prefix_length=1
):
    all_input_ids, all_labels, all_intervention_locations = [], [], []
    all_assistant_starts = []
    for _, row in df.iterrows():
        input_ids = tokenizer(
            row["input"], max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_length = len(input_ids)
        activation_start = (
            int(row["assistant_start"])
            if "assistant_start" in df.columns
            else prefix_length
        )
        intervention_locations = torch.tensor(
            [[i for i in range(activation_start, base_length)]]
        )
        all_input_ids.append(input_ids)
        all_labels.append(row["labels"])
        all_intervention_locations.append(intervention_locations)
        if "assistant_start" in df.columns:
            all_assistant_starts.append(int(row["assistant_start"]))

    dataset_values = {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "intervention_locations": all_intervention_locations
    }
    format_columns = ['input_ids', 'labels', 'intervention_locations']
    if all_assistant_starts:
        dataset_values["assistant_start"] = all_assistant_starts
        format_columns.append("assistant_start")
    train_dataset = datasets.Dataset.from_dict(dataset_values)
    train_dataset.set_format(type='torch', columns=format_columns)

    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = DataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class LinearProbe(Model):
    requires_calibration_scale = True

    @classmethod
    def training_fingerprint_context(cls):
        return {
            **super().training_fingerprint_context(),
            "gradient_accumulation_semantics": (
                GRADIENT_ACCUMULATION_SEMANTICS
            ),
        }

    def __str__(self):
        return 'LinearProbe'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        if mode == "steering":
            intervention_type = kwargs.get("intervention_type", "addition")
            if intervention_type == "addition":
                ax = AdditionIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "clamping":
                ax = SubspaceIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
        else:
            ax = ProbeIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),
            )
        layers = self.steering_layers if self.steering_layers else [self.layer]
        self.ax = ax.to(self.device)
        self.ax.train()
        ax_config = IntervenableConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
            "intervention": self.ax} for l in layers])
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model
    
    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, self.model, examples)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"])
        return train_dataloader

    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax.parameters(), lr=self.training_args.lr, 
            weight_decay=self.training_args.weight_decay)
        accumulation_steps = int(
            self.training_args.gradient_accumulation_steps or 1
        )
        num_microbatches = len(train_dataloader)
        num_training_steps = self.training_args.n_epochs * (
            optimizer_steps_per_epoch(num_microbatches, accumulation_steps)
        )
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
        criterion = torch.nn.BCELoss()
        # Main training loop.
        rank = self.process_rank
        progress_bar, curr_step = tqdm(range(num_training_steps), position=rank, leave=True), 0
        
        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                unit_locations={"sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )}
                subspaces = [{
                    "k": self.training_args.topk
                }]
        
                # Run the forward pass only to collect probe activations.
                _, _ = self.ax_model(
                    base={
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }, unit_locations=unit_locations, subspaces=subspaces, use_cache=False)
                
                latent = self.ax_model.full_intervention_outputs[0].latent[0] # bs, n_tokens
                preds = torch.sigmoid(latent) # bs, n_tokens
                expanded_labels = inputs["labels"].unsqueeze(-1).expand_as(preds) # bs, n_tokens
                # Compute loss only on valid tokens
                loss = criterion(
                    preds[inputs["intervention_masks"].bool()].float(), 
                    expanded_labels[inputs["intervention_masks"].bool()].float()
                )
                l1_loss = sum(p.abs().sum() for p in self.ax.parameters())
                loss += self.training_args.coeff_l1_loss*l1_loss
                
                # accuracy
                pred_labels = (preds > 0.5).long()
                acc = (pred_labels[inputs["intervention_masks"].bool()] == 
                       expanded_labels[inputs["intervention_masks"].bool()]).float().mean()

                loss = normalize_loss_for_accumulation(
                    loss,
                    step,
                    num_microbatches,
                    accumulation_steps,
                )
                loss.backward()
                if is_optimizer_step(
                    step,
                    num_microbatches,
                    accumulation_steps,
                ):
                    set_decoder_norm_to_unit_norm(self.ax)
                    remove_gradient_parallel_to_decoder_directions(self.ax)
                    curr_step += 1
                    curr_lr = get_lr(optimizer)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f || acc %.3f" % (
                            curr_lr, loss, acc))
        progress_bar.close()

    
