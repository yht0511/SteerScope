from .model import Model
import torch, einops
import random
from tqdm.auto import tqdm
import os
import pandas as pd
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from .interventions import (
    SubspaceIntervention,
    AdditionIntervention,
    ConceptVectorIntervention
)
from ..utils.constants import EXAMPLE_TAG
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations,
    get_lr,
    calculate_l1_losses
)
from transformers import get_scheduler
from transformers import set_seed
from .preference_model import *


class ConceptModel(Model):
    def __str__(self):
        return 'ConceptModel'

    def make_model(self, **kwargs):
        pass

    def make_preference_dataloader(self, examples, **kwargs):
        data_module = make_preference_data_module(self.tokenizer, examples, **kwargs)
        g = torch.Generator()
        g.manual_seed(self.seed)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True,
            batch_size=self.training_args.batch_size,
            collate_fn=data_module["data_collator"],
            generator=g)
        return train_dataloader

    def train(self, examples, **kwargs):
        if self.use_wandb:
            import wandb
            logging_metadata = kwargs["logging_metadata"]
            run_name = f"{logging_metadata['model_name']}_{logging_metadata['layer']}_{logging_metadata['concept_id']}"
            wandb_proj = kwargs.get("wandb_project", None)
            wandb_name = kwargs.get("wandb_name", None)
            run = wandb.init(
                project=f"{wandb_proj}",
                entity=wandb_name,
                name=run_name,
                dir="wandb",
            )

        train_dataloader = self.make_preference_dataloader(
            examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(),
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        num_training_steps = self.training_args.n_epochs * (len(train_dataloader) // self.training_args.gradient_accumulation_steps)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
        # Main training loop.
        rank = self.process_rank
        progress_bar, curr_step, logging_step = tqdm(range(num_training_steps), position=rank, leave=True), 0, 0

        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                expanded_batch_size = self.training_args.batch_size * len(self.preference_pairs)
                minibatch_size = self.training_args.batch_size
                num_minibatches = (expanded_batch_size + minibatch_size - 1) // minibatch_size

                winning_inputs = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                    "intervention_locations": [],
                    "steering_factors": [],
                }
                for i in range(self.training_args.batch_size):
                    for pair in self.preference_pairs:
                        winning_inputs["input_ids"].append(batch[f"{pair}_winning_input_ids"][i])
                        winning_inputs["attention_mask"].append(batch[f"{pair}_winning_attention_mask"][i])
                        winning_inputs["labels"].append(batch[f"{pair}_winning_labels"][i])
                        winning_inputs["intervention_locations"].append(batch[f"{pair}_winning_intervention_locations"][i])
                        winning_inputs["steering_factors"].append(torch.tensor(random.choice(self.training_args.steering_factors)))

                batch_metrics = {}
                loss_sum = 0

                for mb in range(num_minibatches):
                    start_idx = mb * minibatch_size
                    end_idx = min((mb + 1) * minibatch_size, expanded_batch_size)

                    if start_idx >= expanded_batch_size:
                        break
                    minibatch_inputs = {
                        k: torch.stack(winning_inputs[k][start_idx:end_idx], dim=0).to(self.device)
                        for k, _ in winning_inputs.items()
                    }
                    if isinstance(self.ax, list):
                        unit_locations = {"sources->base": (
                            None,
                            # repeat along first dimension
                            minibatch_inputs["intervention_locations"].permute(1, 0, 2).tolist() * len(self.ax)
                        )}
                    else:
                        unit_locations = {"sources->base": (
                            None,
                            minibatch_inputs["intervention_locations"].permute(1, 0, 2).tolist()
                        )}

                    subspaces = [{
                        "k": self.training_args.topk,
                        "steering_factor": minibatch_inputs["steering_factors"],
                    }]
                    subspace_repeat = 1 if not isinstance(self.ax, list) else len(self.ax)
                    subspaces = subspaces * subspace_repeat
                    _, cf_outputs = self.ax_model(
                        base={
                            "input_ids": minibatch_inputs["input_ids"],
                            "attention_mask": minibatch_inputs["attention_mask"]
                        }, unit_locations=unit_locations, labels=minibatch_inputs["labels"],
                        subspaces=subspaces, use_cache=False)

                    steer_loss = cf_outputs.loss
                    minibatch_loss = steer_loss

                    # Normalize loss by total number of minibatches for this step
                    # (instead of dividing by gradient_accumulation_steps)
                    minibatch_loss = minibatch_loss / (num_minibatches * self.training_args.gradient_accumulation_steps)

                    minibatch_loss.backward()

                    loss_sum += steer_loss.detach() * (end_idx - start_idx)

                loss = loss_sum / expanded_batch_size

                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.ax_model.parameters(), 1.0)
                    curr_lr = get_lr(optimizer)
                    # optim
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f" % (
                            curr_lr, loss))
                    curr_step += 1

        progress_bar.close()
        if self.use_wandb:
            run.finish()





    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        self.max_activations = {}
        return self.max_activations
