from .model import Model
import torch, einops
from tqdm.auto import tqdm
import os
import pandas as pd
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from .interventions import (
    SteeringVectorIntervention,
    AdditionIntervention,
    SubspaceIntervention
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
from ..utils.training import (
    is_optimizer_step,
    normalize_loss_for_accumulation,
    optimizer_steps_per_epoch,
)


class SteeringVector(Model):
    requires_calibration_scale = True

    def __str__(self):
        return 'SteeringVector'

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
            intervention_type = kwargs.get("intervention_type", "addition")
            ax = SteeringVectorIntervention(
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

    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(), 
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        accumulation_steps = int(self.training_args.gradient_accumulation_steps or 1)
        num_microbatches = len(train_dataloader)
        num_training_steps = self.training_args.n_epochs * optimizer_steps_per_epoch(
            num_microbatches, accumulation_steps
        )
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
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
        
                # forward
                _, cf_outputs = self.ax_model(
                    base={
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }, unit_locations=unit_locations, labels=inputs["labels"],
                    subspaces=subspaces, use_cache=False)
                
                # loss
                loss = cf_outputs.loss
                
                # Keep the original update path unchanged when accumulation is 1.
                if accumulation_steps > 1:
                    loss = normalize_loss_for_accumulation(
                        loss, step, num_microbatches, accumulation_steps
                    )
                loss.backward()
                if is_optimizer_step(step, num_microbatches, accumulation_steps):
                    curr_step += 1
                    curr_lr = get_lr(optimizer)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f" % (curr_lr, loss))
        progress_bar.close()
        set_decoder_norm_to_unit_norm(self.ax)
    
    
