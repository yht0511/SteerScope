from pathlib import Path
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
    LoraIntervention,
    PreferenceLoraIntervention,
)
from ..utils.constants import EXAMPLE_TAG
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr,
    calculate_l1_losses
)
from ..utils.data_utils import (
    parse_positions, 
    get_intervention_locations,
    InterventionDataCollator
)
from dataclasses import dataclass
from transformers import set_seed, get_scheduler, DataCollatorForSeq2Seq, DataCollator
import transformers, datasets
from typing import Dict, Optional, Sequence, Union, List, Any

from .preference_model import *


class PreferenceLoRA(PreferenceModel):
    inference_instance_scope = "per_concept"
    artifact_directory = "preference_lora"

    def __str__(self):
        return 'PreferenceLoRA'

    def make_model(self, **kwargs):
        if kwargs["mode"] == "steering":
            lora_layers = self.steering_layers if self.steering_layers is not None else [self.layer]
        elif kwargs["mode"] == "train":
            lora_layers = [self.layer] if self.training_args.lora_layers is None else self.training_args.lora_layers
        # there is one type of intervention throughout
        self.number_of_interventions = len(lora_layers)
        ax = []
        for _ in range(self.number_of_interventions):
            _ax = PreferenceLoraIntervention(
                input_dim=self.model.config.num_attention_heads*self.model.config.head_dim,
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                alpha=self.training_args.lora_alpha,
                dropout=kwargs.get("dropout", 0.0),
            )
            _ = _ax.to(self.device)
            _ = _ax.train()
            ax.append(_ax)
        self.ax = ax
        
        # This implementation supports one component.
        lora_component = self.training_args.lora_components[0]
        assert len(self.training_args.lora_components) == 1, "Only one component is supported for now"
        ax_config = IntervenableConfig(representations=[{
                "layer": l,
                "component": f"model.layers[{l}].self_attn.{lora_component}.output",
                "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
                # each layer has its own LoRA intervention
                "intervention": ax[i]
            } for i, l in enumerate(lora_layers)])
        print("ax_config:")
        print(ax_config)
        ax_model = IntervenableModel(ax_config, self.model, as_adaptor=True)
        ax_model.set_device(self.device)
        self.ax_model = ax_model
        # lora is concept-ful due to its nature.
        self.concept_id = kwargs.get("concept_id")
        self.preference_pairs = kwargs.get("preference_pairs", ["orig_add"])

    def save(self, dump_dir, **kwargs): 
        # gonna to the folder-based saving, way more easier than 3d matrix handling.
        dump_dir = Path(f"{dump_dir}/preference_lora/{self.concept_id}")
        dump_dir.mkdir(parents=True, exist_ok=True)
        self.ax_model.save(dump_dir) # calls pyvene intervention save

    def load(self, dump_dir=None, **kwargs):
        priority_mode = kwargs.get("priority_mode", "compute_priority")
        self.priority_mode = priority_mode
        # folder-based loading
        self.concept_id = kwargs.get("concept_id")
        dump_dir = Path(f"{dump_dir}/preference_lora/{self.concept_id}")
        self.make_model(**kwargs)
        self.ax_model.load_intervention(dump_dir, include_model=False)

    def prepare_choice_logits(self, examples, **kwargs):
        self.ax_model.eval()

    def choice_forward(self, inputs, batch_examples, **kwargs):
        mag = torch.as_tensor(
            batch_examples["factor"].tolist(), device=self.device
        )
        _, outputs = self.ax_model(
            base=self.choice_model_inputs(
                inputs,
                last_token_only=not kwargs.get("full_sequence", False),
                logits_to_keep=kwargs.get("choice_logits_to_keep"),
            ),
            unit_locations=None,
            subspaces=[{"steering_factor": mag}] * self.number_of_interventions,
            use_cache=False,
        )
        return outputs, mag
