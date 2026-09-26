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
from .concept_model import *


class ConceptVector(ConceptModel):
    def __str__(self):
        return 'ConceptVector'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        print("**Getting embed dim from the following model config**")
        if mode == "steering":
            intervention_type = kwargs.get("intervention_type", "addition")
            if intervention_type == "addition":
                ax = AdditionIntervention(
                    embed_dim=kwargs.get("embed_dim", self.model.config.hidden_size), 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "addition_suppression":
                ax = AdditionSuppressionIntervention(
                    embed_dim=kwargs.get("embed_dim", self.model.config.hidden_size), 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            else:
                raise ValueError(f"Intervention type {intervention_type} not supported")
        else:
            intervention_type = kwargs.get("intervention_type", "addition")
            if intervention_type == "addition":
                ax = PreferenceVectorIntervention(
                    embed_dim=kwargs.get("embed_dim", self.model.config.hidden_size), 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                    dropout=kwargs.get("dropout", 0.0),
                    intervention_positions_dropout=kwargs.get("intervention_positions_dropout", 0.0)
                )
        self.intervention_type = intervention_type
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
        self.preference_pairs = kwargs.get("preference_pairs", ["orig_add"])

 