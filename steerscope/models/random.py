from .model import Model
from .mean import DiffMean as _DiffMean
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
    AdditionIntervention
)
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm, 
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr,
    calculate_l1_losses
)
from transformers import get_scheduler

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, low_rank_dimension):
        super(LogisticRegressionModel, self).__init__()
        # Linear layer: input_dim -> 1 output (since binary classification)
        self.proj = torch.nn.Linear(input_dim, low_rank_dimension)
    
    def forward(self, x):
        return self.proj(x)


class RandomOriginal(Model):
    requires_calibration_scale = True
    
    def __str__(self):
        return 'RandomOriginal'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        if mode in {"latent", "train"}:
            ax = LogisticRegressionModel(
                self.model.config.hidden_size, kwargs.get("low_rank_dimension", 1))
            ax.to(self.device)
            self.ax = ax
        elif mode == "steering":
            ax = AdditionIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),
            )
            self.ax = ax
            self.ax.train()
            ax_config = IntervenableConfig(representations=[{
                "layer": l,
                "component": f"model.layers[{l}].output",
                "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
                "intervention": self.ax} for l in [self.layer]])
            ax_model = IntervenableModel(ax_config, self.model)
            ax_model.set_device(self.device)
            self.ax_model = ax_model
    
    def train(self, examples, **kwargs):
        torch.cuda.empty_cache()
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Dummy training finished :) I'm a random baseline.")


class Random(_DiffMean):
    """DiffMean over a deterministic random split of the training examples."""

    def __str__(self):
        return 'Random'

    @classmethod
    def training_fingerprint_context(cls):
        return {
            **super().training_fingerprint_context(),
            "artifact_version": 2,
            "labels": "balanced_random_split",
            "split_seed": "model_seed",
        }

    @torch.no_grad()
    def train(self, examples, **kwargs):
        if "labels" not in examples.columns:
            raise ValueError("Random requires binary training examples with labels.")
        if len(examples) < 2:
            raise ValueError("Random requires at least two training examples.")

        randomized_examples = examples.sample(
            frac=1.0,
            random_state=self.seed,
        ).reset_index(drop=True)
        examples_per_class = len(randomized_examples) // 2
        if examples_per_class == 0:
            raise ValueError("Random requires at least one example per random class.")
        if len(randomized_examples) % 2:
            logger.warning(
                "Random received an odd number of examples; dropping one example "
                "to preserve the balanced split required by DiffMean."
            )
            randomized_examples = randomized_examples.iloc[
                :2 * examples_per_class
            ].copy()

        randomized_examples.loc[:, "labels"] = (
            [1] * examples_per_class + [0] * examples_per_class
        )
        logger.warning(
            "Random discarded the original labels and assigned %d examples to "
            "each random class using seed %d.",
            examples_per_class,
            self.seed,
        )
        super().train(randomized_examples, **kwargs)
