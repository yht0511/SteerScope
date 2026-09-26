"""Norm-preserving Spherical Steering with antipodal prototypes, a vMF gate, and geodesic updates."""

from __future__ import annotations

from contextlib import contextmanager
import logging

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from .model import Model


logger = logging.getLogger(__name__)


def spherical_geometric_logic(
    x: torch.Tensor,
    mu_t: torch.Tensor,
    mu_h: torch.Tensor,
    kappa: float,
    alpha: float,
    beta: float,
) -> tuple[torch.Tensor, bool]:
    """Apply the official scalar Spherical Steering update.

    This deliberately mirrors the reference implementation and is also useful
    for direct numerical parity checks. Production inference uses the
    equivalent vectorized implementation below.
    """

    original_dtype = x.dtype
    x = x.float()
    mu_t = mu_t.float()
    mu_h = mu_h.float()

    original_norm = x.norm(p=2).clamp_min(1e-12)
    x_hat = x / original_norm
    cos_t = torch.dot(x_hat, mu_t).clamp(-1, 1)
    cos_h = torch.dot(x_hat, mu_h).clamp(-1, 1)
    probabilities = F.softmax(
        torch.stack([float(kappa) * cos_t, float(kappa) * cos_h]), dim=0
    )
    delta = probabilities[1] - probabilities[0]
    if delta <= float(beta):
        return x.to(original_dtype), False

    amount = float(alpha) * (delta - float(beta)) / (1.0 - float(beta))
    amount = torch.clamp(amount, 0.0, 1.0)
    theta = torch.acos(cos_t)
    if theta < 1e-4:
        return x.to(original_dtype), False

    theta_new = (1.0 - amount) * theta
    orthogonal = (x_hat - cos_t * mu_t) / torch.sin(theta)
    rotated_hat = (
        torch.cos(theta_new) * mu_t
        + torch.sin(theta_new) * orthogonal
    )
    return (rotated_hat * original_norm).to(original_dtype), True


def _spherical_geometric_batch(
    hidden_states: torch.Tensor,
    mu_t: torch.Tensor,
    mu_h: torch.Tensor,
    kappa: float,
    alpha: torch.Tensor,
    beta: float,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized equivalent of :func:`spherical_geometric_logic`."""

    original_dtype = hidden_states.dtype
    hidden_float = hidden_states.float()
    mu_t = F.normalize(mu_t.float(), dim=-1).unsqueeze(1)
    mu_h = F.normalize(mu_h.float(), dim=-1).unsqueeze(1)
    original_norm = hidden_float.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    hidden_unit = hidden_float / original_norm

    cos_t = (hidden_unit * mu_t).sum(dim=-1).clamp(-1, 1)
    cos_h = (hidden_unit * mu_h).sum(dim=-1).clamp(-1, 1)
    probabilities = F.softmax(
        torch.stack(
            [float(kappa) * cos_t, float(kappa) * cos_h], dim=-1
        ),
        dim=-1,
    )
    delta = probabilities[..., 1] - probabilities[..., 0]
    alpha = alpha.to(hidden_float.device, torch.float32).reshape(-1, 1)
    amount = alpha * (delta - float(beta)) / (1.0 - float(beta))
    amount = amount.clamp(0.0, 1.0)

    theta = torch.acos(cos_t)
    theta_new = (1.0 - amount) * theta
    sin_theta = torch.sin(theta)
    orthogonal = (
        hidden_unit - cos_t.unsqueeze(-1) * mu_t
    ) / sin_theta.unsqueeze(-1)
    rotated_unit = (
        torch.cos(theta_new).unsqueeze(-1) * mu_t
        + torch.sin(theta_new).unsqueeze(-1) * orthogonal
    )
    rotated = rotated_unit * original_norm

    active = (
        token_mask.to(hidden_float.device).bool()
        & (delta > float(beta))
        & (theta >= 1e-4)
        & (alpha > 0)
    )
    output = torch.where(active.unsqueeze(-1), rotated, hidden_float)
    return output.to(original_dtype), active


class _SphericalPrototypeBank(nn.Module):
    """Checkpoint-compatible storage for one prototype per concept."""

    def __init__(self, hidden_size: int, num_prototypes: int):
        super().__init__()
        self.proj = nn.Linear(
            hidden_size,
            num_prototypes,
            bias=True,
            dtype=torch.float32,
        )
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
        self.proj.requires_grad_(False)


class SphericalSteering(Model):
    """Training-free prototype estimation plus adaptive spherical rotation."""

    requires_mean_activations = False
    requires_calibration_scale = False
    uses_intervention_positions = False
    load_trained_weights = True

    def __str__(self):
        return "SphericalSteering"

    @classmethod
    def training_fingerprint_context(cls):
        return {
            "artifact_version": 1,
            "prototype": "normalize(mean_positive_last_token-mean_negative_last_token)",
            "negative_prototype": "antipodal",
            "gate": "vmf_probability_difference",
            "update": "norm_preserving_geodesic_rotation",
            "factor_semantics": "official_alpha",
        }

    @staticmethod
    def _decoder_layers(model):
        candidates = (
            ("model", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
        )
        for parent_name, child_name in candidates:
            parent = getattr(model, parent_name, None)
            layers = getattr(parent, child_name, None) if parent is not None else None
            if layers is not None:
                return layers
        if getattr(model, "layers", None) is not None:
            return model.layers
        raise TypeError(
            f"{model.__class__.__name__} does not expose a supported decoder "
            "layer stack."
        )

    def _parameters(self) -> tuple[float, float]:
        kappa = float(getattr(self.training_args, "spherical_kappa", 20.0))
        beta = float(getattr(self.training_args, "spherical_beta", 0.1))
        if not torch.isfinite(torch.tensor(kappa)) or kappa <= 0:
            raise ValueError("spherical_kappa must be finite and positive.")
        if not torch.isfinite(torch.tensor(beta)) or not -1.0 <= beta < 1.0:
            raise ValueError("spherical_beta must be finite and in [-1, 1).")
        return kappa, beta

    def make_model(self, **kwargs):
        decoder_layers = self._decoder_layers(self.model)
        if self.layer < 0 or self.layer >= len(decoder_layers):
            raise ValueError(
                f"Invalid Spherical Steering layer {self.layer}; model has "
                f"{len(decoder_layers)} layers."
            )
        self._parameters()
        num_prototypes = int(kwargs.get("low_rank_dimension", 1) or 1)
        if num_prototypes < 1:
            raise ValueError("Spherical Steering requires at least one prototype.")
        self.ax = _SphericalPrototypeBank(
            int(self.model.config.hidden_size), num_prototypes
        ).to(self.device)
        self.ax.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _hidden_from_output(output):
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _replace_hidden(output, hidden):
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    @staticmethod
    def _last_real_positions(attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            attention_mask.shape[1], device=attention_mask.device
        ).unsqueeze(0)
        return (positions * attention_mask.long()).max(dim=1).values

    @classmethod
    def _last_token_mask(cls, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        rows = torch.arange(attention_mask.shape[0], device=attention_mask.device)
        mask[rows, cls._last_real_positions(attention_mask)] = True
        return mask

    @contextmanager
    def _capture_layer(self):
        captured = {}

        def hook(_module, _inputs, output):
            captured["hidden"] = self._hidden_from_output(output)

        handle = self._decoder_layers(self.model)[self.layer].register_forward_hook(hook)
        try:
            yield captured
        finally:
            handle.remove()

    @torch.no_grad()
    def train(self, examples: pd.DataFrame, **kwargs):
        if "labels" not in examples:
            raise KeyError(
                "SphericalSteering requires binarize_dataset: true so positive "
                "and negative sequences have binary labels."
            )
        labels = set(int(value) for value in examples["labels"].dropna().unique())
        if labels != {0, 1}:
            raise ValueError(
                "SphericalSteering requires both positive and negative examples."
            )
        if not hasattr(self, "ax"):
            self.make_model()

        self.model.eval()
        batch_size = int(getattr(self.training_args, "batch_size", 32) or 32)
        if batch_size < 1:
            raise ValueError("SphericalSteering batch_size must be positive.")
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        positive_activations = []
        negative_activations = []
        progress = tqdm(
            range(0, len(examples), batch_size),
            position=self.process_rank,
            leave=True,
            desc=str(self),
        )
        try:
            for start in range(0, len(examples), batch_size):
                batch = examples.iloc[start:start + batch_size]
                inputs = self.tokenizer(
                    batch["input"].tolist(),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                with self._capture_layer() as captured:
                    self.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                if "hidden" not in captured:
                    raise RuntimeError(
                        "SphericalSteering failed to capture the target layer."
                    )
                rows = torch.arange(inputs.input_ids.shape[0], device=self.device)
                positions = self._last_real_positions(inputs.attention_mask)
                activations = captured["hidden"][rows, positions].float().cpu()
                batch_labels = torch.as_tensor(
                    batch["labels"].tolist(), dtype=torch.long
                )
                positive_activations.append(activations[batch_labels == 1])
                negative_activations.append(activations[batch_labels == 0])
                progress.update(1)
        finally:
            progress.close()
            self.tokenizer.padding_side = original_padding_side

        positive = torch.cat(positive_activations, dim=0)
        negative = torch.cat(negative_activations, dim=0)
        if positive.numel() == 0 or negative.numel() == 0:
            raise ValueError(
                "SphericalSteering could not extract both prototype classes."
            )
        difference = positive.mean(dim=0) - negative.mean(dim=0)
        norm = difference.norm()
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError(
                "SphericalSteering found a non-finite or zero prototype direction."
            )
        prototype = difference / norm
        with torch.no_grad():
            self.ax.proj.weight[0].copy_(prototype.to(self.ax.proj.weight.device))
            self.ax.proj.bias.zero_()
        logger.warning(
            "SphericalSteering learned prototype from %s positive and %s negative "
            "sequences at layer %s.",
            len(positive),
            len(negative),
            self.layer,
        )

    def _prototype_indices(self, concept_ids) -> torch.Tensor:
        concept_ids = [int(value) for value in concept_ids]
        if self.concept_id_map is not None:
            concept_ids = [self.concept_id_map[value] for value in concept_ids]
        rows = self.ax.proj.weight.shape[0]
        if rows == 1:
            indices = [0] * len(concept_ids)
        else:
            indices = concept_ids
        if indices and (min(indices) < 0 or max(indices) >= rows):
            raise IndexError(
                f"SphericalSteering checkpoint has {rows} prototype rows but "
                f"received concept IDs {sorted(set(concept_ids))}."
            )
        return torch.as_tensor(indices, device=self.device, dtype=torch.long)

    @staticmethod
    def _strengths(batch_examples: pd.DataFrame, device) -> torch.Tensor:
        strengths = torch.as_tensor(
            batch_examples["factor"].tolist(),
            device=device,
            dtype=torch.float32,
        )
        if not torch.isfinite(strengths).all():
            raise ValueError("SphericalSteering factors must be finite.")
        if (strengths < 0).any() or (strengths > 1).any():
            raise ValueError(
                "SphericalSteering factor is the official alpha and must be "
                "in [0, 1]."
            )
        return strengths

    @contextmanager
    def _intervention(self, concept_indices, strengths, token_mask):
        kappa, beta = self._parameters()
        target_prototypes = self.ax.proj.weight[concept_indices].float()
        target_prototypes = F.normalize(target_prototypes, dim=-1)
        negative_prototypes = -target_prototypes

        def hook(_module, _inputs, output):
            hidden = self._hidden_from_output(output)
            current_mask = token_mask
            if current_mask.shape != hidden.shape[:2]:
                if hidden.shape[1] != 1 or hidden.shape[0] != current_mask.shape[0]:
                    raise ValueError(
                        "SphericalSteering token mask does not align with hidden states."
                    )
                current_mask = torch.ones(
                    hidden.shape[:2], device=hidden.device, dtype=torch.bool
                )
            rotated, _ = _spherical_geometric_batch(
                hidden,
                target_prototypes.to(hidden.device),
                negative_prototypes.to(hidden.device),
                kappa,
                strengths.to(hidden.device),
                beta,
                current_mask.to(hidden.device),
            )
            return self._replace_hidden(output, rotated)

        handle = self._decoder_layers(self.model)[self.layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def predict_steer(self, examples: pd.DataFrame, **kwargs):
        self.model.eval()
        self.ax.eval()
        batch_size = int(kwargs.get("batch_size", 64))
        generation_kwargs = self.generation_kwargs(
            kwargs.get("eval_output_length", 128),
            kwargs.get("temperature", 1.0),
            kwargs.get("do_sample", True),
        )
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        generations = []
        used_strengths = []
        progress = tqdm(
            range(0, len(examples), batch_size),
            position=self.process_rank,
            leave=True,
            disable=not kwargs.get("show_progress", True),
        )
        try:
            for start in range(0, len(examples), batch_size):
                batch = examples.iloc[start:start + batch_size]
                input_column = "steered_input" if kwargs.get("use_synergy") else "input"
                inputs = self.tokenizer(
                    batch[input_column].tolist(),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                strengths = self._strengths(batch, self.device)
                indices = self._prototype_indices(batch["concept_id"].tolist())
                token_mask = self._last_token_mask(inputs.attention_mask)
                with self._intervention(indices, strengths, token_mask):
                    output_ids = self.model.generate(**inputs, **generation_kwargs)
                prompt_width = inputs.input_ids.shape[1]
                generations.extend(self.tokenizer.batch_decode(
                    output_ids[:, prompt_width:], skip_special_tokens=True
                ))
                used_strengths.extend(strengths.cpu().tolist())
                progress.update(1)
        finally:
            progress.close()
            self.tokenizer.padding_side = original_padding_side
        return {
            "steered_generation": generations,
            "strength": used_strengths,
        }

    def prepare_choice_logits(self, examples, **kwargs):
        self.model.eval()
        self.ax.eval()

    def choice_forward(self, inputs, batch_examples, **kwargs):
        strengths = self._strengths(batch_examples, self.device)
        indices = self._prototype_indices(batch_examples["concept_id"].tolist())
        attention_mask = inputs.attention_mask
        if kwargs.get("full_sequence", False):
            token_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            positions = torch.arange(
                attention_mask.shape[1], device=attention_mask.device
            )
            for row_index, (_, row) in enumerate(batch_examples.iterrows()):
                candidate_length = len(row["_choice_token_ids"])
                real_positions = torch.nonzero(
                    attention_mask[row_index], as_tuple=False
                ).squeeze(1)
                if len(real_positions) <= candidate_length:
                    raise ValueError(
                        "SphericalSteering choice candidate has no prompt tokens."
                    )
                intervention_start = real_positions[-candidate_length] - 1
                token_mask[row_index] = (
                    attention_mask[row_index].bool()
                    & (positions >= intervention_start)
                )
        else:
            token_mask = self._last_token_mask(attention_mask)

        model_inputs = self.choice_model_inputs(
            inputs,
            last_token_only=not kwargs.get("full_sequence", False),
            logits_to_keep=kwargs.get("choice_logits_to_keep"),
        )
        with self._intervention(indices, strengths, token_mask):
            outputs = self.model(**model_inputs, use_cache=False)
        return outputs, strengths
