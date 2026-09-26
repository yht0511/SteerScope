"""HiDRA following Equation 9 of arXiv:2606.15092v1 with a fixed Gaussian projection and invertible LeakyReLU."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .model import Model


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HiDRAConfig:
    input_dim: int
    projected_dim: int = 8192
    negative_slope: float = 0.7
    seed: int = 42


class _HiDRAMatrix:
    """Device-local Gaussian projection with a lazy pseudo-inverse."""

    def __init__(self, config: HiDRAConfig, device: torch.device):
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.projection = torch.randn(
            config.projected_dim,
            config.input_dim,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(device)
        self._pseudoinverse: torch.Tensor | None = None

    @property
    def pseudoinverse(self) -> torch.Tensor:
        if self._pseudoinverse is None:
            gram = self.projection.T @ self.projection
            cholesky = torch.linalg.cholesky(gram)
            self._pseudoinverse = torch.cholesky_solve(
                self.projection.T, cholesky
            )
        return self._pseudoinverse


_HIDRA_MATRIX_CACHE: dict[tuple[int, int, int, str], _HiDRAMatrix] = {}


def _canonical_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


class HiDRAProjector:
    """Gaussian lift, invertible nonlinearity, and Moore-Penrose inverse."""

    def __init__(self, config: HiDRAConfig, device: torch.device | str):
        if config.projected_dim < config.input_dim:
            raise ValueError("HiDRA requires projected_dim >= input_dim.")
        if not 0.0 < config.negative_slope < 1.0:
            raise ValueError("HiDRA negative_slope must lie in (0, 1).")
        if config.seed < 0:
            raise ValueError("HiDRA projection seed must be non-negative.")
        self.config = config
        self.device = _canonical_device(device)
        key = (
            config.input_dim,
            config.projected_dim,
            config.seed,
            str(self.device),
        )
        if key not in _HIDRA_MATRIX_CACHE:
            _HIDRA_MATRIX_CACHE[key] = _HiDRAMatrix(config, self.device)
        self._matrix = _HIDRA_MATRIX_CACHE[key]

    @property
    def projection(self) -> torch.Tensor:
        return self._matrix.projection

    @property
    def pseudoinverse(self) -> torch.Tensor:
        return self._matrix.pseudoinverse

    def lift(self, activation: torch.Tensor) -> torch.Tensor:
        projected = activation.float() @ self.projection.T
        return F.leaky_relu(
            projected,
            negative_slope=self.config.negative_slope,
        )

    def inverse_nonlinearity(self, lifted: torch.Tensor) -> torch.Tensor:
        slope = self.config.negative_slope
        return torch.where(lifted >= 0, lifted, lifted / slope)

    def lower(self, lifted: torch.Tensor) -> torch.Tensor:
        return self.inverse_nonlinearity(lifted) @ self.pseudoinverse.T

    def steer(
        self,
        activation: torch.Tensor,
        direction: torch.Tensor,
        alpha: float | torch.Tensor,
    ) -> torch.Tensor:
        """Apply the paper's Equation (9)."""

        lifted = self.lift(activation)
        strength = torch.as_tensor(
            alpha, device=lifted.device, dtype=torch.float32
        )
        while strength.ndim < lifted.ndim:
            strength = strength.unsqueeze(-1)
        return self.lower(lifted + strength * direction.float())


def difference_in_means(
    positive: torch.Tensor,
    negative: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """Estimate HiDRA's lifted-space DiM direction."""

    direction = positive.float().mean(0) - negative.float().mean(0)
    return F.normalize(direction, dim=0) if normalize else direction


def _hidra_geometric_batch(
    hidden_states: torch.Tensor,
    directions: torch.Tensor,
    strengths: torch.Tensor,
    token_mask: torch.Tensor,
    projector: HiDRAProjector,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply HiDRA only to selected nonzero-strength token positions."""

    strengths = strengths.to(hidden_states.device, torch.float32).reshape(-1)
    token_mask = token_mask.to(hidden_states.device).bool()
    active = token_mask & strengths.ne(0).unsqueeze(1)
    if not active.any():
        return hidden_states, active

    rows = torch.arange(
        hidden_states.shape[0], device=hidden_states.device
    ).unsqueeze(1).expand(hidden_states.shape[:2])
    active_rows = rows[active]
    selected = hidden_states[active]
    selected_directions = directions.to(hidden_states.device)[active_rows]
    selected_strengths = strengths[active_rows]
    steered = projector.steer(
        selected,
        selected_directions,
        selected_strengths,
    ).to(hidden_states.dtype)
    output = hidden_states.clone()
    output[active] = steered
    return output, active


class HiDRA(Model):
    """Training-free lifted-space difference-in-means steering."""

    requires_mean_activations = False
    requires_calibration_scale = False
    uses_intervention_positions = False
    load_trained_weights = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.directions: torch.Tensor | None = None
        self.projector: HiDRAProjector | None = None

    def __str__(self):
        return "HiDRA"

    @classmethod
    def training_fingerprint_context(cls):
        return {
            "artifact_version": 1,
            "equation": "A_dagger sigma_inverse(sigma(Ax)+alpha*d)",
            "projection": "iid_standard_gaussian_shared",
            "direction": "lifted_last_token_difference_in_means",
            "intervention": "all_valid_tokens",
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

    def _config(self) -> HiDRAConfig:
        if self.training_args is None:
            raise ValueError("HiDRA requires its training configuration.")
        projected_dim = int(
            getattr(self.training_args, "hidra_projected_dim", 8192)
        )
        negative_slope = float(
            getattr(self.training_args, "hidra_negative_slope", 0.7)
        )
        projection_seed = int(
            getattr(self.training_args, "hidra_projection_seed", 42)
        )
        if not math.isfinite(negative_slope):
            raise ValueError("hidra_negative_slope must be finite.")
        return HiDRAConfig(
            input_dim=int(self.model.config.hidden_size),
            projected_dim=projected_dim,
            negative_slope=negative_slope,
            seed=projection_seed,
        )

    def _normalize_direction(self) -> bool:
        return bool(
            getattr(self.training_args, "hidra_normalize_direction", False)
        )

    def make_model(self, **kwargs):
        decoder_layers = self._decoder_layers(self.model)
        if self.layer < 0 or self.layer >= len(decoder_layers):
            raise ValueError(
                f"Invalid HiDRA layer {self.layer}; model has "
                f"{len(decoder_layers)} layers."
            )
        config = self._config()
        self.projector = HiDRAProjector(config, self.device)
        num_directions = int(kwargs.get("low_rank_dimension", 1) or 1)
        if num_directions < 1:
            raise ValueError("HiDRA requires at least one direction row.")
        self.directions = torch.zeros(
            num_directions,
            config.projected_dim,
            device=self.device,
            dtype=torch.float32,
        )
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
                "HiDRA requires binarize_dataset: true so positive and "
                "negative sequences have binary labels."
            )
        labels = set(int(value) for value in examples["labels"].dropna().unique())
        if labels != {0, 1}:
            raise ValueError("HiDRA requires both positive and negative examples.")
        if self.projector is None or self.directions is None:
            self.make_model()

        self.model.eval()
        batch_size = int(getattr(self.training_args, "batch_size", 32) or 32)
        if batch_size < 1:
            raise ValueError("HiDRA batch_size must be positive.")
        projected_dim = self.projector.config.projected_dim
        positive_sum = torch.zeros(projected_dim, dtype=torch.float32)
        negative_sum = torch.zeros(projected_dim, dtype=torch.float32)
        positive_count = 0
        negative_count = 0
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
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
                    raise RuntimeError("HiDRA failed to capture the target layer.")
                rows = torch.arange(inputs.input_ids.shape[0], device=self.device)
                positions = self._last_real_positions(inputs.attention_mask)
                activations = captured["hidden"][rows, positions]
                lifted = self.projector.lift(activations)
                batch_labels = torch.as_tensor(
                    batch["labels"].tolist(),
                    device=lifted.device,
                    dtype=torch.long,
                )
                positive_batch = lifted[batch_labels == 1]
                negative_batch = lifted[batch_labels == 0]
                if len(positive_batch):
                    positive_sum += positive_batch.sum(dim=0).cpu()
                    positive_count += len(positive_batch)
                if len(negative_batch):
                    negative_sum += negative_batch.sum(dim=0).cpu()
                    negative_count += len(negative_batch)
                progress.update(1)
        finally:
            progress.close()
            self.tokenizer.padding_side = original_padding_side

        if positive_count == 0 or negative_count == 0:
            raise ValueError("HiDRA could not extract both direction classes.")
        direction = (
            positive_sum / positive_count - negative_sum / negative_count
        )
        if self._normalize_direction():
            direction = F.normalize(direction, dim=0)
        norm = direction.norm()
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("HiDRA found a non-finite or zero direction.")
        self.directions[0].copy_(direction.to(self.device))
        logger.warning(
            "HiDRA learned a %s-dimensional direction from %s positive and %s "
            "negative sequences at layer %s (norm=%.6f, normalized=%s).",
            projected_dim,
            positive_count,
            negative_count,
            self.layer,
            float(norm),
            self._normalize_direction(),
        )

    def save(self, dump_dir, **kwargs):
        if self.directions is None:
            raise RuntimeError("HiDRA has no learned directions to save.")
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        model_name = kwargs.get("model_name", str(self))
        torch.save(
            self.directions.detach().float().cpu(),
            dump_dir / f"{model_name}_weight.pt",
        )
        torch.save(
            torch.zeros(self.directions.shape[0], dtype=torch.float32),
            dump_dir / f"{model_name}_bias.pt",
        )

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", str(self))
        concept_id = int(kwargs.get("concept_id", 0))
        priority_mode = kwargs.get("priority_mode", "compute_priority")
        weight_path = Path(dump_dir) / f"{model_name}_weight.pt"
        bias_path = Path(dump_dir) / f"{model_name}_bias.pt"
        weight = torch.load(weight_path, map_location="cpu", weights_only=True)
        bias = torch.load(bias_path, map_location="cpu", weights_only=True)
        if weight.ndim != 2:
            raise ValueError("HiDRA checkpoint weights must be rank two.")
        if bias.reshape(-1).numel() != weight.shape[0]:
            raise ValueError("HiDRA checkpoint weight and bias rows do not match.")
        if weight.shape[1] != self._config().projected_dim:
            raise ValueError(
                f"HiDRA checkpoint width {weight.shape[1]} does not match "
                f"hidra_projected_dim={self._config().projected_dim}."
            )
        if not torch.isfinite(weight).all():
            raise ValueError("HiDRA checkpoint contains non-finite directions.")

        if priority_mode == "mem_priority":
            if concept_id < 0 or concept_id >= weight.shape[0]:
                raise IndexError(
                    f"HiDRA checkpoint has {weight.shape[0]} concepts and cannot "
                    f"select concept ID {concept_id}."
                )
            weight = weight[concept_id:concept_id + 1]
            self.concept_id_map = {concept_id: 0}
        self.make_model(low_rank_dimension=weight.shape[0])
        self.directions.copy_(weight.to(self.device, torch.float32))

    def _direction_indices(self, concept_ids) -> torch.Tensor:
        if self.directions is None:
            raise RuntimeError("HiDRA directions have not been initialized.")
        concept_ids = [int(value) for value in concept_ids]
        if self.concept_id_map is not None:
            concept_ids = [self.concept_id_map[value] for value in concept_ids]
        rows = self.directions.shape[0]
        if rows == 1:
            indices = [0] * len(concept_ids)
        else:
            indices = concept_ids
        if indices and (min(indices) < 0 or max(indices) >= rows):
            raise IndexError(
                f"HiDRA checkpoint has {rows} direction rows but received "
                f"concept IDs {sorted(set(concept_ids))}."
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
            raise ValueError("HiDRA factors must be finite.")
        return strengths

    @contextmanager
    def _intervention(self, direction_indices, strengths, token_mask):
        if self.directions is None or self.projector is None:
            raise RuntimeError("HiDRA must be initialized before inference.")
        selected_directions = self.directions[direction_indices].float()

        def hook(_module, _inputs, output):
            hidden = self._hidden_from_output(output)
            current_mask = token_mask
            if current_mask.shape != hidden.shape[:2]:
                if hidden.shape[1] != 1 or hidden.shape[0] != current_mask.shape[0]:
                    raise ValueError(
                        "HiDRA token mask does not align with hidden states."
                    )
                current_mask = torch.ones(
                    hidden.shape[:2], device=hidden.device, dtype=torch.bool
                )
            steered, _ = _hidra_geometric_batch(
                hidden,
                selected_directions.to(hidden.device),
                strengths.to(hidden.device),
                current_mask.to(hidden.device),
                self.projector,
            )
            if steered is hidden:
                return output
            return self._replace_hidden(output, steered)

        handle = self._decoder_layers(self.model)[self.layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def predict_steer(self, examples: pd.DataFrame, **kwargs):
        self.model.eval()
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
                indices = self._direction_indices(batch["concept_id"].tolist())
                token_mask = inputs.attention_mask.bool()
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

    def choice_forward(self, inputs, batch_examples, **kwargs):
        strengths = self._strengths(batch_examples, self.device)
        indices = self._direction_indices(batch_examples["concept_id"].tolist())
        token_mask = inputs.attention_mask.bool()
        model_inputs = self.choice_model_inputs(
            inputs,
            last_token_only=not kwargs.get("full_sequence", False),
            logits_to_keep=kwargs.get("choice_logits_to_keep"),
        )
        with self._intervention(indices, strengths, token_mask):
            outputs = self.model(**model_inputs, use_cache=False)
        return outputs, strengths

    def get_logits(self, concept_id, k=10):
        # The learned direction lives in the lifted space and cannot be mapped
        # through the residual unembedding as if it were an additive vector.
        return [None], [None]

    def to(self, device):
        self.device = _canonical_device(device)
        if self.directions is not None:
            self.directions = self.directions.to(self.device, torch.float32)
        if self.projector is not None:
            self.projector = HiDRAProjector(self._config(), self.device)
        return self
