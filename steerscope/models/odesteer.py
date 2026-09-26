"""ODESteer and StepODESteer adapted from the released implementation at commit 8a3c481d6493ecb3325eea5ef9c448cccfced7eb."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import logging
import math
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torchdiffeq import odeint
from tqdm.auto import tqdm

from .model import Model


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ODESteerConfig:
    degree: int = 2
    n_components: int = 8000
    gamma: float = 0.1
    coef0: float = 1.0
    linear_classifier: str = "lr"
    sketch_seed: int = 42
    solver: str = "euler"
    steps: int = 10

    def validate(self) -> None:
        if self.degree < 1:
            raise ValueError("ode_degree must be at least one.")
        if self.n_components < 1:
            raise ValueError("ode_n_components must be at least one.")
        if not math.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("ode_gamma must be finite and non-negative.")
        if not math.isfinite(self.coef0) or self.coef0 < 0:
            raise ValueError("ode_coef0 must be finite and non-negative.")
        if self.linear_classifier not in {"lr", "svm"}:
            raise ValueError("ode_linear_classifier must be 'lr' or 'svm'.")
        if self.sketch_seed < 0:
            raise ValueError("ode_sketch_seed must be non-negative.")
        if self.solver not in {"euler", "midpoint", "rk4"}:
            raise ValueError("ode_solver must be euler, midpoint, or rk4.")
        if self.steps < 1:
            raise ValueError("ode_steps must be at least one.")


class NormedPolyCountSketch(nn.Module):
    """Normalized polynomial TensorSketch with the paper's analytical VJP."""

    normalization_epsilon = 1e-12

    def __init__(self, config: ODESteerConfig):
        super().__init__()
        self.config = config
        self.n_features: int | None = None

    def fit(self, inputs: torch.Tensor) -> "NormedPolyCountSketch":
        if inputs.ndim != 2:
            raise ValueError("TensorSketch fit inputs must have shape [N, D].")
        self.n_features = int(inputs.shape[1])
        extended_features = self.n_features + int(self.config.coef0 != 0)
        generator = torch.Generator(device="cpu").manual_seed(self.config.sketch_seed)
        index_hash = torch.randint(
            0,
            self.config.n_components,
            (self.config.degree, extended_features),
            dtype=torch.long,
            generator=generator,
        )
        bit_hash = (
            torch.randint(
                0,
                2,
                (self.config.degree, extended_features),
                dtype=torch.int8,
                generator=generator,
            )
            * 2
            - 1
        )
        self.register_buffer("index_hash", index_hash)
        self.register_buffer("bit_hash", bit_hash)
        return self

    def _ensure_fitted(self) -> None:
        if self.n_features is None or not hasattr(self, "index_hash"):
            raise RuntimeError("TensorSketch must be fitted before use.")

    def _normalize(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norms = inputs.norm(p=2, dim=-1, keepdim=True) + self.normalization_epsilon
        return inputs / norms, norms

    def _extended_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        scaled = inputs * (self.config.gamma**0.5)
        if self.config.coef0 == 0:
            return scaled
        bias = scaled.new_full((scaled.shape[0], 1), self.config.coef0**0.5)
        return torch.cat([scaled, bias], dim=1)

    def _base_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        self._ensure_fitted()
        single = inputs.ndim == 1
        if single:
            inputs = inputs.unsqueeze(0)
        if inputs.ndim != 2:
            raise ValueError("TensorSketch inputs must have shape [D] or [N, D].")

        extended = self._extended_inputs(inputs)
        batch_size, extended_features = extended.shape
        sketches = []
        for degree_index in range(self.config.degree):
            indices = self.index_hash[degree_index, :extended_features]
            signs = self.bit_hash[degree_index, :extended_features].to(extended.dtype)
            values = extended * signs.unsqueeze(0)
            sketch = extended.new_zeros(batch_size, self.config.n_components)
            sketch.scatter_add_(
                1,
                indices.unsqueeze(0).expand(batch_size, extended_features),
                values,
            )
            sketches.append(sketch)
        sketches = torch.stack(sketches, dim=0)
        transformed = torch.fft.irfft(
            torch.prod(torch.fft.rfft(sketches, dim=-1), dim=0),
            n=self.config.n_components,
            dim=-1,
        )
        return transformed.squeeze(0) if single else transformed

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized, _ = self._normalize(inputs)
        return self._base_transform(normalized)

    def fit_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        self.fit(inputs)
        return self.transform(inputs)

    @torch.no_grad()
    def _base_vjp_batch(
        self, inputs: torch.Tensor, vector: torch.Tensor
    ) -> torch.Tensor:
        self._ensure_fitted()
        batch_size = inputs.shape[0]
        extended = self._extended_inputs(inputs)
        extended_features = extended.shape[1]
        indices = self.index_hash[:, :extended_features]
        signs = self.bit_hash[:, :extended_features].to(extended.dtype)

        sketches = []
        for degree_index in range(self.config.degree):
            values = extended * signs[degree_index].unsqueeze(0)
            sketch = extended.new_zeros(batch_size, self.config.n_components)
            sketch.scatter_add_(
                1,
                indices[degree_index]
                .unsqueeze(0)
                .expand(batch_size, extended_features),
                values,
            )
            sketches.append(sketch)
        sketches = torch.stack(sketches, dim=1)

        frequencies = torch.fft.rfft(sketches, dim=2)
        frequency_count = frequencies.shape[2]
        prefix = torch.empty_like(frequencies)
        suffix = torch.empty_like(frequencies)
        prefix[:, 0] = torch.ones(
            frequency_count,
            dtype=frequencies.dtype,
            device=frequencies.device,
        )
        suffix[:, -1] = torch.ones(
            frequency_count,
            dtype=frequencies.dtype,
            device=frequencies.device,
        )
        if self.config.degree > 1:
            prefix[:, 1:] = torch.cumprod(frequencies[:, :-1], dim=1)
            reversed_product = torch.cumprod(
                torch.flip(frequencies[:, 1:], dims=[1]), dim=1
            )
            suffix[:, :-1] = torch.flip(reversed_product, dims=[1])
        other_products = prefix * suffix
        partial_sketches = torch.fft.irfft(
            other_products,
            n=self.config.n_components,
            dim=2,
        ).to(inputs.dtype)

        vector_frequency = torch.fft.rfft(vector.to(inputs.dtype), dim=0)
        partial_frequency = torch.fft.rfft(partial_sketches, dim=2)
        correlations = torch.fft.irfft(
            torch.conj(partial_frequency) * vector_frequency.view(1, 1, -1),
            n=self.config.n_components,
            dim=2,
        ).real

        original_features = int(self.n_features)
        original_indices = indices[:, :original_features]
        original_signs = signs[:, :original_features]
        gathered = correlations.gather(
            2,
            original_indices.unsqueeze(0).expand(batch_size, -1, -1),
        )
        result = (gathered * original_signs.unsqueeze(0)).sum(dim=1)
        return (self.config.gamma**0.5) * result

    @torch.no_grad()
    def vjp(self, inputs: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        if vector.ndim != 1 or vector.numel() != self.config.n_components:
            raise ValueError(
                f"TensorSketch VJP vector must have shape [{self.config.n_components}]."
            )
        single = inputs.ndim == 1
        batched = inputs.unsqueeze(0) if single else inputs
        if batched.ndim != 2:
            raise ValueError("TensorSketch inputs must have shape [D] or [N, D].")
        normalized, norms = self._normalize(batched)
        base_vjp = self._base_vjp_batch(normalized, vector)
        projection = (normalized * base_vjp).sum(dim=1, keepdim=True)
        result = (base_vjp - normalized * projection) / norms
        return result.squeeze(0) if single else result

    def restore_hashes(
        self,
        index_hash: torch.Tensor,
        bit_hash: torch.Tensor,
        n_features: int,
    ) -> None:
        expected = (
            self.config.degree,
            int(n_features) + int(self.config.coef0 != 0),
        )
        if tuple(index_hash.shape) != expected or tuple(bit_hash.shape) != expected:
            raise ValueError(f"Invalid TensorSketch hash shapes; expected {expected}.")
        self.n_features = int(n_features)
        self.register_buffer("index_hash", index_hash.long().contiguous())
        self.register_buffer("bit_hash", bit_hash.to(torch.int8).contiguous())


class ODEKernelClassifier(nn.Module):
    """TensorSketch linear classifier used as the ODE potential."""

    def __init__(self, config: ODESteerConfig):
        super().__init__()
        self.config = config
        self.kernel = NormedPolyCountSketch(config)
        self.fitted = False
        self.density_ratio_coefficient = 1.0

    def fit(
        self, positive: torch.Tensor, negative: torch.Tensor
    ) -> "ODEKernelClassifier":
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC

        positive = positive.detach().cpu().float()
        negative = negative.detach().cpu().float()
        if positive.ndim != 2 or negative.ndim != 2:
            raise ValueError("ODESteer activations must have shape [N, D].")
        if positive.shape[1] != negative.shape[1]:
            raise ValueError("Positive and negative activation widths must match.")
        if len(positive) == 0 or len(negative) == 0:
            raise ValueError(
                "ODESteer requires both positive and negative activations."
            )

        total = len(positive) + len(negative)
        positive_prior = len(positive) / total
        self.density_ratio_coefficient = (1 - positive_prior) / positive_prior
        inputs = torch.cat([positive, negative], dim=0)
        labels = torch.cat(
            [
                torch.ones(len(positive)),
                torch.zeros(len(negative)),
            ]
        )
        features = self.kernel.fit_transform(inputs)
        if self.config.linear_classifier == "lr":
            classifier = LogisticRegression(max_iter=1000)
        else:
            classifier = LinearSVC(max_iter=1000)
        classifier.fit(features, labels)
        coefficient = torch.as_tensor(classifier.coef_.ravel(), dtype=inputs.dtype)
        intercept = torch.as_tensor(classifier.intercept_.ravel(), dtype=inputs.dtype)
        self.register_buffer("coefficient", coefficient)
        self.register_buffer("intercept", intercept)
        self.fitted = True
        return self

    @torch.no_grad()
    def gradient(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("ODESteer classifier has not been fitted.")
        # Match the reference implementation's per-call device alignment. This
        # also makes direct vector-field probes safe immediately after CPU fit.
        self.to(inputs.device)
        return (
            self.kernel.vjp(inputs, self.coefficient) * self.density_ratio_coefficient
        )

    @torch.no_grad()
    def vector_field(self, inputs: torch.Tensor) -> torch.Tensor:
        gradient = self.gradient(inputs)
        return gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-10)

    def checkpoint(self) -> dict:
        if not self.fitted:
            raise RuntimeError("Cannot checkpoint an unfitted ODESteer classifier.")
        return {
            "n_features": int(self.kernel.n_features),
            "index_hash": self.kernel.index_hash.detach().cpu(),
            "bit_hash": self.kernel.bit_hash.detach().cpu(),
            "coefficient": self.coefficient.detach().float().cpu(),
            "intercept": self.intercept.detach().float().cpu(),
            "density_ratio_coefficient": float(self.density_ratio_coefficient),
        }

    def restore(self, payload: dict) -> None:
        self.kernel.restore_hashes(
            payload["index_hash"],
            payload["bit_hash"],
            int(payload["n_features"]),
        )
        coefficient = torch.as_tensor(payload["coefficient"]).float().reshape(-1)
        intercept = torch.as_tensor(payload["intercept"]).float().reshape(-1)
        if coefficient.numel() != self.config.n_components:
            raise ValueError("ODESteer checkpoint coefficient width is invalid.")
        if intercept.numel() != 1:
            raise ValueError("ODESteer checkpoint intercept must be scalar.")
        ratio = float(payload["density_ratio_coefficient"])
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("ODESteer checkpoint density ratio is invalid.")
        if not torch.isfinite(coefficient).all() or not torch.isfinite(intercept).all():
            raise ValueError("ODESteer checkpoint contains non-finite values.")
        self.register_buffer("coefficient", coefficient)
        self.register_buffer("intercept", intercept)
        self.density_ratio_coefficient = ratio
        self.fitted = True


@torch.no_grad()
def ode_transport(
    states: torch.Tensor,
    strengths: torch.Tensor,
    classifier: ODEKernelClassifier,
    *,
    solver: str,
    steps: int,
) -> torch.Tensor:
    """Transport a batch, preserving the official scalar-strength path exactly."""

    strengths = torch.as_tensor(
        strengths, device=states.device, dtype=torch.float32
    ).reshape(-1)
    if strengths.numel() != states.shape[0]:
        raise ValueError("Each ODESteer state requires one steering strength.")
    if not torch.isfinite(strengths).all():
        raise ValueError("ODESteer strengths must be finite.")
    if not strengths.ne(0).any():
        return states

    classifier.to(states.device)
    if torch.equal(strengths, strengths[:1].expand_as(strengths)):
        strength = float(strengths[0])
        if strength == 0:
            return states
        times = torch.tensor([0.0, strength], device=states.device)
        return odeint(
            func=lambda _time, current: classifier.vector_field(current),
            y0=states,
            t=times,
            method=solver,
            options={"step_size": strength / steps},
        )[1]

    times = torch.tensor([0.0, 1.0], device=states.device)
    scales = strengths.unsqueeze(-1)
    return odeint(
        func=lambda _time, current: scales * classifier.vector_field(current),
        y0=states,
        t=times,
        method=solver,
        options={"step_size": 1.0 / steps},
    )[1]


@torch.no_grad()
def step_ode_transport(
    states: torch.Tensor,
    strengths: torch.Tensor,
    classifier: ODEKernelClassifier,
) -> torch.Tensor:
    strengths = torch.as_tensor(
        strengths, device=states.device, dtype=torch.float32
    ).reshape(-1)
    if strengths.numel() != states.shape[0]:
        raise ValueError("Each StepODESteer state requires one steering strength.")
    if not torch.isfinite(strengths).all():
        raise ValueError("StepODESteer strengths must be finite.")
    if not strengths.ne(0).any():
        return states
    classifier.to(states.device)
    return states + strengths.unsqueeze(-1) * classifier.vector_field(states)


@torch.no_grad()
def apply_ode_steering(
    hidden_states: torch.Tensor,
    strengths: torch.Tensor,
    token_mask: torch.Tensor,
    classifier: ODEKernelClassifier,
    *,
    variant: str,
    solver: str,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a state-dependent update only at selected token positions."""

    strengths = strengths.to(hidden_states.device, torch.float32).reshape(-1)
    token_mask = token_mask.to(hidden_states.device).bool()
    if strengths.numel() != hidden_states.shape[0]:
        raise ValueError("ODESteer strength batch does not match hidden states.")
    if token_mask.shape != hidden_states.shape[:2]:
        raise ValueError("ODESteer token mask does not match hidden states.")
    active = token_mask & strengths.ne(0).unsqueeze(1)
    if not active.any():
        return hidden_states, active

    rows = (
        torch.arange(hidden_states.shape[0], device=hidden_states.device)
        .unsqueeze(1)
        .expand(hidden_states.shape[:2])
    )
    active_rows = rows[active]
    selected = hidden_states[active].float()
    selected_strengths = strengths[active_rows]
    if variant == "ode":
        steered = ode_transport(
            selected,
            selected_strengths,
            classifier,
            solver=solver,
            steps=steps,
        )
    elif variant == "step":
        steered = step_ode_transport(selected, selected_strengths, classifier)
    else:
        raise ValueError(f"Unknown ODESteer variant {variant!r}.")
    output = hidden_states.clone()
    output[active] = steered.to(hidden_states.dtype)
    return output, active


class _ODESteerBase(Model):
    requires_mean_activations = False
    requires_calibration_scale = False
    uses_intervention_positions = False
    load_trained_weights = True
    inference_instance_scope = "per_concept"
    variant = "base"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier: ODEKernelClassifier | None = None

    @classmethod
    def training_fingerprint_context(cls):
        return {
            "artifact_version": 1,
            "source_commit": "8a3c481d6493ecb3325eea5ef9c448cccfced7eb",
            "potential": "normalized_polynomial_tensorsketch_linear_classifier",
            "vector_field": "unit_normalized_density_ratio_gradient",
            "activation": "last_real_token_of_complete_positive_or_negative_sequence",
            "intervention": "last_prompt_token_then_each_cached_generation_token",
            "variant": cls.variant,
            "factor_semantics": "ODE_terminal_time_T",
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
            f"{model.__class__.__name__} does not expose a supported decoder layer stack."
        )

    def _config(self) -> ODESteerConfig:
        if self.training_args is None:
            raise ValueError(f"{self.__class__.__name__} requires training arguments.")
        config = ODESteerConfig(
            degree=int(getattr(self.training_args, "ode_degree", 2)),
            n_components=int(getattr(self.training_args, "ode_n_components", 8000)),
            gamma=float(getattr(self.training_args, "ode_gamma", 0.1)),
            coef0=float(getattr(self.training_args, "ode_coef0", 1.0)),
            linear_classifier=str(
                getattr(self.training_args, "ode_linear_classifier", "lr")
            ),
            sketch_seed=int(getattr(self.training_args, "ode_sketch_seed", 42)),
            solver=str(getattr(self.training_args, "ode_solver", "euler")),
            steps=int(getattr(self.training_args, "ode_steps", 10)),
        )
        config.validate()
        return config

    def make_model(self, **kwargs):
        decoder_layers = self._decoder_layers(self.model)
        if self.layer < 0 or self.layer >= len(decoder_layers):
            raise ValueError(
                f"Invalid {self.__class__.__name__} layer {self.layer}; model "
                f"has {len(decoder_layers)} layers."
            )
        self.classifier = ODEKernelClassifier(self._config())
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

        handle = self._decoder_layers(self.model)[self.layer].register_forward_hook(
            hook
        )
        try:
            yield captured
        finally:
            handle.remove()

    @torch.no_grad()
    def train(self, examples: pd.DataFrame, **kwargs):
        if "labels" not in examples:
            raise KeyError(
                f"{self.__class__.__name__} requires binarize_dataset: true."
            )
        labels = {int(value) for value in examples["labels"].dropna().unique()}
        if labels != {0, 1}:
            raise ValueError(f"{self.__class__.__name__} requires both binary classes.")
        if self.classifier is None:
            self.make_model()

        self.model.eval()
        batch_size = int(getattr(self.training_args, "batch_size", 32) or 32)
        if batch_size < 1:
            raise ValueError("ODESteer batch_size must be positive.")
        positive_activations = []
        negative_activations = []
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
                batch = examples.iloc[start : start + batch_size]
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
                    raise RuntimeError("ODESteer failed to capture the target layer.")
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
        self.classifier.fit(positive, negative)
        logger.warning(
            "%s fitted its potential from %s positive and %s negative "
            "activations at layer %s.",
            self.__class__.__name__,
            len(positive),
            len(negative),
            self.layer,
        )

    def save(self, dump_dir, **kwargs):
        if self.classifier is None or not self.classifier.fitted:
            raise RuntimeError(f"{self.__class__.__name__} has no fitted classifier.")
        concept_id = int(kwargs.get("concept_id", 0))
        artifact_dir = Path(dump_dir) / self.artifact_directory / str(concept_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "method": str(self),
                "variant": self.variant,
                "layer": int(self.layer),
                "hidden_size": int(self.model.config.hidden_size),
                "config": asdict(self._config()),
                "classifier": self.classifier.checkpoint(),
            },
            artifact_dir / "classifier.pt",
        )

    def load(self, dump_dir=None, **kwargs):
        concept_id = int(kwargs.get("concept_id", 0))
        checkpoint_path = (
            Path(dump_dir) / self.artifact_directory / str(concept_id) / "classifier.pt"
        )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if int(payload.get("format_version", -1)) != 1:
            raise ValueError("Unsupported ODESteer checkpoint format.")
        if payload.get("variant") != self.variant:
            raise ValueError(
                f"Checkpoint variant {payload.get('variant')!r} does not match "
                f"{self.variant!r}."
            )
        if int(payload.get("layer", -1)) != int(self.layer):
            raise ValueError("ODESteer checkpoint layer does not match evaluation.")
        if int(payload.get("hidden_size", -1)) != int(self.model.config.hidden_size):
            raise ValueError(
                "ODESteer checkpoint hidden size does not match the model."
            )
        expected_config = asdict(self._config())
        if payload.get("config") != expected_config:
            raise ValueError(
                "ODESteer checkpoint configuration does not match the current "
                "training configuration."
            )
        self.make_model()
        self.classifier.restore(payload["classifier"])
        self.classifier.to(self.device)

    @staticmethod
    def _strengths(batch_examples: pd.DataFrame, device) -> torch.Tensor:
        strengths = torch.as_tensor(
            batch_examples["factor"].tolist(),
            device=device,
            dtype=torch.float32,
        )
        if not torch.isfinite(strengths).all():
            raise ValueError("ODESteer factors must be finite.")
        return strengths

    @contextmanager
    def _intervention(self, strengths, token_mask):
        if self.classifier is None or not self.classifier.fitted:
            raise RuntimeError("ODESteer must be fitted or loaded before inference.")
        config = self._config()

        def hook(_module, _inputs, output):
            hidden = self._hidden_from_output(output)
            current_mask = token_mask
            if current_mask.shape != hidden.shape[:2]:
                if hidden.shape[1] != 1 or hidden.shape[0] != current_mask.shape[0]:
                    raise ValueError(
                        "ODESteer token mask does not align with hidden states."
                    )
                current_mask = torch.ones(
                    hidden.shape[:2], device=hidden.device, dtype=torch.bool
                )
            steered, _ = apply_ode_steering(
                hidden,
                strengths.to(hidden.device),
                current_mask.to(hidden.device),
                self.classifier,
                variant=self.variant,
                solver=config.solver,
                steps=config.steps,
            )
            if steered is hidden:
                return output
            return self._replace_hidden(output, steered)

        handle = self._decoder_layers(self.model)[self.layer].register_forward_hook(
            hook
        )
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
                batch = examples.iloc[start : start + batch_size]
                input_column = "steered_input" if kwargs.get("use_synergy") else "input"
                inputs = self.tokenizer(
                    batch[input_column].tolist(),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                strengths = self._strengths(batch, self.device)
                token_mask = self._last_token_mask(inputs.attention_mask)
                with self._intervention(strengths, token_mask):
                    output_ids = self.model.generate(**inputs, **generation_kwargs)
                prompt_width = inputs.input_ids.shape[1]
                generations.extend(
                    self.tokenizer.batch_decode(
                        output_ids[:, prompt_width:], skip_special_tokens=True
                    )
                )
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
        if self.classifier is None or not self.classifier.fitted:
            raise RuntimeError("ODESteer must be fitted or loaded before inference.")
        self.model.eval()

    def choice_forward(self, inputs, batch_examples, **kwargs):
        strengths = self._strengths(batch_examples, self.device)
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
                    raise ValueError("ODESteer choice candidate has no prompt tokens.")
                intervention_start = real_positions[-candidate_length] - 1
                token_mask[row_index] = attention_mask[row_index].bool() & (
                    positions >= intervention_start
                )
        else:
            token_mask = self._last_token_mask(attention_mask)

        model_inputs = self.choice_model_inputs(
            inputs,
            last_token_only=not kwargs.get("full_sequence", False),
            logits_to_keep=kwargs.get("choice_logits_to_keep"),
        )
        with self._intervention(strengths, token_mask):
            outputs = self.model(**model_inputs, use_cache=False)
        return outputs, strengths

    def to(self, device):
        super().to(device)
        if self.classifier is not None:
            self.classifier.to(device)
        return self


class ODESteer(_ODESteerBase):
    """Full fixed-step ODE transport from arXiv:2602.17560."""

    artifact_directory = "odesteer"
    variant = "ode"

    def __str__(self):
        return "ODESteer"


class StepODESteer(_ODESteerBase):
    """The paper's one-step ODE baseline using the same learned field."""

    artifact_directory = "step_odesteer"
    variant = "step"

    def __str__(self):
        return "StepODESteer"
