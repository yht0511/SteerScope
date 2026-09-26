import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import (
    ProxyHandler,
    Request,
    build_opener,
)

import pandas as pd
import torch


EASYSTEER_METHODS = frozenset({
    "DiffMean",
    "PCA",
    "LAT",
    "Random",
    "RandomOriginal",
    "LinearProbe",
    "SteeringVector",
    "LsReFT",
    "GemmaScopeSAE",
    "GemmaScopeSAEMaxAUC",
})

_SAE_METHODS = frozenset({
    "GemmaScopeSAE",
    "GemmaScopeSAEMaxAUC",
})


class EasySteerError(RuntimeError):
    """Base exception for the optional EasySteer inference backend."""


class EasySteerUnsupported(EasySteerError):
    """The request cannot preserve the legacy inference semantics."""


class EasySteerUnavailable(EasySteerError):
    """The configured EasySteer service is unavailable or returned an error."""


@dataclass(frozen=True)
class EasySteerVectorSpec:
    path: Path
    scale: float
    target_layers: tuple[int, ...]
    algorithm: str = "direct"
    normalize: bool = False


class EasySteerVectorResolver:
    """Resolve SteerScope checkpoints into per-concept EasySteer vectors."""

    def __init__(
        self,
        cache_dir: str | Path,
        default_checkpoint_dir: str | Path,
        master_data_dir: str | Path,
    ):
        self.cache_dir = Path(cache_dir)
        self.default_checkpoint_dir = Path(default_checkpoint_dir)
        self.master_data_dir = Path(master_data_dir)

    def resolve(
        self,
        target,
        *,
        factor: float,
        target_layers: list[int] | tuple[int, ...],
        intervention_type: str,
        disable_neuronpedia_max_act: bool,
    ) -> EasySteerVectorSpec:
        if target.method not in EASYSTEER_METHODS:
            raise EasySteerUnsupported(
                f"Method '{target.method}' is not enabled for EasySteer."
            )
        if (intervention_type or "addition") != "addition":
            raise EasySteerUnsupported(
                "EasySteer currently preserves the legacy semantics only for "
                "steering_intervention_type='addition'."
            )
        if not math.isfinite(float(factor)):
            raise EasySteerUnsupported(
                f"Steering factor must be finite, got {factor!r}."
            )
        layers = tuple(int(layer) for layer in target_layers)
        if not layers:
            raise EasySteerUnsupported(
                "EasySteer requires at least one steering layer."
            )

        checkpoint_dir = self._checkpoint_dir(target)
        vector = self._load_vector(target, checkpoint_dir)
        calibration_scale = self._calibration_scale(
            target,
            checkpoint_dir,
            disable_neuronpedia_max_act=disable_neuronpedia_max_act,
        )
        path = self._export_vector(
            vector,
            method=target.method,
            concept_id=int(target.concept.concept_id),
        )
        return EasySteerVectorSpec(
            path=path,
            scale=float(factor) * calibration_scale,
            target_layers=layers,
        )

    def _checkpoint_dir(self, target) -> Path:
        if target.artifact.kind == "steering_vector":
            return Path(target.artifact.path).parent
        if target.artifact.path is not None:
            return Path(target.artifact.path)
        return self.default_checkpoint_dir

    def _load_vector(self, target, checkpoint_dir: Path) -> torch.Tensor:
        concept_id = int(target.concept.concept_id)
        if target.artifact.kind == "steering_vector":
            source = Path(target.artifact.path)
            raw = self._torch_load(source)
            vector = self._unwrap_tensor(raw, source)
            return self._select_vector_row(
                vector,
                concept_id=concept_id,
                source=source,
                standalone=True,
            )

        if target.method in _SAE_METHODS:
            source = checkpoint_dir / f"{target.method}.pt"
            raw = self._torch_load(source)
            if not isinstance(raw, dict) or "W_dec" not in raw:
                raise EasySteerUnsupported(
                    f"SAE checkpoint {source} does not contain W_dec."
                )
            vector = torch.as_tensor(raw["W_dec"])
        else:
            source = checkpoint_dir / f"{target.method}_weight.pt"
            vector = self._unwrap_tensor(self._torch_load(source), source)
        return self._select_vector_row(
            vector,
            concept_id=concept_id,
            source=source,
            standalone=False,
        )

    def _calibration_scale(
        self,
        target,
        checkpoint_dir: Path,
        *,
        disable_neuronpedia_max_act: bool,
    ) -> float:
        if target.artifact.kind == "steering_vector":
            return 1.0
        if target.method in _SAE_METHODS:
            if disable_neuronpedia_max_act:
                raise EasySteerUnsupported(
                    "SAE dataset-derived max activations require the legacy "
                    "model runtime. Disable EasySteer for this evaluator or set "
                    "disable_neuronpedia_max_act=false."
                )
            return self._sae_calibration_scale(target, checkpoint_dir)

        source = checkpoint_dir / f"{target.method}_scale.pt"
        raw = self._torch_load(source)
        scales = torch.as_tensor(raw).detach().cpu().reshape(-1)
        concept_id = int(target.concept.concept_id)
        if concept_id >= scales.numel():
            raise EasySteerUnsupported(
                f"Calibration scale {source} has {scales.numel()} entries and "
                f"cannot select concept ID {concept_id}."
            )
        return self._positive_scale(float(scales[concept_id]), source)

    def _sae_calibration_scale(self, target, checkpoint_dir: Path) -> float:
        scales_path = checkpoint_dir / f"{target.method}_scale.pt"
        scales = torch.as_tensor(self._torch_load(scales_path)).reshape(-1)
        concept_id = int(target.concept.concept_id)
        if concept_id >= scales.numel():
            raise EasySteerUnsupported(
                f"SAE calibration scale {scales_path} has {scales.numel()} "
                f"entries and cannot select concept ID {concept_id}."
            )
        return self._positive_scale(float(scales[concept_id]), scales_path)

    @staticmethod
    def _torch_load(path: Path) -> Any:
        if not path.is_file():
            raise EasySteerUnsupported(f"Checkpoint file not found: {path}")
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise EasySteerUnsupported(
                f"Cannot load EasySteer vector data from {path}: {error}"
            ) from error

    @staticmethod
    def _unwrap_tensor(raw: Any, source: Path) -> torch.Tensor:
        if isinstance(raw, dict):
            for key in ("steering_vector", "vector", "weight"):
                if key in raw:
                    raw = raw[key]
                    break
        try:
            return torch.as_tensor(raw)
        except (TypeError, ValueError) as error:
            raise EasySteerUnsupported(
                f"Checkpoint {source} does not contain a tensor vector."
            ) from error

    @staticmethod
    def _select_vector_row(
        tensor: torch.Tensor,
        *,
        concept_id: int,
        source: Path,
        standalone: bool,
    ) -> torch.Tensor:
        tensor = tensor.detach().cpu()
        if tensor.ndim == 1:
            vector = tensor
        elif tensor.ndim == 2:
            if standalone and tensor.shape[0] == 1:
                vector = tensor[0]
            elif concept_id < tensor.shape[0]:
                vector = tensor[concept_id]
            else:
                raise EasySteerUnsupported(
                    f"Checkpoint {source} has {tensor.shape[0]} rows and cannot "
                    f"select concept ID {concept_id}."
                )
        else:
            raise EasySteerUnsupported(
                f"Checkpoint {source} must contain [hidden_size] or "
                f"[num_concepts, hidden_size], got {tuple(tensor.shape)}."
            )
        vector = vector.float().contiguous()
        if not torch.isfinite(vector).all():
            raise EasySteerUnsupported(
                f"Checkpoint {source} contains a non-finite steering vector."
            )
        return vector

    def _export_vector(
        self,
        vector: torch.Tensor,
        *,
        method: str,
        concept_id: int,
    ) -> Path:
        digest = hashlib.sha256(vector.numpy().tobytes()).hexdigest()[:16]
        safe_method = "".join(
            character if character.isalnum() else "-"
            for character in method
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        destination = (
            self.cache_dir
            / f"{safe_method}-concept-{concept_id}-{digest}.pt"
        )
        if destination.is_file():
            return destination.resolve()
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=self.cache_dir,
            suffix=".pt.tmp",
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(vector, temporary_path)
            os.replace(temporary_path, destination)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return destination.resolve()

    @staticmethod
    def _positive_scale(value: float, source: Path) -> float:
        if not math.isfinite(value) or value <= 0:
            raise EasySteerUnsupported(
                f"Calibration scale from {source} must be finite and positive, "
                f"got {value!r}."
            )
        return value


class EasySteerRuntime:
    """OpenAI-compatible client for an externally managed EasySteer server."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        resolver: EasySteerVectorResolver,
        timeout: float = 120.0,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.model_name = model_name
        self.resolver = resolver
        self.timeout = float(timeout)
        self._healthy = False
        # EasySteer is normally local. Do not accidentally send local prompts
        # through the shell's HTTP proxy.
        self._opener = build_opener(ProxyHandler({}))

    def ensure_ready(self) -> None:
        if self._healthy:
            return
        response = self._json_request("GET", "/v1/models")
        model_ids = {
            str(item.get("id"))
            for item in response.get("data", [])
            if isinstance(item, dict)
        }
        if self.model_name not in model_ids:
            raise EasySteerUnavailable(
                f"EasySteer at {self.base_url} serves {sorted(model_ids)}, not "
                f"the requested model '{self.model_name}'."
            )
        self._healthy = True

    def predict(
        self,
        target,
        examples: pd.DataFrame,
        *,
        target_layers: list[int] | tuple[int, ...],
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        seed: int,
        intervention_type: str,
        intervene_on_prompt: bool,
        disable_neuronpedia_max_act: bool,
        compute_perplexity: bool,
    ) -> dict[str, list[Any]]:
        if compute_perplexity:
            raise EasySteerUnsupported(
                "Common-model perplexity scoring still requires the legacy "
                "runtime."
            )
        inference_modes = {
            str(value)
            for value in examples.get(
                "inference_mode", pd.Series(dtype=str)
            ).dropna()
            if str(value)
        }
        if inference_modes:
            raise EasySteerUnsupported(
                "Choice-logit inference is not yet implemented by the "
                f"EasySteer runtime: {sorted(inference_modes)}."
            )
        if "input" not in examples:
            raise EasySteerUnsupported(
                "EasySteer generation requires an 'input' prompt column."
            )
        factor_column = (
            "model_factor" if "model_factor" in examples else "factor"
        )
        if factor_column not in examples:
            raise EasySteerUnsupported(
                "EasySteer generation requires 'factor' or 'model_factor'."
            )
        factors = {
            float(value) for value in examples[factor_column].tolist()
        }
        if len(factors) != 1:
            raise EasySteerUnsupported(
                "One EasySteer target request must contain exactly one factor, "
                f"got {sorted(factors)}."
            )
        factor = next(iter(factors))
        spec = self.resolver.resolve(
            target,
            factor=factor,
            target_layers=target_layers,
            intervention_type=intervention_type,
            disable_neuronpedia_max_act=disable_neuronpedia_max_act,
        )
        self.ensure_ready()

        prompts = examples["input"].tolist()
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise EasySteerUnsupported(
                "Every EasySteer input prompt must be a string."
            )
        generations = []
        request_batch_size = max(1, int(batch_size))
        for start in range(0, len(prompts), request_batch_size):
            batch_prompts = prompts[start:start + request_batch_size]
            request = {
                "model": self.model_name,
                "prompt": batch_prompts,
                "max_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "top_k": 50,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "truncate_prompt_tokens": 1024,
                "seed": int(seed) + start,
                "stream": False,
                "steer_vector_request": self._steer_request(
                    target,
                    factor=factor,
                    spec=spec,
                    intervene_on_prompt=intervene_on_prompt,
                ),
            }
            response = self._json_request(
                "POST", "/v1/completions", request
            )
            choices = response.get("choices")
            if not isinstance(choices, list):
                raise EasySteerUnavailable(
                    f"EasySteer response has no choices: {response!r}"
                )
            ordered = sorted(
                choices, key=lambda choice: int(choice.get("index", 0))
            )
            batch_generations = [
                choice.get("text") for choice in ordered
            ]
            if (
                len(batch_generations) != len(batch_prompts)
                or any(not isinstance(text, str) for text in batch_generations)
            ):
                raise EasySteerUnavailable(
                    "EasySteer returned a different number of generations than "
                    f"prompts ({len(batch_generations)} vs "
                    f"{len(batch_prompts)})."
                )
            generations.extend(batch_generations)
        return {
            "steered_generation": generations,
            "strength": [spec.scale] * len(generations),
        }

    @staticmethod
    def _steer_request(
        target,
        *,
        factor: float,
        spec: EasySteerVectorSpec,
        intervene_on_prompt: bool,
    ) -> dict[str, Any]:
        identity = json.dumps({
            "path": str(spec.path),
            "scale": spec.scale,
            "layers": spec.target_layers,
            "algorithm": spec.algorithm,
            "normalize": spec.normalize,
            "prefill": bool(intervene_on_prompt),
        }, sort_keys=True)
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        request_id = int(digest[:8], 16) % 2_147_483_646 + 1
        name = (
            f"steerscope-{target.method}-c{target.concept.concept_id}-"
            f"f{factor:g}-{digest[:8]}"
        )
        common = {
            "path": str(spec.path),
            "scale": spec.scale,
            "prefill_trigger_tokens": [-1] if intervene_on_prompt else None,
            "generate_trigger_tokens": [-1],
            "algorithm": spec.algorithm,
            "normalize": spec.normalize,
        }
        if len(spec.target_layers) == 1:
            return {
                "steer_vector_name": name,
                "steer_vector_int_id": request_id,
                "steer_vector_local_path": str(spec.path),
                "scale": spec.scale,
                "target_layers": list(spec.target_layers),
                "prefill_trigger_tokens": common[
                    "prefill_trigger_tokens"
                ],
                "generate_trigger_tokens": [-1],
                "algorithm": spec.algorithm,
                "normalize": spec.normalize,
            }
        return {
            "steer_vector_name": name,
            "steer_vector_int_id": request_id,
            "vector_configs": [
                {
                    **common,
                    "target_layers": [layer],
                }
                for layer in spec.target_layers
            ],
        }

    def _json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = (
            json.dumps(payload).encode("utf-8")
            if payload is not None
            else None
        )
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._opener.open(
                request, timeout=self.timeout
            ) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise EasySteerUnavailable(
                f"EasySteer {method} {path} returned HTTP {error.code}: {body}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise EasySteerUnavailable(
                f"Cannot reach EasySteer at {self.base_url}: {error}"
            ) from error
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as error:
            raise EasySteerUnavailable(
                f"EasySteer returned invalid JSON for {path}: {body[:500]!r}"
            ) from error
        if not isinstance(parsed, dict):
            raise EasySteerUnavailable(
                f"EasySteer returned a non-object response for {path}."
            )
        if parsed.get("error"):
            raise EasySteerUnavailable(
                f"EasySteer returned an error for {path}: {parsed['error']}"
            )
        return parsed
