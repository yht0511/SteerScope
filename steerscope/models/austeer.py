"""AUSteer using the released FFN atomic-unit ranking and multiplicative intervention."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from .model import Model


logger = logging.getLogger(__name__)


def signed_consistency_scores(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
) -> torch.Tensor:
    """Return the official signed AU consistency score for each coordinate."""

    if positive_activations.shape != negative_activations.shape:
        raise ValueError("AUSteer positive and negative activations must align.")
    if positive_activations.ndim < 2 or positive_activations.shape[0] == 0:
        raise ValueError("AUSteer requires at least one paired activation.")
    gains = positive_activations.float() - negative_activations.float()
    larger = (gains > 0).sum(dim=0)
    smaller = (gains < 0).sum(dim=0)
    scores = torch.maximum(larger, smaller).float() / gains.shape[0]
    return torch.where(larger < smaller, -scores, scores)


def select_atomic_units(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Select AUs with the same stable absolute-score ordering as ``set_MFU``."""

    if scores.ndim != 2:
        raise ValueError("AUSteer AU scores must have shape [layers, hidden].")
    if topk < 1:
        raise ValueError("austeer_topk must be positive.")
    if topk > scores.numel():
        raise ValueError(
            f"austeer_topk={topk} exceeds the {scores.numel()} available AUs."
        )
    if not torch.isfinite(scores).all():
        raise ValueError("AUSteer AU scores must be finite.")

    flat_scores = scores.detach().float().cpu().reshape(-1)
    # Python's sort is stable.  This reproduces the official tie order, whose
    # flattened traversal is layer-major and then coordinate-major.
    order = sorted(
        range(flat_scores.numel()),
        key=lambda index: abs(float(flat_scores[index])),
        reverse=True,
    )
    flat_mask = torch.zeros_like(flat_scores)
    nonzero = 0
    for index in order:
        score = flat_scores[index]
        flat_mask[index] = score
        if score != 0:
            nonzero += 1
        if nonzero >= topk:
            break
    return flat_mask.reshape_as(scores)


class AUSteer(Model):
    """Official coordinate-level AUSteer localization and FFN intervention."""

    requires_mean_activations = False
    requires_calibration_scale = False
    uses_intervention_positions = False
    load_trained_weights = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.au_masks: torch.Tensor | None = None

    def __str__(self):
        return "AUSteer"

    @classmethod
    def training_fingerprint_context(cls):
        return {
            "artifact_version": 1,
            "localization": "paired_last_token_signed_consistency",
            "atomic_unit": "ffn_output_coordinate_window_1",
            "selection": "global_stable_abs_topk",
            "intervention": "ffn_output_times_one_plus_alpha_beta_all_tokens",
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

    def _ffn_modules(self):
        modules = []
        for layer_index, layer in enumerate(self._decoder_layers(self.model)):
            module = getattr(layer, "mlp", None)
            if module is None:
                raise TypeError(
                    f"Decoder layer {layer_index} does not expose an MLP module; "
                    "the integrated AUSteer variant requires layer.mlp outputs."
                )
            modules.append(module)
        return modules

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

    def _topk(self) -> int:
        if self.training_args is None:
            raise ValueError("AUSteer requires its training configuration.")
        value = int(getattr(self.training_args, "austeer_topk", 10) or 10)
        if value < 1:
            raise ValueError("austeer_topk must be positive.")
        return value

    def make_model(self, **kwargs):
        modules = self._ffn_modules()
        hidden_size = int(self.model.config.hidden_size)
        topk = self._topk()
        if topk > len(modules) * hidden_size:
            raise ValueError(
                f"austeer_topk={topk} exceeds the "
                f"{len(modules) * hidden_size} available FFN AUs."
            )
        num_masks = int(kwargs.get("low_rank_dimension", 1) or 1)
        if num_masks < 1:
            raise ValueError("AUSteer requires at least one AU mask row.")
        self.au_masks = torch.zeros(
            num_masks,
            len(modules),
            hidden_size,
            device=self.device,
            dtype=torch.float32,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _paired_examples(examples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        required = {"input", "labels", "pair_id"}
        missing = required - set(examples.columns)
        if missing:
            raise KeyError(
                "AUSteer requires binarize_dataset: true and paired data; "
                f"missing columns: {sorted(missing)}."
            )
        if examples["pair_id"].isna().any():
            raise ValueError("AUSteer pair_id values cannot be missing.")
        labels = set(int(value) for value in examples["labels"].dropna().unique())
        if labels != {0, 1}:
            raise ValueError("AUSteer requires both positive and negative examples.")

        positive = examples[examples["labels"].astype(int) == 1].copy()
        negative = examples[examples["labels"].astype(int) == 0].copy()
        if positive["pair_id"].duplicated().any() or negative["pair_id"].duplicated().any():
            raise ValueError(
                "AUSteer requires exactly one positive and one negative row per pair_id."
            )
        positive_ids = positive["pair_id"].tolist()
        if set(positive_ids) != set(negative["pair_id"].tolist()):
            raise ValueError(
                "AUSteer positive and negative rows must have identical pair_id sets."
            )
        negative = negative.set_index("pair_id", drop=False).loc[positive_ids]
        return positive.reset_index(drop=True), negative.reset_index(drop=True)

    @contextmanager
    def _capture_all_ffn(self, last_positions: torch.Tensor):
        captured = [None] * len(self._ffn_modules())
        handles = []

        for layer_index, module in enumerate(self._ffn_modules()):
            def hook(_module, _inputs, output, layer_index=layer_index):
                hidden = self._hidden_from_output(output)
                if hidden.ndim != 3:
                    raise ValueError(
                        "AUSteer expects FFN outputs with shape [batch, sequence, hidden]."
                    )
                positions = last_positions.to(hidden.device)
                rows = torch.arange(hidden.shape[0], device=hidden.device)
                captured[layer_index] = (
                    hidden[rows, positions].detach().float().cpu()
                )

            handles.append(module.register_forward_hook(hook))
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def _capture_batch(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        last_positions = self._last_real_positions(inputs.attention_mask)
        with self._capture_all_ffn(last_positions) as captured:
            self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                use_cache=False,
                return_dict=True,
            )
        if any(value is None for value in captured):
            missing = [index for index, value in enumerate(captured) if value is None]
            raise RuntimeError(
                f"AUSteer failed to capture FFN outputs at layers {missing}."
            )
        return torch.stack(captured, dim=1)

    @torch.no_grad()
    def train(self, examples: pd.DataFrame, **kwargs):
        positive, negative = self._paired_examples(examples)
        if self.au_masks is None:
            self.make_model()
        self.model.eval()

        batch_size = int(getattr(self.training_args, "batch_size", 32) or 32)
        if batch_size < 1:
            raise ValueError("AUSteer batch_size must be positive.")
        num_layers = len(self._ffn_modules())
        hidden_size = int(self.model.config.hidden_size)
        larger = torch.zeros(num_layers, hidden_size, dtype=torch.int64)
        smaller = torch.zeros_like(larger)
        pair_count = 0
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        progress = tqdm(
            range(0, len(positive), batch_size),
            position=self.process_rank,
            leave=True,
            desc=str(self),
        )
        try:
            for start in range(0, len(positive), batch_size):
                positive_batch = positive.iloc[start:start + batch_size]
                negative_batch = negative.iloc[start:start + batch_size]
                positive_activations = self._capture_batch(
                    positive_batch["input"].tolist()
                )
                negative_activations = self._capture_batch(
                    negative_batch["input"].tolist()
                )
                gains = positive_activations - negative_activations
                larger += (gains > 0).sum(dim=0)
                smaller += (gains < 0).sum(dim=0)
                pair_count += gains.shape[0]
                progress.update(1)
        finally:
            progress.close()
            self.tokenizer.padding_side = original_padding_side

        scores = torch.maximum(larger, smaller).float() / pair_count
        scores = torch.where(larger < smaller, -scores, scores)
        mask = select_atomic_units(scores, self._topk())
        self.au_masks[0].copy_(mask.to(self.device))
        selected = int(torch.count_nonzero(mask))
        logger.warning(
            "AUSteer localized %s/%s FFN AUs from %s paired sequences across "
            "%s layers (requested k=%s).",
            selected,
            mask.numel(),
            pair_count,
            num_layers,
            self._topk(),
        )

    def save(self, dump_dir, **kwargs):
        if self.au_masks is None:
            raise RuntimeError("AUSteer has no learned AU mask to save.")
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        model_name = kwargs.get("model_name", str(self))
        flattened = self.au_masks.detach().float().cpu().flatten(start_dim=1)
        torch.save(flattened, dump_dir / f"{model_name}_weight.pt")
        torch.save(
            torch.zeros(flattened.shape[0], dtype=torch.float32),
            dump_dir / f"{model_name}_bias.pt",
        )

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", str(self))
        concept_id = int(kwargs.get("concept_id", 0))
        priority_mode = kwargs.get("priority_mode", "compute_priority")
        weight = torch.load(
            Path(dump_dir) / f"{model_name}_weight.pt",
            map_location="cpu",
            weights_only=True,
        )
        bias = torch.load(
            Path(dump_dir) / f"{model_name}_bias.pt",
            map_location="cpu",
            weights_only=True,
        )
        if weight.ndim != 2:
            raise ValueError("AUSteer checkpoint weights must be rank two.")
        if bias.reshape(-1).numel() != weight.shape[0]:
            raise ValueError("AUSteer checkpoint weight and bias rows do not match.")
        expected_width = len(self._ffn_modules()) * int(self.model.config.hidden_size)
        if weight.shape[1] != expected_width:
            raise ValueError(
                f"AUSteer checkpoint width {weight.shape[1]} does not match "
                f"{expected_width} FFN AUs in the base model."
            )
        if not torch.isfinite(weight).all():
            raise ValueError("AUSteer checkpoint contains non-finite AU scores.")

        if priority_mode == "mem_priority":
            if concept_id < 0 or concept_id >= weight.shape[0]:
                raise IndexError(
                    f"AUSteer checkpoint has {weight.shape[0]} concepts and cannot "
                    f"select concept ID {concept_id}."
                )
            weight = weight[concept_id:concept_id + 1]
            self.concept_id_map = {concept_id: 0}
        self.make_model(low_rank_dimension=weight.shape[0])
        self.au_masks.copy_(weight.reshape_as(self.au_masks).to(self.device))

    def _mask_indices(self, concept_ids) -> torch.Tensor:
        if self.au_masks is None:
            raise RuntimeError("AUSteer AU masks have not been initialized.")
        concept_ids = [int(value) for value in concept_ids]
        if self.concept_id_map is not None:
            concept_ids = [self.concept_id_map[value] for value in concept_ids]
        rows = self.au_masks.shape[0]
        indices = [0] * len(concept_ids) if rows == 1 else concept_ids
        if indices and (min(indices) < 0 or max(indices) >= rows):
            raise IndexError(
                f"AUSteer checkpoint has {rows} mask rows but received concept "
                f"IDs {sorted(set(concept_ids))}."
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
            raise ValueError("AUSteer factors must be finite.")
        return strengths

    @contextmanager
    def _intervention(self, mask_indices, strengths, token_mask):
        if self.au_masks is None:
            raise RuntimeError("AUSteer must be initialized before inference.")
        selected_masks = self.au_masks[mask_indices].float()
        handles = []

        for layer_index, module in enumerate(self._ffn_modules()):
            def hook(_module, _inputs, output, layer_index=layer_index):
                hidden = self._hidden_from_output(output)
                current_mask = token_mask
                if current_mask.shape != hidden.shape[:2]:
                    if hidden.shape[1] != 1 or hidden.shape[0] != current_mask.shape[0]:
                        raise ValueError(
                            "AUSteer token mask does not align with FFN outputs."
                        )
                    current_mask = torch.ones(
                        hidden.shape[:2], device=hidden.device, dtype=torch.bool
                    )
                active = (
                    current_mask.to(hidden.device).bool()
                    & strengths.to(hidden.device).ne(0).unsqueeze(1)
                )
                if not active.any():
                    return output
                scale = (
                    selected_masks[:, layer_index].to(hidden.device)
                    * strengths.to(hidden.device).unsqueeze(1)
                ).to(hidden.dtype)
                steered = hidden + hidden * scale.unsqueeze(1)
                result = torch.where(active.unsqueeze(-1), steered, hidden)
                return self._replace_hidden(output, result)

            handles.append(module.register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
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
                indices = self._mask_indices(batch["concept_id"].tolist())
                with self._intervention(
                    indices, strengths, inputs.attention_mask.bool()
                ):
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
        indices = self._mask_indices(batch_examples["concept_id"].tolist())
        model_inputs = self.choice_model_inputs(
            inputs,
            last_token_only=not kwargs.get("full_sequence", False),
            logits_to_keep=kwargs.get("choice_logits_to_keep"),
        )
        with self._intervention(
            indices, strengths, inputs.attention_mask.bool()
        ):
            outputs = self.model(**model_inputs, use_cache=False)
        return outputs, strengths

    def get_logits(self, concept_id, k=10):
        # AUSteer is a multiplicative FFN operator, not a residual direction.
        return [None], [None]

    def to(self, device):
        self.device = torch.device(device)
        if self.au_masks is not None:
            self.au_masks = self.au_masks.to(self.device, torch.float32)
        return self
