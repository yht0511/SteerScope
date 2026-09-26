"""PSR distills prompt-steered answer-token residuals into one layer (SPSR) or every decoder layer (APSR)."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
import inspect
import json
import logging
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from .model import Model
from .prompt import PromptSteering
from ..templates.prompt_templates import T_GENERATE_PREPEND_STEERING_PROMPT
from ..utils.constants import HAS_SYSTEM_PROMPT_MODELS
from ..utils.model_utils import get_suffix_length
from ..utils.training import (
    GRADIENT_ACCUMULATION_SEMANTICS,
    is_optimizer_step,
    normalize_loss_for_accumulation,
    optimizer_steps_per_epoch,
)


logger = logging.getLogger(__name__)


@dataclass
class _PSRTeacherCacheEntry:
    """CPU-packed teacher residuals and any pre-intervention prefix loss for one training record."""

    residuals: torch.Tensor
    prefix_loss: torch.Tensor


class FocusedPSRIntervention(nn.Module):
    """Apply a learned ReLU-gated direction to answer-token hidden states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.steering_proj = nn.Linear(
            hidden_size, 1, bias=False, dtype=torch.float32
        )
        self.location_fit_proj = nn.Linear(
            hidden_size, 1, bias=True, dtype=torch.float32
        )
        self.location_activation = nn.ReLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        strengths: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        hidden_float = hidden_states.to(self.steering_proj.weight.dtype)
        location_fit = self.location_activation(
            self.location_fit_proj(hidden_float)
        )
        coefficients = (
            strengths.to(hidden_float.device, hidden_float.dtype).reshape(-1, 1, 1)
            * location_fit
            * response_mask.to(hidden_float.dtype).unsqueeze(-1)
        )
        steered = hidden_float + coefficients * self.steering_proj.weight[0]
        return steered.to(original_dtype)


class _PSRBase(Model):
    """Shared implementation for the single-layer and all-layer PSR variants."""

    inference_instance_scope = "per_concept"
    uses_prompt_generation = True
    requires_mean_activations = False
    requires_calibration_scale = False
    uses_intervention_positions = False
    load_trained_weights = True

    variant = "base"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_model_name = kwargs.get("lm_model_name")
        self.psr_modules = nn.ModuleDict()
        self.psr_layers: list[int] = []
        self.prompt_instruction: str | None = None
        self._suffix_length: int | None = None
        self._supports_num_logits_to_keep: bool | None = None

    @classmethod
    def training_fingerprint_context(cls):
        return {
            "artifact_version": 2,
            "variant": cls.variant,
            "objective": "prompt_steering_imitation",
            "module": "focused_relu_answer_only_no_coefficient_bias",
            "layers_to_imitate": "all",
            "normalize_by_initial_psi": True,
            "prompt_generation_template": T_GENERATE_PREPEND_STEERING_PROMPT,
            "composition": "{instruction}\\n\\nQuestion: {raw_input}",
            "gradient_accumulation_semantics": (
                GRADIENT_ACCUMULATION_SEMANTICS
            ),
        }

    def _selected_layers(self, decoder_layers: nn.ModuleList) -> list[int]:
        raise NotImplementedError

    @staticmethod
    def _decoder_layers(model) -> nn.ModuleList:
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

    def make_model(self, **kwargs):
        decoder_layers = self._decoder_layers(self.model)
        selected = self._selected_layers(decoder_layers)
        if not selected:
            raise ValueError(f"{self.__class__.__name__} selected no steering layers.")
        invalid = [layer for layer in selected if layer < 0 or layer >= len(decoder_layers)]
        if invalid:
            raise ValueError(
                f"Invalid PSR layer(s) {invalid}; model has {len(decoder_layers)} layers."
            )
        self.psr_layers = [int(layer) for layer in selected]
        hidden_size = int(self.model.config.hidden_size)
        self.psr_modules = nn.ModuleDict({
            str(layer): FocusedPSRIntervention(hidden_size)
            for layer in self.psr_layers
        })
        self.psr_modules.to(self.device)

        # Only the focused modules are optimized.  Keeping the subject model
        # frozen also prevents the target PromptSteering branch from drifting.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _generate_prompt_instruction(self, examples, **kwargs) -> str:
        prompt_model = PromptSteering(
            self.model,
            self.tokenizer,
            layer=self.layer,
            training_args=self.training_args,
            lm_model_name=self.lm_model_name,
            device=self.device,
            seed=self.seed,
            dump_dir=self.dump_dir,
        )
        prompt_model.train(examples, **kwargs)
        artifact = prompt_model._trained_prompt
        if not artifact or not artifact.get("instruction"):
            raise RuntimeError("PromptSteering did not produce a PSR instruction.")
        return str(artifact["instruction"]).strip()

    def _system_messages(self) -> list[dict[str, str]]:
        if self.lm_model_name in HAS_SYSTEM_PROMPT_MODELS:
            return [{"role": "system", "content": "You are a helpful assistant."}]
        return []

    def _pair_tokens(self, question: str, answer: str) -> tuple[list[int], int]:
        prompt_messages = self._system_messages() + [
            {"role": "user", "content": question}
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": answer}
        ]
        prompt_tokens = list(self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        ))
        full_tokens = list(self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        ))
        if self._suffix_length is None:
            self._suffix_length = int(get_suffix_length(self.tokenizer)[0])
        if self._suffix_length:
            full_tokens = full_tokens[:-self._suffix_length]
        if not prompt_tokens or not full_tokens:
            raise ValueError("PSR chat templating produced an empty token sequence.")

        # Official PSR starts answer-only steering/loss at the final prompt
        # token, which is also shared by both paired views.
        response_start = len(prompt_tokens) - 1
        if response_start >= len(full_tokens):
            raise ValueError("PSR example contains no aligned response tokens.")

        context_length = int(
            getattr(self.model.config, "max_position_embeddings", 0) or 0
        )
        if context_length and len(full_tokens) > context_length:
            raise ValueError(
                f"PSR sequence has {len(full_tokens)} tokens but the subject "
                f"model context is {context_length}. Shorten the generated "
                "PromptSteering instruction instead of truncating y'."
            )
        return [int(token) for token in full_tokens], response_start

    def _training_records(self, examples) -> list[dict[str, object]]:
        required = {"raw_input", "raw_output"}
        missing = required - set(examples.columns)
        if missing:
            raise KeyError(
                f"{self.__class__.__name__} requires preserved training columns "
                f"{sorted(missing)}."
            )
        if self.prompt_instruction is None:
            raise RuntimeError("PSR prompt instruction has not been generated.")

        records = []
        for record_id, (_, row) in enumerate(examples.iterrows()):
            question = str(row["raw_input"])
            answer = str(row["raw_output"])
            steered_question = (
                f"{self.prompt_instruction}\n\nQuestion: {question}"
            )
            base_tokens, base_start = self._pair_tokens(question, answer)
            prompt_tokens, prompt_start = self._pair_tokens(
                steered_question, answer
            )
            base_tail = base_tokens[base_start:]
            prompt_tail = prompt_tokens[prompt_start:]
            if base_tail != prompt_tail:
                raise ValueError(
                    "PSR response tokenization differs between x+y' and x'+y'. "
                    "The two branches cannot be aligned faithfully."
                )
            records.append({
                "record_id": record_id,
                "base_tokens": base_tokens,
                "base_start": base_start,
                "prompt_tokens": prompt_tokens,
                "prompt_start": prompt_start,
            })
        if not records:
            raise ValueError("PSR requires at least one positive training example.")
        return records

    def _pad_branch(
        self,
        records: list[dict[str, object]],
        token_key: str,
        start_key: str,
    ) -> dict[str, torch.Tensor]:
        sequences = [list(record[token_key]) for record in records]
        starts = [int(record[start_key]) for record in records]
        max_length = max(len(sequence) for sequence in sequences)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("PSR requires a tokenizer pad or EOS token.")

        input_ids = []
        attention_masks = []
        response_masks = []
        for sequence, start in zip(sequences, starts):
            padding = max_length - len(sequence)
            input_ids.append([pad_token_id] * padding + sequence)
            attention_masks.append([0] * padding + [1] * len(sequence))
            adjusted_start = padding + start
            response_masks.append(
                [False] * adjusted_start
                + [True] * (max_length - adjusted_start)
            )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "response_mask": torch.tensor(response_masks, dtype=torch.bool),
        }

    def _collate_records(self, records, *, include_prompt: bool):
        records = list(records)
        batch = {
            "record_ids": torch.tensor(
                [int(record["record_id"]) for record in records],
                dtype=torch.long,
            ),
            "base": self._pad_branch(records, "base_tokens", "base_start"),
        }
        if include_prompt:
            batch["prompt"] = self._pad_branch(
                records, "prompt_tokens", "prompt_start"
            )
        return batch

    def _make_psr_dataloader(
        self,
        records: list[dict[str, object]],
        *,
        shuffle: bool,
        include_prompt: bool,
    ):
        generator = torch.Generator().manual_seed(123)
        return DataLoader(
            records,
            batch_size=int(self.training_args.batch_size),
            shuffle=shuffle,
            generator=generator if shuffle else None,
            collate_fn=partial(
                self._collate_records,
                include_prompt=include_prompt,
            ),
        )

    @staticmethod
    def _hidden_from_output(output):
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _replace_hidden(output, hidden):
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    def _can_limit_capture_logits(self) -> bool:
        """Return whether final-token-only logit projection leaves the hooked residuals unchanged."""

        if self._supports_num_logits_to_keep is None:
            try:
                parameters = inspect.signature(self.model.forward).parameters
            except (TypeError, ValueError):
                self._supports_num_logits_to_keep = False
            else:
                self._supports_num_logits_to_keep = (
                    "num_logits_to_keep" in parameters
                )
        return self._supports_num_logits_to_keep

    def _capture_forward(self, branch: dict[str, torch.Tensor]) -> None:
        kwargs = {
            "input_ids": branch["input_ids"],
            "attention_mask": branch["attention_mask"],
            "use_cache": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        if self._can_limit_capture_logits():
            kwargs["num_logits_to_keep"] = 1
        self.model(**kwargs)

    def _teacher_cache_is_safe(self) -> bool:
        """Return whether teacher residuals are deterministic across epochs for the current model and batching setup."""

        if int(self.training_args.batch_size) != 1:
            logger.warning(
                "%s teacher cache disabled: strict reuse requires batch_size=1.",
                self.__class__.__name__,
            )
            return False

        config = getattr(self.model, "config", None)
        if getattr(config, "model_type", None) != "gemma2":
            logger.warning(
                "%s teacher cache disabled: deterministic reuse is only "
                "validated for Gemma-2.",
                self.__class__.__name__,
            )
            return False

        dropout_types = (
            nn.Dropout,
            nn.Dropout1d,
            nn.Dropout2d,
            nn.Dropout3d,
            nn.AlphaDropout,
            nn.FeatureAlphaDropout,
        )
        for module in self.model.modules():
            if isinstance(module, dropout_types) and float(module.p) > 0:
                logger.warning(
                    "%s teacher cache disabled: %s has p=%s.",
                    self.__class__.__name__,
                    module.__class__.__name__,
                    module.p,
                )
                return False
            if isinstance(module, nn.RReLU):
                logger.warning(
                    "%s teacher cache disabled: RReLU is stochastic in train mode.",
                    self.__class__.__name__,
                )
                return False
            if isinstance(
                module,
                (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
            ):
                logger.warning(
                    "%s teacher cache disabled: BatchNorm changes in train mode.",
                    self.__class__.__name__,
                )
                return False

        if config is not None:
            for name, value in vars(config).items():
                if "drop" not in name.lower() or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and float(value) > 0:
                    logger.warning(
                        "%s teacher cache disabled: model config %s=%s.",
                        self.__class__.__name__,
                        name,
                        value,
                    )
                    return False
        return True

    @contextmanager
    def _forward_hooks(
        self,
        *,
        strengths: torch.Tensor | None = None,
        response_mask: torch.Tensor | None = None,
        capture: bool = False,
        capture_layers: range | list[int] | tuple[int, ...] | None = None,
    ):
        decoder_layers = self._decoder_layers(self.model)
        handles = []
        captured: dict[int, torch.Tensor] = {}

        if strengths is not None:
            if response_mask is None:
                raise ValueError("PSR intervention requires a response mask.")
            for layer in self.psr_layers:
                intervention = self.psr_modules[str(layer)]

                def intervention_hook(
                    _module,
                    _inputs,
                    output,
                    intervention=intervention,
                ):
                    hidden = self._hidden_from_output(output)
                    current_mask = response_mask
                    if current_mask.shape[1] != hidden.shape[1]:
                        # Cached decoding presents one token after the initial
                        # full-prompt forward pass.
                        if hidden.shape[1] != 1:
                            raise ValueError(
                                "PSR inference mask does not align with hidden states."
                            )
                        current_mask = torch.ones(
                            hidden.shape[:2],
                            device=hidden.device,
                            dtype=torch.bool,
                        )
                    steered = intervention(
                        hidden,
                        strengths,
                        current_mask.to(hidden.device),
                    )
                    return self._replace_hidden(output, steered)

                handles.append(
                    decoder_layers[layer].register_forward_hook(intervention_hook)
                )

        if capture:
            capture_layer_set = (
                None if capture_layers is None else set(capture_layers)
            )
            for layer, decoder_layer in enumerate(decoder_layers):
                if (
                    capture_layer_set is not None
                    and layer not in capture_layer_set
                ):
                    continue

                def capture_hook(_module, _inputs, output, layer=layer):
                    captured[layer] = self._hidden_from_output(output)

                handles.append(decoder_layer.register_forward_hook(capture_hook))

        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def _capture_branch(
        self,
        branch: dict[str, torch.Tensor],
        *,
        intervene: bool,
        requires_grad: bool,
        capture_layers: range | list[int] | tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        branch = {key: value.to(self.device) for key, value in branch.items()}
        strengths = (
            torch.ones(branch["input_ids"].shape[0], device=self.device)
            if intervene else None
        )
        gradient_context = nullcontext() if requires_grad else torch.no_grad()
        with gradient_context:
            with self._forward_hooks(
                strengths=strengths,
                response_mask=branch["response_mask"] if intervene else None,
                capture=True,
                capture_layers=capture_layers,
            ) as captured:
                self._capture_forward(branch)
        return dict(captured)

    def _imitation_loss(
        self,
        batch,
        *,
        requires_grad: bool,
        teacher_cache: dict[int, _PSRTeacherCacheEntry] | None = None,
        populate_teacher_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if populate_teacher_cache and teacher_cache is None:
            raise ValueError("Populating the PSR teacher cache requires a cache.")
        use_cached_target = (
            teacher_cache is not None and not populate_teacher_cache
        )
        if not use_cached_target and "prompt" not in batch:
            raise KeyError("Uncached PSR loss requires the prompt branch.")

        base_branch = {
            key: value.to(self.device) for key, value in batch["base"].items()
        }
        prompt_branch = None
        target = None
        if not use_cached_target:
            prompt_branch = {
                key: value.to(self.device)
                for key, value in batch["prompt"].items()
            }
            target = self._capture_branch(
                prompt_branch,
                intervene=False,
                requires_grad=False,
            )

        num_layers = len(self._decoder_layers(self.model))
        first_cached_layer = min(self.psr_layers)
        loss_layers = (
            range(first_cached_layer, num_layers)
            if use_cached_target
            else range(num_layers)
        )
        steered = self._capture_branch(
            base_branch,
            intervene=True,
            requires_grad=requires_grad,
            capture_layers=loss_layers if use_cached_target else None,
        )

        batch_size = base_branch["input_ids"].shape[0]
        record_ids = [int(value) for value in batch["record_ids"].tolist()]
        cached_entries = None
        cached_residuals = None
        if use_cached_target:
            if batch_size != 1:
                raise RuntimeError(
                    "Strict PSR teacher-cache reuse only supports batch_size=1."
                )
            try:
                cached_entries = [teacher_cache[record_ids[0]]]
            except KeyError as error:
                raise KeyError(
                    f"PSR teacher cache is missing record {record_ids[0]}."
                ) from error
            cached_residuals = cached_entries[0].residuals.to(self.device)
            per_example = cached_entries[0].prefix_loss.to(
                self.device, dtype=torch.float32
            ).reshape(1).clone()
        else:
            per_example = torch.zeros(
                batch_size,
                device=self.device,
                dtype=torch.float32,
            )

        base_mask = base_branch["response_mask"]
        prompt_mask = (
            None if prompt_branch is None else prompt_branch["response_mask"]
        )
        prefix_losses = torch.zeros_like(per_example)
        for layer in loss_layers:
            if layer not in steered:
                raise RuntimeError(f"PSR failed to capture residual layer {layer}.")
            for index in range(per_example.shape[0]):
                base_segment = steered[layer][index][base_mask[index]]
                if use_cached_target:
                    prompt_segment = cached_residuals[
                        layer - first_cached_layer
                    ]
                else:
                    if target is None or layer not in target:
                        raise RuntimeError(
                            f"PSR failed to capture residual layer {layer}."
                        )
                    prompt_segment = target[layer][index][prompt_mask[index]]
                if base_segment.shape != prompt_segment.shape:
                    raise ValueError(
                        "PSR aligned activation segments have different shapes: "
                        f"{tuple(base_segment.shape)} vs "
                        f"{tuple(prompt_segment.shape)}."
                    )
                per_example[index] = per_example[index] + torch.mean(
                    (base_segment - prompt_segment) ** 2
                )
            if populate_teacher_cache and layer + 1 == first_cached_layer:
                prefix_losses.copy_(per_example.detach())

        if populate_teacher_cache:
            if target is None or prompt_mask is None:
                raise RuntimeError("PSR did not retain teacher targets to cache.")
            if first_cached_layer == 0:
                prefix_losses.zero_()
            for index, record_id in enumerate(record_ids):
                if record_id in teacher_cache:
                    raise RuntimeError(
                        f"PSR teacher cache saw duplicate record {record_id}."
                    )
                residuals = torch.stack([
                    target[layer][index][prompt_mask[index]].detach()
                    for layer in range(first_cached_layer, num_layers)
                ]).to("cpu", copy=True)
                teacher_cache[record_id] = _PSRTeacherCacheEntry(
                    residuals=residuals,
                    prefix_loss=prefix_losses[index].detach().to(
                        "cpu", dtype=torch.float32, copy=True
                    ),
                )
        return per_example.mean(), per_example

    def _initial_psi_scale(
        self,
        dataloader,
        *,
        teacher_cache: dict[int, _PSRTeacherCacheEntry] | None = None,
    ) -> float:
        total = 0.0
        count = 0
        for batch in dataloader:
            _, per_example = self._imitation_loss(
                batch,
                requires_grad=False,
                teacher_cache=teacher_cache,
                populate_teacher_cache=teacher_cache is not None,
            )
            total += float(per_example.sum().cpu())
            count += int(per_example.numel())
        if count == 0:
            raise ValueError("Cannot normalize PSR loss with an empty dataloader.")
        scale = total / count
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid initial PSR imitation loss: {scale}.")
        return scale

    def train(self, examples, **kwargs):
        if "category" in examples.columns:
            examples = examples[examples["category"] == "positive"].copy()
        if examples.empty:
            raise ValueError("PSR requires positive training examples.")
        self.prompt_instruction = self._generate_prompt_instruction(
            examples, **kwargs
        )
        records = self._training_records(examples)
        use_teacher_cache = self._teacher_cache_is_safe()
        teacher_cache = {} if use_teacher_cache else None
        try:
            evaluation_loader = self._make_psr_dataloader(
                records,
                shuffle=False,
                include_prompt=True,
            )
            initial_psi = self._initial_psi_scale(
                evaluation_loader,
                teacher_cache=teacher_cache,
            )
            logger.warning(
                "%s initial prompt-imitation loss: %.6f",
                self.__class__.__name__,
                initial_psi,
            )
            if teacher_cache is not None:
                if len(teacher_cache) != len(records):
                    raise RuntimeError(
                        "PSR teacher cache did not cover every training record: "
                        f"{len(teacher_cache)} vs {len(records)}."
                    )
                cache_bytes = sum(
                    entry.residuals.numel()
                    * entry.residuals.element_size()
                    + entry.prefix_loss.numel()
                    * entry.prefix_loss.element_size()
                    for entry in teacher_cache.values()
                )
                logger.warning(
                    "%s cached %s teacher records on CPU (%.1f MiB).",
                    self.__class__.__name__,
                    len(teacher_cache),
                    cache_bytes / (1024 ** 2),
                )

            train_loader = self._make_psr_dataloader(
                records,
                shuffle=True,
                include_prompt=teacher_cache is None,
            )
            for module in self.psr_modules.values():
                module.train()
            # The official fit path places both the subject model and focused
            # modules in train mode. Gemma-2 has no active residual dropout, so
            # the paired target remains deterministic while preserving parity.
            self.model.train()

            gradient_accumulation_steps = int(
                self.training_args.gradient_accumulation_steps or 1
            )
            if gradient_accumulation_steps < 1:
                raise ValueError("gradient_accumulation_steps must be positive.")
            n_epochs = int(self.training_args.n_epochs)
            num_microbatches = len(train_loader)
            updates_per_epoch = optimizer_steps_per_epoch(
                num_microbatches, gradient_accumulation_steps
            )
            total_updates = max(1, n_epochs * updates_per_epoch)
            optimizer = torch.optim.AdamW(
                self.psr_modules.parameters(),
                lr=float(self.training_args.lr),
                weight_decay=float(self.training_args.weight_decay or 0.0),
            )
            scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=total_updates,
            )
            optimizer.zero_grad()
            progress = tqdm(
                total=n_epochs * len(train_loader),
                position=self.process_rank,
                leave=True,
                desc=str(self),
            )
            try:
                for epoch in range(n_epochs):
                    epoch_losses = []
                    for step, batch in enumerate(train_loader):
                        raw_loss, _ = self._imitation_loss(
                            batch,
                            requires_grad=True,
                            teacher_cache=teacher_cache,
                        )
                        loss = raw_loss / initial_psi
                        normalized_loss = normalize_loss_for_accumulation(
                            loss,
                            step,
                            num_microbatches,
                            gradient_accumulation_steps,
                        )
                        normalized_loss.backward()
                        epoch_losses.append(float(loss.detach().cpu()))

                        should_update = is_optimizer_step(
                            step,
                            num_microbatches,
                            gradient_accumulation_steps,
                        )
                        if should_update:
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                        progress.update(1)
                        progress.set_postfix(
                            loss=f"{epoch_losses[-1]:.6f}",
                            lr=f"{scheduler.get_last_lr()[0]:.3g}",
                        )
                    if (
                        epoch_losses
                        and sum(epoch_losses) / len(epoch_losses) < 0.001
                    ):
                        logger.warning(
                            "%s converged after epoch %s.",
                            self.__class__.__name__,
                            epoch + 1,
                        )
                        break
            finally:
                progress.close()
        finally:
            if teacher_cache is not None:
                teacher_cache.clear()
        self.model.eval()
        for module in self.psr_modules.values():
            module.eval()

    def save(self, dump_dir, **kwargs):
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        model_name = kwargs.get("model_name", str(self))
        weights = {}
        biases = {}
        for layer in self.psr_layers:
            module = self.psr_modules[str(layer)]
            weights[f"layer_{layer}.steering"] = (
                module.steering_proj.weight.detach().cpu()
            )
            weights[f"layer_{layer}.location"] = (
                module.location_fit_proj.weight.detach().cpu()
            )
            biases[f"layer_{layer}.location"] = (
                module.location_fit_proj.bias.detach().cpu().reshape(1, 1)
            )
        torch.save(weights, dump_dir / f"{model_name}_weight.pt")
        torch.save(biases, dump_dir / f"{model_name}_bias.pt")
        with open(
            dump_dir / f"{model_name}_prompt.json", "w", encoding="utf-8"
        ) as file:
            json.dump({
                "concept_id": kwargs.get("concept_id"),
                "instruction": self.prompt_instruction,
                "generator_model": getattr(self.training_args, "lm_model", None),
                "temperature": getattr(
                    self.training_args, "prompt_temperature", 0.0
                ),
            }, file, indent=2, sort_keys=True)

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", str(self))
        concept_id = int(kwargs.get("concept_id", 0))
        weight_path = Path(dump_dir) / f"{model_name}_weight.pt"
        bias_path = Path(dump_dir) / f"{model_name}_bias.pt"
        weights = torch.load(weight_path, map_location="cpu", weights_only=True)
        biases = torch.load(bias_path, map_location="cpu", weights_only=True)
        self.make_model(**kwargs)
        first = next(iter(weights.values()))
        if concept_id >= first.shape[0]:
            if first.shape[0] == 1:
                selected = 0
            else:
                raise IndexError(
                    f"{model_name} checkpoint has {first.shape[0]} concepts and "
                    f"cannot select concept ID {concept_id}."
                )
        else:
            selected = concept_id
        for layer in self.psr_layers:
            module = self.psr_modules[str(layer)]
            steering = weights[f"layer_{layer}.steering"][selected]
            location = weights[f"layer_{layer}.location"][selected]
            location_bias = biases[f"layer_{layer}.location"][selected]
            module.steering_proj.weight.data.copy_(
                steering.reshape_as(module.steering_proj.weight)
            )
            module.location_fit_proj.weight.data.copy_(
                location.reshape_as(module.location_fit_proj.weight)
            )
            module.location_fit_proj.bias.data.copy_(
                location_bias.reshape_as(module.location_fit_proj.bias)
            )
            module.eval()
        self.concept_id_map = {concept_id: 0}

    def to(self, device):
        self.device = device
        self.psr_modules.to(device)
        return self

    @staticmethod
    def _last_real_positions(attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            attention_mask.shape[1], device=attention_mask.device
        ).unsqueeze(0)
        return (positions * attention_mask.long()).max(dim=1).values

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.model.eval()
        self.psr_modules.eval()
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        batch_size = int(kwargs.get("batch_size", 64))
        generations = []
        strengths_out = []
        generation_kwargs = self.generation_kwargs(
            kwargs.get("eval_output_length", 128),
            kwargs.get("temperature", 1.0),
            kwargs.get("do_sample", True),
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
                strengths = torch.as_tensor(
                    batch["factor"].tolist(),
                    device=self.device,
                    dtype=torch.float32,
                )
                response_mask = torch.zeros_like(
                    inputs["attention_mask"], dtype=torch.bool
                )
                if kwargs.get("intervene_on_prompt", True):
                    last_positions = self._last_real_positions(
                        inputs["attention_mask"]
                    )
                    response_mask[
                        torch.arange(len(batch), device=self.device),
                        last_positions,
                    ] = True
                with self._forward_hooks(
                    strengths=strengths,
                    response_mask=response_mask,
                    capture=False,
                ):
                    output = self.model.generate(
                        **inputs,
                        **generation_kwargs,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                prompt_length = inputs["input_ids"].shape[1]
                generations.extend(
                    self.tokenizer.decode(
                        sequence[prompt_length:], skip_special_tokens=True
                    )
                    for sequence in output
                )
                strengths_out.extend(strengths.cpu().tolist())
        finally:
            self.tokenizer.padding_side = original_padding_side
        return {
            "steered_generation": generations,
            "strength": strengths_out,
        }

    def prepare_choice_logits(self, examples, **kwargs):
        self.model.eval()
        self.psr_modules.eval()

    def choice_forward(self, inputs, batch_examples, **kwargs):
        strengths = torch.as_tensor(
            batch_examples["factor"].tolist(),
            device=self.device,
            dtype=torch.float32,
        )
        attention_mask = inputs["attention_mask"]
        end_positions = self._last_real_positions(attention_mask)
        if kwargs.get("full_sequence", False) and "_choice_token_ids" in batch_examples:
            candidate_lengths = torch.as_tensor(
                [len(value) for value in batch_examples["_choice_token_ids"]],
                device=self.device,
            )
            start_positions = end_positions - candidate_lengths
        else:
            start_positions = end_positions
        positions = torch.arange(
            attention_mask.shape[1], device=self.device
        ).unsqueeze(0)
        response_mask = (
            positions >= start_positions.unsqueeze(1)
        ) & attention_mask.bool()
        with self._forward_hooks(
            strengths=strengths,
            response_mask=response_mask,
            capture=False,
        ):
            outputs = self.model(
                **self.choice_model_inputs(
                    inputs,
                    last_token_only=not kwargs.get("full_sequence", False),
                    logits_to_keep=kwargs.get("choice_logits_to_keep"),
                ),
                use_cache=False,
            )
        return outputs, strengths

    def get_logits(self, concept_id, k=10):
        # Focused PSR is activation-dependent, so a single unembedded vector is
        # not a complete description of its token-level intervention.
        return [None], [None]


class SPSR(_PSRBase):
    """Single-layer Prompt Steering Replacement (official S-PSR)."""

    variant = "single_layer"

    def __str__(self):
        return "SPSR"

    def _selected_layers(self, decoder_layers: nn.ModuleList) -> list[int]:
        return [int(self.layer)]


class APSR(_PSRBase):
    """All-layer Prompt Steering Replacement (official A-PSR)."""

    variant = "all_layers"

    def __str__(self):
        return "APSR"

    def _selected_layers(self, decoder_layers: nn.ModuleList) -> list[int]:
        return list(range(len(decoder_layers)))
