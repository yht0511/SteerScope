"""Native SteerScope adapter for Flow-Learned Activation Steering (FLAS)."""

from __future__ import annotations

import json
import logging
import math
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from .flas_core import (
    FLASConfig,
    FLASDataset,
    FLASGenerator,
    build_flow_model_from_base,
    collate_flas_batch,
    compute_diversity_loss,
    integrate_euler,
)
from .flas_core.compatibility import text_config, text_decoder
from .model import Model

logger = logging.getLogger(__name__)


class _FLASTrainingIntegrator(nn.Module):
    """Expose one DDP forward while retaining FLAS's internal Euler steps."""

    def __init__(self, flow_function, n_steps):
        super().__init__()
        self.flow_function = flow_function
        self.n_steps = int(n_steps)

    def forward(
        self,
        hidden_states,
        concept_hidden,
        concept_mask,
        terminal_times,
        padding_mask,
    ):
        steered, velocity, _ = integrate_euler(
            self.flow_function,
            hidden_states,
            concept_hidden,
            concept_mask,
            terminal_times,
            self.n_steps,
            padding_mask=padding_mask,
            use_cache=False,
        )
        return steered, velocity


class FLAS(Model):
    """One concept-conditioned flow shared by every training concept."""

    training_granularity = "all_concepts"
    inference_instance_scope = "shared"
    artifact_directory = "flas"
    requires_mean_activations = False
    requires_calibration_scale = False
    uses_intervention_positions = False
    load_trained_weights = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flas_config: FLASConfig | None = None
        self.flow_fn: nn.Module | None = None
        self.concept_enc: nn.Module | None = None
        self.trained_concept_ids: list[int] = []
        self.heldout_concept_ids: list[int] = []

    def __str__(self):
        return "FLAS"

    @classmethod
    def training_fingerprint_context(cls):
        return {
            "artifact_version": 3,
            "source_commit": "720ef8a67697d9b94130b374b5b3a1522a782566",
            "method": "flow_learned_activation_steering",
            "training_recipe": "paper_appendix_a_physical_batch",
            "factor_semantics": "flow_terminal_time_T",
            "integration": "fixed_step_euler",
            "concept_encoder": "frozen_first_two_base_model_layers",
            "training_precision": "bf16-mixed",
            # Preserve the released trainer's actual behavior.  The 2B paper
            # recipe has accumulation=1, so this is also identical to Table 4
            # there; the released 9B code advances both on microbatches.
            "scheduler_step_unit": "accumulated_microbatch",
            "validation_interval_unit": "training_microbatch",
            "transformers_compatibility": "gemma2_4.45.1",
        }

    def _config(self) -> FLASConfig:
        if self.flas_config is None:
            if self.training_args is None:
                raise ValueError(
                    "FLAS requires training arguments or a checkpoint config."
                )
            self.flas_config = FLASConfig.from_training_args(self.training_args)
        return self.flas_config

    def _build_model(self, *, initialize_from_base: bool) -> None:
        config = self._config()
        text_config(self.model.config)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.flow_fn, self.concept_enc = build_flow_model_from_base(
            self.model,
            layer=int(self.layer),
            num_blocks=config.num_blocks,
            time_conditioned=True,
            init_from_base=initialize_from_base,
        )
        self.flow_fn.to(self.device)
        self.concept_enc.to(self.device)
        self.concept_enc.eval()

    def make_model(self, **kwargs):
        self.flas_config = FLASConfig.from_training_args(self.training_args)
        self._build_model(initialize_from_base=True)
        self.flow_fn.train()

    @property
    def _flow_dtype(self):
        if self.flow_fn is None:
            raise RuntimeError("FLAS has not been constructed or loaded.")
        return next(self.flow_fn.parameters()).dtype

    @staticmethod
    def _distributed_world_size():
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    def _split_training_data(self, examples: pd.DataFrame):
        config = self._config()
        concept_ids = sorted(int(value) for value in examples.concept_id.unique())
        generator = torch.Generator().manual_seed(int(self.seed))
        # Keep the released split's RNG order: it samples a concept
        # permutation even when the requested holdout size is zero.
        permutation = torch.randperm(
            len(concept_ids), generator=generator
        ).tolist()
        heldout = set()
        if config.val_n_concepts:
            maximum = max(0, len(concept_ids) - 1)
            count = min(config.val_n_concepts, maximum)
            heldout = {concept_ids[index] for index in permutation[:count]}
        training = examples[~examples.concept_id.isin(heldout)].reset_index(drop=True)
        if training.empty:
            raise ValueError("FLAS concept holdout removed every training example.")

        validation_count = min(config.n_val_samples, len(training) // 10)
        if validation_count:
            validation_indices = torch.randperm(len(training), generator=generator)[
                :validation_count
            ].tolist()
            validation_mask = torch.zeros(len(training), dtype=torch.bool)
            validation_mask[validation_indices] = True
            validation = training.iloc[validation_indices].reset_index(drop=True)
            training = training.iloc[~validation_mask.numpy()].reset_index(drop=True)
        else:
            validation = training.iloc[:0].copy()
        self.heldout_concept_ids = sorted(heldout)
        self.trained_concept_ids = sorted(
            int(value) for value in training.concept_id.unique()
        )
        return training, validation

    def _dataloader(self, dataframe, *, training, epoch=0):
        config = self._config()
        dataset = FLASDataset(dataframe)
        world_size = self._distributed_world_size()
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=self.process_rank,
                shuffle=training,
                seed=int(self.seed),
                drop_last=training,
            )
            sampler.set_epoch(epoch)
        generator = torch.Generator().manual_seed(int(self.seed) + int(epoch))
        num_workers = config.num_workers if training else min(2, config.num_workers)
        return DataLoader(
            dataset,
            batch_size=int(self.training_args.batch_size),
            # The standalone FLAS trainer shuffles both train and validation.
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=bool(training),
            collate_fn=partial(
                collate_flas_batch,
                tokenizer=self.tokenizer,
                max_length=config.max_length,
                concept_max_length=config.concept_max_length,
            ),
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )

    def _training_precision_context(self):
        """Match Lightning's default ``precision='bf16-mixed'`` on CUDA."""
        device = (
            self.device
            if isinstance(self.device, torch.device)
            else torch.device(self.device)
        )
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _forward_with_flow(self, batch, flow_integrator, terminal_time):
        with self._training_precision_context():
            return self._forward_with_flow_impl(
                batch, flow_integrator, terminal_time
            )

    def _forward_with_flow_impl(self, batch, flow_integrator, terminal_time):
        with torch.no_grad():
            concept_hidden = self.concept_enc(
                batch["concept_input_ids"], batch["concept_attention_mask"]
            )
        captured = {}

        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            original = output[0] if is_tuple else output
            hidden = original.float()
            batch_size = hidden.shape[0]
            times = torch.as_tensor(
                terminal_time, device=hidden.device, dtype=hidden.dtype
            ).reshape(-1)
            if times.numel() == 1:
                times = times.expand(batch_size)
            steered, velocity = flow_integrator(
                hidden,
                concept_hidden.to(hidden.dtype),
                batch["concept_attention_mask"].float(),
                times,
                batch["attention_mask"].float(),
            )
            captured["velocity"] = velocity
            replacement = steered.to(original.dtype)
            return (replacement, *output[1:]) if is_tuple else replacement

        layer = text_decoder(self.model).layers[int(self.layer)]
        handle = layer.register_forward_hook(hook)
        try:
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
                return_dict=True,
            )
        finally:
            handle.remove()
        return outputs.loss, captured.get("velocity")

    @staticmethod
    def _to_device(batch, device):
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    @staticmethod
    def _scheduler(optimizer, warmup_steps, total_steps, accumulation_steps):
        def scale(step):
            # Match the released Lightning trainer, which advances this
            # scheduler on optimizer updates but expresses its curve in
            # accumulated training microbatches.
            source_step = step * accumulation_steps
            if source_step < warmup_steps:
                return source_step / max(warmup_steps, 1)
            progress = (source_step - warmup_steps) / max(
                total_steps - warmup_steps, 1
            )
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)

    @torch.no_grad()
    def _validation_loss(self, dataloader, flow_integrator):
        if dataloader is None:
            return None
        config = self._config()
        flow_integrator.eval()
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        count = torch.zeros((), device=self.device, dtype=torch.float64)
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= config.val_batches:
                break
            batch = self._to_device(batch, self.device)
            loss, _ = self._forward_with_flow(batch, flow_integrator, terminal_time=1.0)
            batch_size = int(batch["input_ids"].shape[0])
            loss_sum += loss.detach().double() * batch_size
            count += batch_size
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        flow_integrator.train()
        if count.item() == 0:
            return None
        return float((loss_sum / count).item())

    def train(self, examples: pd.DataFrame, **kwargs):
        if self.flow_fn is None or self.concept_enc is None:
            self.make_model()
        config = self._config()
        training, validation = self._split_training_data(examples)
        previous_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            train_loader = self._dataloader(training, training=True)
            if len(train_loader) == 0:
                raise ValueError(
                    "FLAS needs at least one complete training batch; reduce batch_size."
                )
            validation_loader = (
                self._dataloader(validation, training=False)
                if not validation.empty
                else None
            )
            world_size = self._distributed_world_size()
            flow_integrator = _FLASTrainingIntegrator(self.flow_fn, config.n_steps).to(
                self.device
            )
            if world_size > 1:
                device_index = (
                    self.device.index
                    if isinstance(self.device, torch.device)
                    else torch.device(self.device).index
                )
                flow_integrator = DDP(
                    flow_integrator,
                    device_ids=[device_index],
                    output_device=device_index,
                    find_unused_parameters=False,
                )
            optimizer = torch.optim.AdamW(
                self.flow_fn.parameters(),
                lr=float(self.training_args.lr),
                weight_decay=float(self.training_args.weight_decay),
            )
            accumulation = int(self.training_args.gradient_accumulation_steps or 1)
            if accumulation < 1:
                raise ValueError("FLAS gradient_accumulation_steps must be positive.")
            scheduler = self._scheduler(
                optimizer,
                config.warmup_steps,
                config.total_steps,
                accumulation,
            )
            optimizer.zero_grad(set_to_none=True)
            update_step = 0
            micro_step = 0
            epoch = 0
            best_loss = float("inf")
            best_state = None
            stale_validations = 0
            progress = tqdm(
                total=config.total_steps,
                position=self.process_rank,
                disable=self.process_rank != 0,
                desc="FLAS",
            )
            stop = False
            while update_step < config.total_steps and not stop:
                if isinstance(train_loader.sampler, DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)
                for batch in train_loader:
                    batch = self._to_device(batch, self.device)
                    terminal_time = (
                        torch.rand((), device=self.device)
                        * (config.t_max - config.t_min)
                        + config.t_min
                    )
                    synchronize = (micro_step + 1) % accumulation == 0
                    sync_context = (
                        nullcontext()
                        if synchronize or not isinstance(flow_integrator, DDP)
                        else flow_integrator.no_sync()
                    )
                    with sync_context:
                        language_loss, velocity = self._forward_with_flow(
                            batch, flow_integrator, terminal_time
                        )
                        diversity_loss = compute_diversity_loss(
                            velocity,
                            batch["concept_ids"],
                            batch["attention_mask"],
                        )
                        loss = (
                            language_loss + config.div_weight * diversity_loss
                        ) / accumulation
                        loss.backward()
                    micro_step += 1
                    if synchronize:
                        torch.nn.utils.clip_grad_norm_(
                            self.flow_fn.parameters(), 1.0
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        update_step += 1
                        progress.update(1)
                        progress.set_postfix(
                            lm=f"{language_loss.item():.4f}",
                            div=f"{diversity_loss.item():.4f}",
                        )
                    should_validate = validation_loader is not None and (
                        micro_step % config.val_every == 0
                    )
                    if should_validate:
                        validation_loss = self._validation_loss(
                            validation_loader, flow_integrator
                        )
                        if validation_loss is not None and validation_loss < best_loss:
                            best_loss = validation_loss
                            stale_validations = 0
                            best_state = {
                                key: value.detach().cpu().clone()
                                for key, value in self.flow_fn.state_dict().items()
                            }
                        else:
                            stale_validations += 1
                            if stale_validations >= config.patience:
                                stop = True
                    if update_step >= config.total_steps or stop:
                        break
                epoch += 1
            progress.close()
            if best_state is not None:
                self.flow_fn.load_state_dict(best_state)
            self.flow_fn.eval()
            logger.warning(
                "FLAS trained on %s examples across %s concepts; held out %s concepts.",
                len(training),
                len(self.trained_concept_ids),
                len(self.heldout_concept_ids),
            )
        finally:
            self.tokenizer.padding_side = previous_padding

    def _checkpoint_directory(self, dump_dir):
        root = Path(dump_dir)
        native = root / self.artifact_directory
        if native.exists() or not (root / "config.json").exists():
            return native
        return root

    def save(self, dump_dir, **kwargs):
        if self.flow_fn is None:
            raise RuntimeError("FLAS has no trained flow to save.")
        artifact = Path(dump_dir) / self.artifact_directory
        artifact.mkdir(parents=True, exist_ok=True)
        model_config = text_config(self.model.config)
        payload = {
            "format_version": 1,
            "method": "FLAS",
            "model_type": model_config.model_type,
            "base_model": getattr(self.model.config, "_name_or_path", None),
            "layer": int(self.layer),
            "hidden_size": int(model_config.hidden_size),
            "flas_config": self._config().to_dict(),
            "trained_concept_ids": self.trained_concept_ids,
            "heldout_concept_ids": self.heldout_concept_ids,
        }
        with open(artifact / "config.json", "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
        state = {
            key: (
                value.detach().to(torch.bfloat16).cpu()
                if value.is_floating_point()
                else value.detach().cpu()
            )
            for key, value in self.flow_fn.state_dict().items()
        }
        torch.save({"flow_fn": state}, artifact / "flow.pt")

    @staticmethod
    def _load_flow_state(artifact):
        candidates = (
            artifact / "flow.pt",
            artifact / "flow.safetensors",
            artifact / "model.safetensors",
            artifact / "final.pt",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            safetensors_files = sorted(artifact.glob("*.safetensors"))
            if len(safetensors_files) == 1:
                path = safetensors_files[0]
        if path is None:
            best = sorted(artifact.glob("best_step*.pt"))
            path = best[0] if best else None
        if path is None:
            raise FileNotFoundError(f"No FLAS flow checkpoint found in {artifact}.")
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(path), device="cpu")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return payload.get("flow_fn", payload)

    def load(self, dump_dir=None, **kwargs):
        artifact = self._checkpoint_directory(dump_dir)
        config_path = artifact / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"FLAS config not found: {config_path}")
        with open(config_path, encoding="utf-8") as file:
            payload = json.load(file)
        if "flas_config" in payload:
            if int(payload.get("format_version", -1)) != 1:
                raise ValueError("Unsupported FLAS checkpoint format.")
            if int(payload.get("layer", -1)) != int(self.layer):
                raise ValueError("FLAS checkpoint layer does not match evaluation.")
            model_config = text_config(self.model.config)
            if int(payload.get("hidden_size", -1)) != int(model_config.hidden_size):
                raise ValueError(
                    "FLAS checkpoint hidden size does not match the model."
                )
            config_values = payload["flas_config"]
            self.trained_concept_ids = [
                int(value) for value in payload.get("trained_concept_ids", [])
            ]
            self.heldout_concept_ids = [
                int(value) for value in payload.get("heldout_concept_ids", [])
            ]
        else:
            # Released standalone FLAS checkpoints store the trainer arguments
            # directly in config.json.
            configured_layer = int(payload.get("layer", self.layer))
            if configured_layer != int(self.layer):
                raise ValueError("FLAS checkpoint layer does not match evaluation.")
            config_values = payload
        self.flas_config = FLASConfig.from_mapping(config_values)
        state = self._load_flow_state(artifact)
        if not state:
            raise ValueError("FLAS checkpoint contains an empty flow state.")
        checkpoint_dtype = next(iter(state.values())).dtype
        self._build_model(initialize_from_base=False)
        self.flow_fn.to(dtype=checkpoint_dtype)
        missing, unexpected = self.flow_fn.load_state_dict(state, strict=False)
        meaningful_missing = [
            key for key in missing if not key.endswith("rotary_emb.inv_freq")
        ]
        meaningful_unexpected = [
            key for key in unexpected if not key.endswith("rotary_emb.inv_freq")
        ]
        if meaningful_missing or meaningful_unexpected:
            raise ValueError(
                "FLAS checkpoint parameters do not match the configured flow: "
                f"missing={meaningful_missing[:5]}, unexpected={meaningful_unexpected[:5]}."
            )
        self.flow_fn.to(self.device).eval()
        self.concept_enc.to(self.device).eval()

    @staticmethod
    def _strengths(examples, device):
        strengths = torch.as_tensor(
            examples["factor"].tolist(), device=device, dtype=torch.float32
        )
        if not torch.isfinite(strengths).all():
            raise ValueError("FLAS factors must be finite.")
        if (strengths < 0).any():
            raise ValueError("FLAS flow times must be non-negative.")
        return strengths

    @torch.no_grad()
    def _encode_concepts(self, texts):
        previous_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            encoded = self.tokenizer(
                [str(text) for text in texts],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._config().concept_max_length,
            ).to(self.device)
        finally:
            self.tokenizer.padding_side = previous_padding
        hidden = self.concept_enc(encoded["input_ids"], encoded["attention_mask"])
        return hidden.to(self._flow_dtype), encoded["attention_mask"].float()

    @contextmanager
    def _choice_intervention(
        self, concept_hidden, concept_mask, strengths, attention_mask
    ):
        if torch.count_nonzero(strengths).item() == 0:
            yield
            return
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            original = output[0] if is_tuple else output
            hidden, _, _ = integrate_euler(
                self.flow_fn,
                original.to(self._flow_dtype),
                concept_hidden,
                concept_mask,
                strengths,
                self._config().n_steps,
                padding_mask=attention_mask.float(),
                use_cache=False,
                position_ids=position_ids,
            )
            replacement = hidden.to(original.dtype)
            return (replacement, *output[1:]) if is_tuple else replacement

        handle = (
            text_decoder(self.model).layers[int(self.layer)].register_forward_hook(hook)
        )
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def predict_steer(self, examples: pd.DataFrame, **kwargs):
        if self.flow_fn is None or self.concept_enc is None:
            raise RuntimeError("FLAS must be loaded before inference.")
        self.model.eval()
        self.flow_fn.eval()
        self.concept_enc.eval()
        batch_size = int(kwargs.get("batch_size", 64))
        input_column = "steered_input" if kwargs.get("use_synergy") else "input"
        generations = []
        used_strengths = []
        progress = tqdm(
            range(0, len(examples), batch_size),
            position=self.process_rank,
            leave=True,
            disable=not kwargs.get("show_progress", True),
        )
        generator = FLASGenerator(
            self.model,
            self.tokenizer,
            self.flow_fn,
            self.concept_enc,
            self.layer,
            n_steps=self._config().n_steps,
            concept_max_length=self._config().concept_max_length,
            # The released FLAS generator truncates formatted prompts at 512.
            input_max_length=512,
            device=self.device,
        )
        try:
            for start in range(0, len(examples), batch_size):
                batch = examples.iloc[start : start + batch_size]
                if "input_concept" not in batch:
                    raise KeyError("FLAS inference requires an input_concept column.")
                strengths = self._strengths(batch, self.device)
                generations.extend(
                    generator.generate_batch(
                        batch[input_column].astype(str).tolist(),
                        batch["input_concept"].astype(str).tolist(),
                        strengths.tolist(),
                        max_new_tokens=int(kwargs.get("eval_output_length", 128)),
                        temperature=float(kwargs.get("temperature", 1.0)),
                        do_sample=bool(kwargs.get("do_sample", True)),
                    )
                )
                used_strengths.extend(strengths.float().cpu().tolist())
                progress.update(1)
        finally:
            progress.close()
        return {
            "steered_generation": generations,
            "strength": used_strengths,
        }

    def prepare_choice_logits(self, examples, **kwargs):
        if self.flow_fn is None or self.concept_enc is None:
            raise RuntimeError("FLAS must be loaded before candidate inference.")
        self.model.eval()
        self.flow_fn.eval()
        self.concept_enc.eval()

    def choice_forward(self, inputs, batch_examples, **kwargs):
        if "input_concept" not in batch_examples:
            raise KeyError("FLAS candidate inference requires input_concept.")
        strengths = self._strengths(batch_examples, self.device)
        concept_hidden, concept_mask = self._encode_concepts(
            batch_examples["input_concept"].tolist()
        )
        model_inputs = self.choice_model_inputs(
            inputs,
            last_token_only=not kwargs.get("full_sequence", False),
            logits_to_keep=kwargs.get("choice_logits_to_keep"),
        )
        with self._choice_intervention(
            concept_hidden,
            concept_mask,
            strengths,
            inputs["attention_mask"],
        ):
            outputs = self.model(**model_inputs, use_cache=False)
        return outputs, strengths

    def to(self, device):
        super().to(device)
        if self.flow_fn is not None:
            self.flow_fn.to(device)
        if self.concept_enc is not None:
            self.concept_enc.to(device)
        return self


__all__ = ["FLAS"]
