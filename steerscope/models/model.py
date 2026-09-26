from dataclasses import dataclass
import inspect
import torch, einops, os
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    gather_residual_activations, 
)
from ..utils.data_utils import *
from pyvene import (
    IntervenableModel,
)
from transformers import set_seed
import transformers, datasets
from typing import Dict, Optional, Sequence, Union, List, Any
from ..inference.utils import prepare_df

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

class BaseModel(object):
    """Base class for all models."""
    training_granularity = "per_concept"
    inference_instance_scope = "shared"
    artifact_directory = None
    requires_training_args = True
    load_trained_weights = True
    uses_intervention_positions = True
    requires_mean_activations = True
    requires_calibration_scale = False
    # Method-scoped additions to the shared ModelParams schema which genuinely
    # affect this method's checkpoint. train.py excludes registered scoped
    # arguments from every method that does not opt in to them.
    training_fingerprint_scoped_args = frozenset()

    def __init__(self, **kwargs):
        pass

    def __str__(self):
        pass

    @classmethod
    def training_fingerprint_context(cls):
        if cls.requires_calibration_scale:
            return {
                "calibration": "paired_assistant_output_per_example_v2"
            }
        return {}

    @property
    def process_rank(self):
        """Return rank zero when running inference outside torchrun."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def make_model(self, **kwargs):
        pass

    def make_dataloader(self, examples, **kwargs):
        pass

    @staticmethod
    def generation_kwargs(max_new_tokens, temperature, do_sample=True):
        """Build consistent generation arguments for sampling or greedy decoding."""
        do_sample = bool(do_sample)
        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            temperature = float(temperature)
            if temperature <= 0:
                raise ValueError(
                    "Sampling generation requires a positive temperature."
                )
            generation_kwargs["temperature"] = temperature
        return generation_kwargs

    def train(self, examples, **kwargs):
        pass

    @torch.no_grad()
    def calibrate(self, examples, **kwargs):
        """Set the steering scale to the largest positive-example token projection onto the learned direction."""
        if "labels" not in examples.columns:
            raise ValueError(
                f"{self.__class__.__name__} calibration requires binary labels."
            )
        positive_examples = examples[examples["labels"] == 1]
        if "assistant_start" not in positive_examples.columns:
            raise ValueError(
                f"{self.__class__.__name__} calibration requires assistant_start "
                "to exclude user and chat-template tokens."
            )
        if positive_examples.empty:
            raise ValueError(
                f"{self.__class__.__name__} calibration requires positive examples."
            )
        if not hasattr(self.ax, "proj"):
            raise TypeError(
                f"{self.__class__.__name__} cannot be calibrated because its "
                "training intervention has no projection direction."
            )

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        batch_size = kwargs.get(
            "batch_size", getattr(self.training_args, "batch_size", 32)
        )
        prefix_length = kwargs.get("prefix_length", 1)
        direction = self.ax.proj.weight.data[0].float()
        per_example_maxima = []
        try:
            for start in range(0, len(positive_examples), batch_size):
                batch = positive_examples.iloc[start:start + batch_size]
                inputs = self.tokenizer(
                    batch["input"].tolist(),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                activations = gather_residual_activations(
                    self.model, self.layer, inputs
                ).detach()
                # The residual layer and the intervention can live on different
                # devices when the base model is dispatched/offloaded.  Perform
                # calibration on the device that owns the gathered residuals.
                activation_device = activations.device
                batch_direction = direction.to(activation_device)
                assistant_starts = torch.as_tensor(
                    batch["assistant_start"].tolist(), device=activation_device
                )
                positions = torch.arange(
                    activations.shape[1], device=activation_device
                ).unsqueeze(0)
                masks = (
                    inputs["attention_mask"].to(activation_device).bool()
                    & (positions >= assistant_starts.unsqueeze(1))
                )
                for example_activations, mask in zip(activations, masks):
                    projections = (
                        example_activations[mask].float().matmul(batch_direction)
                    )
                    if projections.numel() == 0:
                        raise ValueError(
                            f"{self.__class__.__name__} found an example with no "
                            "assistant output tokens during calibration."
                        )
                    per_example_maxima.append(projections.max())
        finally:
            self.tokenizer.padding_side = original_padding_side

        calibration_scale = torch.stack(per_example_maxima).max()
        # Preserve the legacy fallback for non-positive maxima.
        if not torch.isfinite(calibration_scale) or calibration_scale <= 0:
            calibration_scale = direction.new_tensor(50.0)
        self.calibration_scale = calibration_scale.detach().cpu().reshape(1)
        logger.warning(
            "%s max-activation calibration scale: %.4f",
            self.__class__.__name__,
            self.calibration_scale.item(),
        )
        return self.calibration_scale

    def save(self, dump_dir, **kwargs):
        pass

    def load(self, dump_dir, **kwargs):
        pass

    def predict_steer(self, examples, **kwargs):
        pass

    def prepare_inference_examples(self, examples, **kwargs):
        """Return the model-specific input view used for inference."""
        return examples

    def predict_choice_logits(self, examples, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support candidate-logit inference."
        )

    def predict_choice_loglikelihoods(self, examples, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support candidate-loglikelihood "
            "inference."
        )

    def get_logits(self, concept_id, k=10):
        pass

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        pass

    def to(self, device):
        pass


class Model(BaseModel):

    def __init__(self, model, tokenizer, layer, training_args=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        # abstracting layer
        self.layer = layer
        self.training_args = training_args
        self.max_activations = {}
        # Set default device to GPU if available, otherwise CPU
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.seed = kwargs.get("seed", 42)
        self.steering_layers = kwargs.get("steering_layers", None)
        self.num_of_layers = len(self.steering_layers) if self.steering_layers else 1
        self.dump_dir = kwargs.get("dump_dir", None)
        self.use_wandb = kwargs.get("use_wandb", False)
        self.concept_id_map = kwargs.get("concept_id_map")

    def make_model(self, **kwargs):
        pass

    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, examples, **kwargs)
        g = torch.Generator()
        g.manual_seed(self.seed)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True,
            batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"],
            generator=g)
        return train_dataloader
    
    def train(self, examples, **kwargs):
        pass
        
    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        if self.requires_calibration_scale and not hasattr(self, "calibration_scale"):
            raise RuntimeError(
                f"Refusing to save {model_name} without a calibration scale."
            )
        weight_file = dump_dir / f"{model_name}_weight.pt"
        weight = self.ax.proj.weight.data.cpu()
        if weight_file.exists():
            weight = torch.cat([torch.load(weight_file), weight], dim=0)
        torch.save(weight, weight_file)
        
        bias_file = dump_dir / f"{model_name}_bias.pt"
        bias = self.ax.proj.bias.data.cpu()
        if bias_file.exists():
            bias = torch.cat([torch.load(bias_file), bias], dim=0)
        torch.save(bias, bias_file)

        self._save_calibration_scale(dump_dir, model_name)

    def _save_calibration_scale(self, dump_dir, model_name):
        if self.requires_calibration_scale and not hasattr(self, "calibration_scale"):
            raise RuntimeError(
                f"Refusing to save {model_name} without a calibration scale."
            )
        if hasattr(self, "calibration_scale"):
            scale_file = dump_dir / f"{model_name}_scale.pt"
            scale = torch.as_tensor(self.calibration_scale).detach().cpu().reshape(-1)
            if not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(
                    f"Refusing to save invalid calibration scale for {model_name}."
                )
            if scale_file.exists():
                scale = torch.cat([torch.load(scale_file), scale], dim=0)
            torch.save(scale, scale_file)

    def load(self, dump_dir=None, **kwargs):
        priority_mode = kwargs.get("priority_mode", "compute_priority")
        self.priority_mode = priority_mode
        model_name = kwargs.get("model_name", self.__str__())
        scale_file = os.path.join(dump_dir, f"{model_name}_scale.pt")
        scales = None
        if os.path.exists(scale_file):
            scales = torch.load(scale_file, map_location="cpu").reshape(-1)
            if not torch.isfinite(scales).all() or (scales <= 0).any():
                raise ValueError(
                    f"Calibration scale {scale_file} must contain only finite, "
                    "positive values."
                )
        elif self.requires_calibration_scale:
            raise FileNotFoundError(
                f"{self.__class__.__name__} requires a calibration scale, but "
                f"the checkpoint does not contain {scale_file}. Retrain the "
                "method with calibration enabled or use a compatible checkpoint."
            )
        if priority_mode == "mem_priority":
            # prioritize MEM
            concept_id = kwargs.get("concept_id")
            weight = torch.load(
                f"{dump_dir}/{model_name}_weight.pt",
                map_location=torch.device("cpu"),
                mmap=True  # Enable memory mapping
            )
            bias = torch.load(
                f"{dump_dir}/{model_name}_bias.pt",
                map_location=torch.device("cpu"),
                mmap=True  # Enable memory mapping
            )
            weight_rank_1 = weight[concept_id].unsqueeze(0)
            bias_rank_1 = bias[concept_id].unsqueeze(0)
            # load only 1 rank to prevent OOM, and faster inference
            self.make_model(**kwargs)
            self.ax.proj.weight.data = weight_rank_1.to(self.device)
            self.ax.proj.bias.data = bias_rank_1.to(self.device)
            if scales is not None:
                if concept_id >= scales.numel():
                    raise ValueError(
                        f"Calibration scale {scale_file} has {scales.numel()} "
                        f"entries and cannot select concept ID {concept_id}."
                    )
                self.max_activations = {
                    concept_id: float(scales[concept_id])
                }
        elif priority_mode == "compute_priority":
            # prioritize COMPUTE
            print(f"Loading {model_name} from {dump_dir}.")
            weight = torch.load(
                f"{dump_dir}/{model_name}_weight.pt",
                map_location=torch.device("cpu")
            )
            bias = torch.load(
                f"{dump_dir}/{model_name}_bias.pt",
                map_location=torch.device("cpu")
            )
            # override low_rank_dimension in kwargs
            kwargs["low_rank_dimension"] = weight.shape[0]
            self.make_model(**kwargs)
            self.ax.proj.weight.data = weight.to(self.device)
            self.ax.proj.bias.data = bias.to(self.device)
            if scales is not None:
                if scales.numel() != weight.shape[0]:
                    raise ValueError(
                        f"Calibration scale {scale_file} has {scales.numel()} "
                        f"entries, but {model_name}_weight.pt has "
                        f"{weight.shape[0]} concept rows."
                    )
                self.max_activations = {
                    concept_id: float(scale)
                    for concept_id, scale in enumerate(scales)
                }
    


    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.ax.eval()
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"
        # Resolve the model-specific concept ID column.
        concept_id_col = "sae_id" if "sae" in self.__str__().lower() and not kwargs.get("disable_neuronpedia_max_act", False) else "concept_id"
        use_synergy = kwargs.get("use_synergy", False)

        # iterate rows in batch
        batch_size = kwargs.get("batch_size", 64)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        generation_kwargs = self.generation_kwargs(
            eval_output_length,
            temperature,
            kwargs.get("do_sample", True),
        )
        all_generations = []
        all_perplexities = []
        all_strenghts = []
        # Main training loop.
        rank = self.process_rank
        progress_bar = tqdm(
            range(0, len(examples), batch_size),
            position=rank,
            leave=True,
            disable=not kwargs.get("show_progress", True),
        )
        for i in range(0, len(examples), batch_size):
            batch_examples = examples.iloc[i:i+batch_size]
            if use_synergy:
                input_strings = batch_examples['steered_input'].tolist()
            else:
                input_strings = batch_examples['input'].tolist()
            mag = torch.tensor(batch_examples['factor'].tolist()).to(self.device)
            concept_ids = batch_examples["concept_id"].tolist()
            if self.concept_id_map is not None:
                concept_ids = [self.concept_id_map[concept_id] for concept_id in concept_ids]
            idx = torch.tensor(concept_ids).to(self.device)
            max_acts = torch.tensor([
                self.max_activations.get(id, 1.0) 
                for id in batch_examples[concept_id_col].tolist()]).to(self.device)
            # logger.warning(f"Using max activations: {max_acts}")
            # tokenize input_strings
            inputs = self.tokenizer(
                input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            _, generations = self.ax_model.generate(
                inputs, 
                unit_locations=None, intervene_on_prompt=True, 
                subspaces=[{"idx": idx, "mag": mag, "max_act": max_acts, 
                            "prefix_length": kwargs["prefix_length"]}]*self.num_of_layers,
                **generation_kwargs,
            )

            # Decode only the generated suffix.
            prompt_token_counts = [len(input_ids) for input_ids in inputs.input_ids]
            generated_texts = [
                self.tokenizer.decode(generation[prompt_token_count:], skip_special_tokens=True)
                for generation, prompt_token_count in zip(generations, prompt_token_counts)
            ]
            all_generations += generated_texts
            all_strenghts.extend((mag*max_acts).tolist())
            progress_bar.update(1)

        return {
            "steered_generation": all_generations,
            "strength": all_strenghts,
        }

    def choice_input_field(self, examples):
        """Return the model-specific input column used for candidate scoring."""
        return "input"

    def prepare_choice_logits(self, examples, **kwargs):
        """Prepare model-specific state before candidate-logit batches run."""
        if not hasattr(self, "ax") or not hasattr(self, "ax_model"):
            raise RuntimeError(
                f"{self.__class__.__name__} must initialize its steering model "
                "before candidate-logit inference."
            )
        self.ax.eval()

    def finish_choice_logits(self, **kwargs):
        """Release temporary model state created for candidate-logit inference."""

    def choice_forward(self, inputs, batch_examples, **kwargs):
        """Run the standard concept-indexed pyvene steering forward pass."""
        mag = torch.as_tensor(
            batch_examples["factor"].tolist(), device=self.device
        )
        concept_ids = batch_examples["concept_id"].tolist()
        if self.concept_id_map is not None:
            concept_ids = [
                self.concept_id_map[concept_id] for concept_id in concept_ids
            ]
        idx = torch.as_tensor(concept_ids, device=self.device)
        concept_id_col = (
            "sae_id"
            if "sae" in self.__str__().lower()
            and not kwargs.get("disable_neuronpedia_max_act", False)
            else "concept_id"
        )
        max_acts = torch.as_tensor([
            self.max_activations.get(concept_id, 1.0)
            for concept_id in batch_examples[concept_id_col].tolist()
        ], device=self.device)
        _, outputs = self.ax_model(
            base=self.choice_model_inputs(
                inputs,
                last_token_only=not kwargs.get("full_sequence", False),
                logits_to_keep=kwargs.get("choice_logits_to_keep"),
            ),
            unit_locations=None,
            subspaces=[{
                "idx": idx,
                "mag": mag,
                "max_act": max_acts,
                "prefix_length": kwargs["prefix_length"],
            }] * self.num_of_layers,
            use_cache=False,
        )
        return outputs, mag * max_acts

    def choice_model_inputs(
        self, inputs, *, last_token_only=True, logits_to_keep=None
    ):
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        try:
            parameters = inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "num_logits_to_keep" in parameters:
            if logits_to_keep is not None:
                model_inputs["num_logits_to_keep"] = int(logits_to_keep)
            elif last_token_only:
                model_inputs["num_logits_to_keep"] = 1
        return model_inputs

    def choice_input_ids(self, inputs, batch_examples, **kwargs):
        """Return the token IDs aligned with ``choice_forward`` outputs."""
        return inputs["input_ids"]

    def choice_attention_mask(self, inputs, batch_examples, **kwargs):
        return inputs["attention_mask"]

    @staticmethod
    def _last_token_logits(outputs, attention_mask):
        logits = outputs.logits.float()
        if logits.ndim != 3 or logits.shape[0] != attention_mask.shape[0]:
            raise ValueError(
                "Candidate-logit forward must return [batch, sequence, vocabulary] "
                "logits for the input batch."
            )
        if logits.shape[1] == 1:
            return logits[:, 0]
        if logits.shape[1] != attention_mask.shape[1]:
            raise ValueError(
                "Candidate-logit sequence length must align with the input attention "
                "mask unless the model returns only its final-token logits."
            )
        sequence_positions = torch.arange(
            attention_mask.shape[1], device=attention_mask.device
        ).unsqueeze(0)
        positions = (sequence_positions * attention_mask.long()).max(dim=1).values
        return logits[
            torch.arange(logits.shape[0], device=logits.device), positions
        ]

    @torch.no_grad()
    def predict_choice_logits(self, examples, **kwargs):
        """Return next-token candidate logits through the model's own forward path."""
        batch_size = kwargs.get("batch_size", 64)
        all_choice_logits = []
        all_strengths = []
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        progress_bar = None
        try:
            self.prepare_choice_logits(examples, **kwargs)
            progress_bar = tqdm(
                range(0, len(examples), batch_size),
                position=self.process_rank,
                leave=True,
                disable=not kwargs.get("show_progress", True),
            )
            for start in range(0, len(examples), batch_size):
                batch_examples = examples.iloc[start:start + batch_size]
                input_field = self.choice_input_field(batch_examples)
                if input_field not in batch_examples:
                    raise KeyError(
                        f"{self.__class__.__name__} requires evaluation input "
                        f"column '{input_field}'."
                    )
                inputs = self.tokenizer(
                    batch_examples[input_field].tolist(),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                choice_token_ids = torch.as_tensor(
                    [
                        list(token_ids)
                        for token_ids in batch_examples["choice_token_ids"]
                    ],
                    device=self.device,
                    dtype=torch.long,
                )
                if choice_token_ids.ndim != 2:
                    raise ValueError(
                        "choice_token_ids must contain the same number of candidates "
                        "for every example."
                    )
                outputs, strengths = self.choice_forward(
                    inputs, batch_examples, **kwargs
                )
                next_token_logits = self._last_token_logits(
                    outputs,
                    self.choice_attention_mask(
                        inputs, batch_examples, **kwargs
                    ),
                )
                if choice_token_ids.numel() and (
                    choice_token_ids.min() < 0
                    or choice_token_ids.max() >= next_token_logits.shape[-1]
                ):
                    raise ValueError("choice_token_ids contains an invalid token ID.")
                batch_choice_logits = torch.gather(
                    next_token_logits,
                    dim=1,
                    index=choice_token_ids,
                )
                all_choice_logits.extend(batch_choice_logits.cpu().tolist())
                all_strengths.extend(
                    torch.as_tensor(strengths).detach().cpu().tolist()
                )
                progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()
            self.finish_choice_logits(**kwargs)
            self.tokenizer.padding_side = original_padding_side

        return {
            "choice_logits": all_choice_logits,
            "strength": all_strengths,
        }
    def predict_choice_loglikelihoods(self, examples, **kwargs):
        """Score each complete candidate in one teacher-forced forward pass."""
        if "choice_texts" not in examples:
            raise KeyError("choice_loglikelihood inference requires choice_texts.")
        batch_size = int(kwargs.get("batch_size", 64))
        if batch_size < 1:
            raise ValueError("choice_loglikelihood batch_size must be positive.")
        work_rows = []
        candidate_counts = []
        for row_position, (_, row) in enumerate(examples.iterrows()):
            choices = row["choice_texts"]
            if isinstance(choices, str) or not isinstance(choices, (list, tuple)):
                raise TypeError("choice_texts must be a list of candidate strings.")
            if not choices:
                raise ValueError("choice_texts must contain at least one candidate.")
            input_field = self.choice_input_field(examples)
            prompt = str(row[input_field])
            candidate_counts.append(len(choices))
            for choice_position, choice in enumerate(choices):
                choice = str(choice)
                if not choice:
                    raise ValueError("choice_loglikelihood candidates cannot be empty.")
                if not choice[0].isspace():
                    raise ValueError(
                        "choice_loglikelihood candidates must start with whitespace "
                        "so prompt/candidate token boundaries are unambiguous."
                    )
                candidate_ids = self.tokenizer.encode(
                    choice, add_special_tokens=False
                )
                if not candidate_ids:
                    raise ValueError(f"Candidate {choice!r} encoded to no tokens.")
                item = row.copy()
                item[input_field] = prompt + choice
                item["_choice_row_position"] = row_position
                item["_choice_position"] = choice_position
                item["_choice_token_ids"] = [int(value) for value in candidate_ids]
                work_rows.append(item)

        work = pd.DataFrame(work_rows).reset_index(drop=True)
        scores = [[0.0] * count for count in candidate_counts]
        mean_scores = [[0.0] * count for count in candidate_counts]
        strengths_by_row = [None] * len(examples)
        original_padding_side = self.tokenizer.padding_side
        original_truncation_side = self.tokenizer.truncation_side
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        progress_bar = None
        try:
            self.prepare_choice_logits(work, **kwargs)
            progress_bar = tqdm(
                range(0, len(work), batch_size),
                position=self.process_rank,
                leave=True,
                disable=not kwargs.get("show_progress", True),
            )
            for start in range(0, len(work), batch_size):
                batch_examples = work.iloc[start:start + batch_size]
                input_field = self.choice_input_field(batch_examples)
                inputs = self.tokenizer(
                    batch_examples[input_field].tolist(),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                max_candidate_tokens = max(
                    len(values) for values in batch_examples["_choice_token_ids"]
                )
                outputs, strengths = self.choice_forward(
                    inputs,
                    batch_examples,
                    full_sequence=True,
                    choice_logits_to_keep=max_candidate_tokens + 1,
                    **kwargs,
                )
                attention_mask = self.choice_attention_mask(
                    inputs, batch_examples, **kwargs
                )
                input_ids = self.choice_input_ids(
                    inputs, batch_examples, **kwargs
                )
                logits = outputs.logits
                if logits.ndim != 3 or logits.shape[0] != input_ids.shape[0]:
                    raise ValueError(
                        "Candidate loglikelihood requires batched sequence logits."
                    )
                if attention_mask.shape != input_ids.shape:
                    raise ValueError("Candidate attention mask does not align with input IDs.")
                strength_values = torch.as_tensor(strengths).detach().cpu().tolist()
                for offset, (_, item) in enumerate(batch_examples.iterrows()):
                    candidate_ids = list(item["_choice_token_ids"])
                    positions = torch.nonzero(
                        attention_mask[offset], as_tuple=False
                    ).squeeze(1)
                    if len(positions) <= len(candidate_ids):
                        raise ValueError(
                            "Candidate and prompt do not fit together in the model context."
                        )
                    candidate_positions = positions[-len(candidate_ids):]
                    actual_ids = input_ids[offset, candidate_positions].tolist()
                    if actual_ids != candidate_ids:
                        raise ValueError(
                            "Tokenizer changed the prompt/candidate boundary; candidates "
                            "must be valid standalone continuations with leading whitespace."
                        )
                    prediction_positions = candidate_positions - 1
                    output_offset = input_ids.shape[1] - logits.shape[1]
                    output_positions = prediction_positions - output_offset
                    if output_positions.min() < 0 or output_positions.max() >= logits.shape[1]:
                        raise ValueError(
                            "Model did not return enough trailing logits to score "
                            "the complete candidate."
                        )
                    # Convert only candidate-position logits to FP32. Casting the
                    # complete [batch, sequence, vocabulary] tensor would consume
                    # several GiB for Gemma's large vocabulary.
                    candidate_logits = logits[
                        offset, output_positions, :
                    ].float()
                    selected = -torch.nn.functional.cross_entropy(
                        candidate_logits,
                        input_ids[offset, candidate_positions],
                        reduction="none",
                    )
                    score = float(selected.sum().cpu())
                    row_position = int(item["_choice_row_position"])
                    choice_position = int(item["_choice_position"])
                    scores[row_position][choice_position] = score
                    mean_scores[row_position][choice_position] = score / len(candidate_ids)
                    if strengths_by_row[row_position] is None:
                        strengths_by_row[row_position] = strength_values[offset]
                progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()
            self.finish_choice_logits(**kwargs)
            self.tokenizer.padding_side = original_padding_side
            self.tokenizer.truncation_side = original_truncation_side

        return {
            "choice_loglikelihoods": scores,
            "choice_mean_loglikelihoods": mean_scores,
            "strength": strengths_by_row,
        }

    def get_logits(self, concept_id, k=10):
        top_logits, neg_logits = [None], [None]
        if concept_id is not None:
            W_U = self.model.lm_head.weight.T
            W_U = W_U * (self.model.model.norm.weight +
                        torch.ones_like(self.model.model.norm.weight))[:, None]
            W_U -= einops.reduce(
                W_U, "d_model d_vocab -> 1 d_vocab", "mean"
            )

            vocab_logits = self.ax.proj.weight.data[concept_id] @ W_U
            top_values, top_indices = vocab_logits.topk(k=k, sorted=True)
            top_tokens = self.tokenizer.batch_decode(top_indices.unsqueeze(dim=-1))
            top_logits = [list(zip(top_tokens, top_values.tolist()))]
            
            neg_values, neg_indices = vocab_logits.topk(k=k, largest=False, sorted=True)
            neg_tokens = self.tokenizer.batch_decode(neg_indices.unsqueeze(dim=-1))
            neg_logits = [list(zip(neg_tokens, neg_values.tolist()))]
        return top_logits, neg_logits
    
    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        # Training-time calibration is stored with newer checkpoints. Prefer it
        # over the legacy latent parquet path when available.
        if self.max_activations:
            return self.max_activations
        max_activations = {} # sae_id to max_activation
        # Loop over saved latent files in dump_dir.
        for file in os.listdir(dump_dir):
            if file.startswith("latent_") and file.endswith(".parquet"):
                latent_path = os.path.join(dump_dir, file)
                latent = pd.read_parquet(latent_path)
                # loop through unique sorted concept_id
                for concept_id in sorted(latent["concept_id"].unique()):
                    concept_latent = latent[latent["concept_id"] == concept_id]
                    max_act = concept_latent[f"{self.__str__()}_max_act"].max()
                    max_activations[concept_id] = max_act if max_act > 0 else 50
        self.max_activations = max_activations
        return max_activations  

    def to(self, device):
        """Move model to specified device"""
        self.device = device
        if hasattr(self, 'ax'):
            self.ax = self.ax.to(device)
            if hasattr(self, 'ax_model'):
                if isinstance(self.ax_model, IntervenableModel):
                    self.ax_model.set_device(device)
                else:
                    self.ax_model = self.ax_model.to(device)
        return self
