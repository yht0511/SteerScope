from pathlib import Path
from .model import Model
import torch, einops
import gc
from tqdm.auto import tqdm
import os
import pandas as pd
from ..utils.constants import EXAMPLE_TAG
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr,
    calculate_l1_losses
)
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers, datasets
from transformers import get_scheduler
from transformers import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer
from ..utils.training import (
    GRADIENT_ACCUMULATION_SEMANTICS,
    is_optimizer_step,
    normalize_loss_for_accumulation,
    optimizer_steps_per_epoch,
)


def _fsdp_gradient_accumulation_steps(
    dataset_size,
    per_device_batch_size,
    world_size,
    configured_steps,
):
    """Cap accumulation at the per-rank epoch length to prevent an all-no_sync FSDP epoch."""
    dataset_size = int(dataset_size)
    per_device_batch_size = int(per_device_batch_size)
    world_size = int(world_size)
    configured_steps = int(configured_steps)
    if dataset_size < 1:
        raise ValueError("FSDP SFT requires a non-empty training dataset.")
    if per_device_batch_size < 1 or world_size < 1 or configured_steps < 1:
        raise ValueError("FSDP SFT batch, world size, and accumulation must be positive.")

    per_rank_examples = (dataset_size + world_size - 1) // world_size
    per_rank_microbatches = (
        per_rank_examples + per_device_batch_size - 1
    ) // per_device_batch_size
    return min(configured_steps, per_rank_microbatches)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

@dataclass
class DataCollator(object):
    
    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_seq_len = max([len(inst["input_ids"]) for inst in instances])
        
        for inst in instances:
            non_pad_len = len(inst["input_ids"])

            _input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)])
            inst["input_ids"] = torch.cat((inst["input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _input_id_paddings)).int()

            _label_paddings = torch.tensor([-100 for _ in range(max_seq_len - non_pad_len+1)])
            inst["labels"] = torch.cat((inst["labels"], _label_paddings))
            
            inst["attention_mask"] = (inst["input_ids"] != self.tokenizer.pad_token_id).int()

        batch_inputs = self.data_collator(instances)
        return batch_inputs


def make_sft_data_module(
    tokenizer: transformers.PreTrainedTokenizer, df, 
    dataset_category="continuation",
    positions="all", # "all_prompt" or "all" or "f1+l1" (pyreft formatting)
    exclude_bos=True,
    prefix_length=1,
    **kwargs
):
    """Make dataset and collator for supervised fine-tuning with kl div loss."""
    if not exclude_bos:
        prefix_length = 0
    
    all_base_input_ids, all_output_ids = [], []
    all_prompt_lengths = []
    for _, row in df.iterrows():
        _input, _output = row["input"], row["output"]
        # prepare input ids
        base_prompt = _input
        if isinstance(_output, float):
            _output = tokenizer.eos_token
        base_input = base_prompt + _output
        base_prompt_ids = tokenizer(
            base_prompt, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_input_ids = tokenizer(
            base_input, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_length = len(base_input_ids)

        # output ids with prompt token mask
        output_ids = base_input_ids.clone()
        output_ids[:base_prompt_length] = -100

        all_base_input_ids.append(base_input_ids)
        all_output_ids.append(output_ids)
        
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "labels": all_output_ids,
    })
    train_dataset.set_format(
        type='torch', columns=['input_ids', 'labels'])

    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = DataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class SFT(Model):
    inference_instance_scope = "per_concept"
    artifact_directory = "sft"
    requires_mean_activations = False

    @classmethod
    def training_fingerprint_context(cls):
        return {
            **super().training_fingerprint_context(),
            "gradient_accumulation_semantics": (
                GRADIENT_ACCUMULATION_SEMANTICS
            ),
        }

    def __init__(self, model, tokenizer, layer, training_args=None, **kwargs):
        super().__init__(model, tokenizer, layer, training_args, **kwargs)
        self.lm_model_name = kwargs.get("lm_model_name")

    def __str__(self):
        return 'SFT'
    
    def make_model(self, **kwargs):
        self.ax_model = self.model
        # PEFT leaves the base model frozen after LoRA is unloaded. SFT is a
        # full-parameter baseline, so restore trainability before optimizing.
        self.ax_model.requires_grad_(True)
        self.concept_id = kwargs.get("concept_id")

    def save(self, dump_dir, **kwargs):
        # folder-based saving
        dump_dir = Path(f"{dump_dir}/sft/{self.concept_id}")
        trainer = getattr(self, "_fsdp_trainer", None)
        if trainer is None:
            dump_dir.mkdir(parents=True, exist_ok=True)
            self.ax_model.save_pretrained(dump_dir)
            return

        # Optimizer moments are not part of the inference artifact. Releasing
        # them first leaves enough headroom for FSDP's full-state-dict gather.
        optimizer = trainer.optimizer
        trainer.optimizer = None
        trainer.lr_scheduler = None
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        # Gather FP32 FSDP state collectively; cast only the saved artifact to BF16.
        state_dict = trainer.accelerator.get_state_dict(trainer.model)
        if trainer.args.should_save:
            for name, value in state_dict.items():
                if torch.is_floating_point(value) and value.dtype != torch.bfloat16:
                    state_dict[name] = value.to(dtype=torch.bfloat16)
            trainer._save(str(dump_dir), state_dict=state_dict)

    def load(self, dump_dir, **kwargs):
        # folder-based loading
        self.concept_id = kwargs.get("concept_id")
        dump_dir = Path(f"{dump_dir}/sft/{self.concept_id}")
        self.ax_model = AutoModelForCausalLM.from_pretrained(
            dump_dir, torch_dtype=torch.bfloat16)
        self.ax_model.to(self.device)

    def _train(self, examples, **kwargs):
        self.ax_model.train()
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(), 
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        accumulation_steps = int(
            self.training_args.gradient_accumulation_steps or 1
        )
        num_microbatches = len(train_dataloader)
        num_training_steps = self.training_args.n_epochs * (
            optimizer_steps_per_epoch(num_microbatches, accumulation_steps)
        )
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
        norm_loss_fn = torch.nn.MSELoss()
        # Main training loop.
        rank = self.process_rank
        progress_bar, curr_step = tqdm(range(num_training_steps), position=rank, leave=True), 0
        
        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
        
                # forward
                outputs = self.ax_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"]
                )
                
                # loss
                loss = outputs.loss
                loss = loss.mean()
                loss = normalize_loss_for_accumulation(
                    loss,
                    step,
                    num_microbatches,
                    accumulation_steps,
                )
                # grads
                loss.backward()

                # Perform optimization step every gradient_accumulation_steps
                if is_optimizer_step(
                    step,
                    num_microbatches,
                    accumulation_steps,
                ):
                    torch.nn.utils.clip_grad_norm_(self.ax_model.parameters(), 1.0)
                    curr_step += 1
                    curr_lr = get_lr(optimizer)
                    # optim
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f" % (curr_lr, loss))
        progress_bar.close()

    def _train_fsdp(self, examples, **kwargs):
        self.ax_model.train()
        data_module = make_sft_data_module(self.tokenizer, examples, **kwargs)
        torch.cuda.empty_cache()

        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else 1
        )
        configured_accumulation_steps = int(
            self.training_args.gradient_accumulation_steps or 1
        )
        accumulation_steps = _fsdp_gradient_accumulation_steps(
            len(data_module["train_dataset"]),
            self.training_args.batch_size,
            world_size,
            configured_accumulation_steps,
        )
        if accumulation_steps != configured_accumulation_steps and (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ):
            print(
                "FSDP SFT capped gradient_accumulation_steps from "
                f"{configured_accumulation_steps} to {accumulation_steps} "
                "to match the per-rank epoch length.",
                flush=True,
            )

        # huggingface trainer with FSDP training
        training_args = TrainingArguments(
            output_dir=str(self.dump_dir),
            logging_steps=1,
            save_strategy="no",
            num_train_epochs=self.training_args.n_epochs,
            per_device_train_batch_size=self.training_args.batch_size,
            gradient_accumulation_steps=accumulation_steps,
            evaluation_strategy="no",  # "no" means no eval loop, only training
            learning_rate=self.training_args.lr,
            weight_decay=self.training_args.weight_decay,
            warmup_ratio=0.00,
            lr_scheduler_type="linear",
            # Foreach AdamW creates a full-size sqrt temporary at the first
            # optimizer step. That transient exceeds two 80GB GPUs for 9B;
            # fused AdamW implements the same update without the extra peak.
            optim="adamw_torch_fused",
            fsdp="full_shard auto_wrap",
            fsdp_transformer_layer_cls_to_wrap="Gemma2DecoderLayer",
            bf16=True,  # whether to use bf16
            do_train=True,
            do_eval=False,
            report_to=[]
        )

        trainer = Trainer(
            model=self.ax_model, 
            tokenizer=self.tokenizer, 
            args=training_args, 
            **data_module
        )
        trainer.train()
        self._fsdp_trainer = trainer

    def train(self, examples, **kwargs):
        if "gemma-2-9b" in self.lm_model_name:
            # huggingface trainer ith FSDP training
            self._train_fsdp(examples, **kwargs)
        else:
            self._train(examples, **kwargs)

    def prepare_choice_logits(self, examples, **kwargs):
        self.ax_model.eval()

    def choice_forward(self, inputs, batch_examples, **kwargs):
        return (
            self.ax_model(
                **self.choice_model_inputs(
                    inputs,
                    last_token_only=not kwargs.get("full_sequence", False),
                    logits_to_keep=kwargs.get("choice_logits_to_keep"),
                ),
                use_cache=False,
            ),
            batch_examples["factor"].tolist(),
        )

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.ax_model.eval()
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"

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
            input_strings = batch_examples['input'].tolist()
            # tokenize input_strings
            inputs = self.tokenizer(
                input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            generations = self.ax_model.generate(
                **inputs,
                **generation_kwargs,
            )

            # Decode only the generated suffix.
            prompt_token_counts = [len(input_ids) for input_ids in inputs.input_ids]
            generated_texts = [
                self.tokenizer.decode(generation[prompt_token_count:], skip_special_tokens=True)
                for generation, prompt_token_count in zip(generations, prompt_token_counts)
            ]
            all_generations += generated_texts
            progress_bar.update(1)

        return {
            "steered_generation": all_generations,
        }
    
