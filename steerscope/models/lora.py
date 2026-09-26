from pathlib import Path
from .model import Model
import peft
from peft import PeftModel, LoraConfig, get_peft_model
import torch, einops
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
from transformers import get_scheduler
from transformers import set_seed
from ..utils.training import (
    GRADIENT_ACCUMULATION_SEMANTICS,
    is_optimizer_step,
    normalize_loss_for_accumulation,
    optimizer_steps_per_epoch,
)


class LoRA(Model):
    inference_instance_scope = "per_concept"
    artifact_directory = "lora"
    requires_mean_activations = False

    @classmethod
    def training_fingerprint_context(cls):
        return {
            **super().training_fingerprint_context(),
            "gradient_accumulation_semantics": (
                GRADIENT_ACCUMULATION_SEMANTICS
            ),
        }

    def __str__(self):
        return 'LoRA'
    
    def make_model(self, **kwargs):
        peft_config = LoraConfig(
            r=self.training_args.low_rank_dimension,
            lora_alpha=self.training_args.lora_alpha,
            target_modules=self.training_args.lora_components,
            layers_to_transform=self.training_args.lora_layers,
            use_rslora=True, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM"
        )
        ax_model = get_peft_model(self.model, peft_config)
        ax_model.to(self.device)
        ax_model.print_trainable_parameters()
        self.ax_model = ax_model
        # lora is concept-ful due to its nature.
        self.concept_id = kwargs.get("concept_id")

    def save(self, dump_dir, **kwargs):
        # folder-based saving
        dump_dir = Path(f"{dump_dir}/lora/{self.concept_id}")
        dump_dir.mkdir(parents=True, exist_ok=True)
        self.ax_model.save_pretrained(dump_dir)

    def load(self, dump_dir, **kwargs):
        # folder-based loading
        self.concept_id = kwargs.get("concept_id")
        dump_dir = Path(f"{dump_dir}/lora/{self.concept_id}")
        self.ax_model = PeftModel.from_pretrained(
            self.model, dump_dir)

    def train(self, examples, **kwargs):
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
