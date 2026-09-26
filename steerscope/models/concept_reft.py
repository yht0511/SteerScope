from pathlib import Path
from .model import Model
import torch, einops
from tqdm.auto import tqdm
import os
import pandas as pd
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from .interventions import (
    LoraIntervention,
    PreferenceLoreftIntervention,
)
from ..utils.constants import EXAMPLE_TAG
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr,
    calculate_l1_losses
)
from ..utils.data_utils import (
    parse_positions, 
    get_intervention_locations,
    InterventionDataCollator
)
from dataclasses import dataclass
from transformers import set_seed, get_scheduler, DataCollatorForSeq2Seq, DataCollator
import transformers, datasets
from typing import Dict, Optional, Sequence, Union, List, Any

from .concept_model import *
from .reft import FactorScaledPositionedChoiceLogitsMixin, make_eval_data_module

# using pyreft out-of-the-box
import pyreft

class ConceptLoReFT(FactorScaledPositionedChoiceLogitsMixin, ConceptModel):
    inference_instance_scope = "per_concept"
    artifact_directory = "concept_loreft"

    def __str__(self):
        return 'ConceptLoReFT'

    def make_model(self, **kwargs):
        # there is one type of intervention throughout
        reft_layers = [self.layer] if self.training_args.reft_layers is None else self.training_args.reft_layers
        self.number_of_interventions = len(reft_layers)
        ax = []
        for _ in range(self.number_of_interventions):
            _ax = PreferenceLoreftIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                dropout=kwargs.get("dropout", 0.0),
            )
            _ = _ax.to(self.device)
            _ = _ax.train()
            ax.append(_ax)
        self.ax = ax
        
        # This implementation supports one component.
        ax_config = IntervenableConfig(representations=[{
                "layer": l, "component": f"model.layers[{l}].output" if "gemma-3" in self.lm_model_name else "block_output",
                "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
                # each layer has its own intervention
                "intervention": ax[i]
            } for i, l in enumerate(reft_layers)])
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model
        # lora is concept-ful due to its nature.
        self.concept_id = kwargs.get("concept_id")
        self.number_of_interventions = len(reft_layers)
        self.intervention_positions = self.training_args.reft_positions
        self.preference_pairs = kwargs.get("preference_pairs", ["orig_add"])
        
    def save(self, dump_dir, **kwargs): 
        # gonna to the folder-based saving, way more easier than 3d matrix handling.
        dump_dir = Path(f"{dump_dir}/concept_loreft/{self.concept_id}")
        dump_dir.mkdir(parents=True, exist_ok=True)
        self.ax_model.save(dump_dir) # calls pyvene intervention save

    def load(self, dump_dir=None, **kwargs):
        priority_mode = kwargs.get("priority_mode", "compute_priority")
        
        self.priority_mode = priority_mode
        # folder-based loading
        self.concept_id = kwargs.get("concept_id")
        dump_dir = Path(f"{dump_dir}/concept_loreft/{self.concept_id}")
        self.make_model(**kwargs)
        self.ax_model.load_intervention(dump_dir, include_model=False)

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"
        # Resolve the model-specific concept ID column.
        concept_id_col = "concept_id"
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
        all_strenghts = []
        # Main training loop.
        rank = self.process_rank

        data_module = make_eval_data_module(
            self.tokenizer, self.model, examples, 
            positions=self.intervention_positions,
            num_interventions=self.number_of_interventions, 
            nonstop=True, share_weights=True
        )
        eval_dataloader = DataLoader(
            data_module["eval_dataset"], shuffle=False,
            batch_size=kwargs.get("batch_size"), 
            collate_fn=data_module["data_collator"])
        
        torch.cuda.empty_cache()
        all_batch_examples = [examples.iloc[i:i+batch_size] for i in range(0, len(examples), batch_size)]
        progress_bar = tqdm(
            all_batch_examples,
            position=rank,
            leave=True,
            disable=not kwargs.get("show_progress", True),
        )
        for i, batch in enumerate(eval_dataloader):
            # prepare input
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            unit_locations={"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )}
            batch_examples = all_batch_examples[i]
            mag = torch.tensor(batch_examples['factor'].tolist()).to(self.device)
            _, generations = self.ax_model.generate(
                {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}, 
                unit_locations=unit_locations, intervene_on_prompt=True, 
                subspaces=[{"steering_factor": mag}]*self.number_of_interventions,
                **generation_kwargs,
            )
            
            # Decode only the generated suffix.
            prompt_token_counts = [len(input_ids) for input_ids in inputs["input_ids"]]
            generated_texts = [
                self.tokenizer.decode(generation[prompt_token_count:], skip_special_tokens=True)
                for generation, prompt_token_count in zip(generations, prompt_token_counts)
            ]
            all_generations += generated_texts
            all_strenghts.extend((mag).tolist())
            progress_bar.update(1)

        return {
            "steered_generation": all_generations,
            "strength": all_strenghts,
        }
