"""Prompt-based steering methods with model-owned prompt construction."""

import asyncio
import json
from pathlib import Path

import httpx
from openai import AsyncOpenAI
import torch

from steerscope.models.language_models import LanguageModel
from steerscope.utils.api_clients import openai_client_credentials
from steerscope.utils.constants import HAS_SYSTEM_PROMPT_MODELS
from steerscope.utils.prompt_utils import get_steering_prompts
from steerscope.templates.prompt_templates import T_GENERATE_PREPEND_STEERING_PROMPT

from .model import Model


SIMPLE_STEERING_PROMPT = (
    "You must answer the question with content related to %s even if it is not "
    "related to the question or it does not make sense."
)
PROMPT_ARTIFACT = "PromptSteering_prompts.json"


class PromptSteering(Model):
    """Steer with one LLM-generated, concept-specific instruction."""

    uses_prompt_generation = True
    requires_training_args = False
    load_trained_weights = True
    uses_intervention_positions = False
    requires_mean_activations = False
    records_model_input = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_by_concept = {}
        self._trained_prompt = None
        self.lm_model_name = kwargs.get("lm_model_name")

    def __str__(self):
        return "PromptSteering"

    @classmethod
    def training_fingerprint_context(cls):
        return {
            "artifact_version": 1,
            "generation_template": T_GENERATE_PREPEND_STEERING_PROMPT,
            "api_tag": "get_steering_prompts",
            "composition": "{instruction}\\n\\nQuestion: {raw_input}",
        }

    def make_model(self, **kwargs):
        pass

    def train(self, examples, **kwargs):
        concept_id = int(kwargs["logging_metadata"]["concept_id"])
        concept = str(kwargs["concept"])
        model_args = self.training_args
        generator_model = getattr(model_args, "lm_model", None)
        if not generator_model:
            raise ValueError(
                "PromptSteering training requires models.PromptSteering.lm_model."
            )
        temperature = float(getattr(model_args, "prompt_temperature", 0.0))
        cache_dir = Path(self.dump_dir) / "prompt_steering_cache" / f"rank_{self.process_rank}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        async def generate():
            client = AsyncOpenAI(
                **openai_client_credentials("generation"),
                timeout=60.0,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_keepalive_connections=100,
                        max_connections=1000,
                    ),
                    headers={"Connection": "close"},
                ),
                # LanguageModel owns retries so attempts are logged and not multiplied.
                max_retries=0,
            )
            language_model = LanguageModel(
                generator_model,
                client,
                dump_dir=cache_dir,
                use_cache=True,
                cache_level="prompt",
                cache_tag="prompt_steering_instructions",
                master_data_dir=kwargs.get("master_data_dir"),
                temperature=temperature,
            )
            try:
                instruction = (await get_steering_prompts(
                    language_model, [concept]
                ))[0].strip()
                language_model.save_cache()
                return instruction
            finally:
                await language_model.close()

        instruction = asyncio.run(generate())
        if not instruction:
            raise ValueError(
                f"PromptSteering generated an empty instruction for concept {concept_id}."
            )
        self._trained_prompt = {
            "concept_id": concept_id,
            "concept": concept,
            "instruction": instruction,
            "generator_model": generator_model,
            "temperature": temperature,
        }

    def save(self, dump_dir, **kwargs):
        if self._trained_prompt is None:
            raise RuntimeError("PromptSteering has no generated instruction to save.")
        path = Path(dump_dir) / PROMPT_ARTIFACT
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self._trained_prompt, file, indent=2, sort_keys=True)

    def load(self, dump_dir=None, **kwargs):
        path = Path(dump_dir) / PROMPT_ARTIFACT
        if not path.exists():
            raise FileNotFoundError(
                f"PromptSteering artifact not found: {path}. Run train first."
            )
        with open(path, encoding="utf-8") as file:
            entries = json.load(file)
        if isinstance(entries, dict) and "concept_id" in entries:
            entries = [entries]
        self.prompt_by_concept = {
            int(entry["concept_id"]): str(entry["instruction"])
            for entry in entries
        }

    def instruction_for(self, concept_id, concept):
        try:
            return self.prompt_by_concept[int(concept_id)]
        except KeyError as error:
            raise KeyError(
                f"PromptSteering has no trained instruction for concept "
                f"{concept_id} ({concept!r})."
            ) from error

    def prepare_inference_examples(self, examples, **kwargs):
        if "raw_input" not in examples:
            raise KeyError("PromptSteering requires evaluator column 'raw_input'.")
        prepared = examples.copy()
        prepared["input"] = prepared.apply(
            lambda row: self._format_task_prompt(
                self.instruction_for(row["concept_id"], row.get("input_concept", "")),
                str(row["raw_input"]),
            ),
            axis=1,
        )
        return prepared

    def _format_task_prompt(self, instruction, raw_input):
        content = f"{instruction}\n\nQuestion: {raw_input}"
        messages = []
        if self.lm_model_name in HAS_SYSTEM_PROMPT_MODELS:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": content})
        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if tokens and self.tokenizer.bos_token_id is not None and tokens[0] == self.tokenizer.bos_token_id:
            tokens = tokens[1:]
        return self.tokenizer.decode(tokens)

    def prepare_choice_logits(self, examples, **kwargs):
        self.model.eval()

    def choice_forward(self, inputs, batch_examples, **kwargs):
        return (
            self.model(
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
        self.model.eval()
        self.tokenizer.padding_side = "left"
        batch_size = kwargs.get("batch_size", 64)
        output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        generation_kwargs = self.generation_kwargs(
            output_length,
            temperature,
            kwargs.get("do_sample", True),
        )
        generations = []
        for start in range(0, len(examples), batch_size):
            batch = examples.iloc[start:start + batch_size]
            inputs = self.tokenizer(
                batch["input"].tolist(), return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
            prompt_token_counts = [len(input_ids) for input_ids in inputs.input_ids]
            generations.extend(
                self.tokenizer.decode(value[length:], skip_special_tokens=True)
                for value, length in zip(output, prompt_token_counts)
            )
        return {"steered_generation": generations}

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        return {}


class SimplePromptSteering(PromptSteering):
    """Steer with the original fixed concept instruction and no artifact."""

    load_trained_weights = False

    def __str__(self):
        return "SimplePromptSteering"

    def load(self, dump_dir=None, **kwargs):
        pass

    def instruction_for(self, concept_id, concept):
        return SIMPLE_STEERING_PROMPT % concept
