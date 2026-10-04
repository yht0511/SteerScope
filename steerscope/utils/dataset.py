import asyncio
import os
import time

import pandas as pd
from datasets import load_from_disk

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


from ..models.language_models import LanguageModel
from ..utils.prompt_utils import *
from ..utils.constants import *


async def run_tasks(tasks):
    # Gather and run all provided tasks concurrently, and collect their results
    results = await asyncio.gather(*tasks)
    return results

class DatasetFactory(object):
    """Main class of async generating training pairs for two subspaces"""

    def __init__(
        self, model, client, tokenizer, dataset_category, num_of_examples, output_length, dump_dir, 
        use_cache=True, master_data_dir=None, start_concept_id=0, is_chat_model=True, include_system_prompt=False, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir

        # prepare lm model
        lm_model = kwargs.get("lm_model", "DeepSeek-V3.2-Instruct")
        self.api_concurrency = int(kwargs.get("api_concurrency", 64))
        if self.api_concurrency < 2 or self.api_concurrency % 2 != 0:
            raise ValueError("api_concurrency must be an even integer of at least 2")
        self.use_cache = use_cache
        self.lm_model = LanguageModel(
            lm_model, client, dump_dir, 
            use_cache=use_cache, master_data_dir=master_data_dir,
            api_batch_size=self.api_concurrency // 2,
        )
        self.seed = kwargs.get("seed", 42)
        self.logger = kwargs.get("logger", logger)

        # load seed sentences
        self.seed_sentences = load_from_disk(os.path.join(master_data_dir, "seed_sentences"))
        self.seed_instructions = load_from_disk(os.path.join(master_data_dir, "seed_instructions"))
        self.dataset_category = dataset_category
        self.overwrite_inference_data_dir = kwargs.get("overwrite_inference_data_dir", None)
        if self.overwrite_inference_data_dir is not None and os.path.exists(self.overwrite_inference_data_dir):
            # load pre-generated data
            self.pregenerated_inference_df = pd.read_parquet(os.path.join(self.overwrite_inference_data_dir, "latent_eval_data.parquet"))
            self.logger.warning(f"Loaded pre-generated data from {self.overwrite_inference_data_dir}.")

    def save_cache(self):
        """Save the language model cache before exiting"""
        self.lm_model.save_cache()

    def close(self):
        """Close the persistent cache and HTTP client after final reporting."""
        asyncio.run(self.lm_model.close())

    def reset_stats(self):
        """Reset API costs"""
        if self.use_cache:
            self.lm_model.dump()
        self.lm_model.stats.print_report()
        self.lm_model.stats.reset()

    def prepare_genre_concepts(self, concepts, **kwargs):
        start = time.time()
        tasks = []

        # prepare genres if needed
        concept_genres_map = kwargs.get("concept_genres_map", None)
        if concept_genres_map is not None:
            return concept_genres_map
        if concept_genres_map is None:
            logger.warning("Creating genre for the inputs (not provided).")
            genre_task = get_concept_genres(
                self.lm_model, concepts, 
                api_tag=kwargs.get("api_tag", "")
            )
            tasks.append(genre_task)
        
        # run tasks
        res = asyncio.run(run_tasks(tasks))
        concept_genres_map = res[0]

        # log
        logger.warning(f"Init finished in {round(time.time() - start, 3)} sec.")
        return concept_genres_map

    def prepare_concepts(self, concepts, **kwargs):
        if self.overwrite_inference_data_dir is not None and os.path.exists(self.overwrite_inference_data_dir):
            self.logger.warning("Using pre-generated metadata.")
            return {}, {}

        start = time.time()
        tasks = []
        
        # contrast concepts
        logger.warning("Creating contrast concepts for the inputs.")
        contrast_task = get_contrast_concepts(
            self.lm_model, concepts, kwargs.get("contrast_concepts_map", None), 
            api_tag=kwargs.get("api_tag", ""))
        tasks.append(contrast_task)

        # prepare genres if needed
        concept_genres_map = kwargs.get("concept_genres_map", None)
        if concept_genres_map is None:
            logger.warning("Creating genre for the inputs (not provided).")
            genre_task = get_concept_genres(
                self.lm_model, concepts, 
                api_tag=kwargs.get("api_tag", "")
            )
            tasks.append(genre_task)
        
        # run tasks
        res = asyncio.run(run_tasks(tasks))
        contrast_concepts_map = res[0]
        if len(res) > 1:
            concept_genres_map = res[1]

        # log
        for concept in concepts:
            logger.warning(f"Found {len(contrast_concepts_map[concept])} contrast concept(s) for concept: {concept}.")
        logger.warning(f"Init finished in {round(time.time() - start, 3)} sec.")
        return concept_genres_map, contrast_concepts_map

    def create_imbalance_eval_df(self, subset_n, factor=100):
        # Use one shared imbalanced negative set across concepts.
        self.logger.warning(
            "Using pre-generated data for imbalanced eval dataset "
            "(positive examples only occupy less than 1% of the dataset).")
        if factor is None:
            factor = 100
        negative_n_upsamples = int(subset_n*factor) # 100x more negative examples than positive ones.
        # Sample negatives from other concepts.
        negative_df = self.pregenerated_inference_df[self.pregenerated_inference_df["category"] == "negative"].copy()
        negative_df = negative_df.sample(negative_n_upsamples, random_state=self.seed)
        negative_df["output_concept"] = EMPTY_CONCEPT
        # overwrite negative df fields to be compatible.
        concept_df = negative_df
        return concept_df

    def create_train_df(self, concept, n, concept_genres_map, **kwargs):
        tokenizer = self.tokenizer
        
        start = time.time()
        self.logger.warning("Creating dataframe.")
        all_examples = []

        output_length = kwargs.get("output_length", 32)

        functors = []
        if self.dataset_category == "continuation":
            functors = [continue_with_concept, continue_without_concept]
        else:
            functors = [response_with_concept, response_without_concept]
        
        # random sentence or instruction
        genre = concept_genres_map[concept][0]
        per_category_n = int(n // 2)
        concepts_random_content, source_indices = get_random_content(
            self.seed_sentences if self.dataset_category == "continuation" else self.seed_instructions, 
            tokenizer=tokenizer, count=per_category_n,
            genres=[genre], concepts=[concept], length=None, split="train",
            seed=self.seed, return_indices=True,
        )

        paired_content = concepts_random_content[concept]

        # Generate positive and negative answers for exactly the same prompts.
        positive_task = functors[0](
            self.lm_model, self.tokenizer, 
            concepts=[concept] * len(paired_content),
            content=paired_content, length=output_length)
        negative_task = functors[1](
            self.lm_model, self.tokenizer,
            concepts=[concept] * len(paired_content),
            content=paired_content, length=output_length)
        positive_outputs, negative_outputs = asyncio.run(
            run_tasks([positive_task, negative_task])
        )
        expected_outputs = len(paired_content)
        if (
            len(positive_outputs) != expected_outputs
            or len(negative_outputs) != expected_outputs
        ):
            raise ValueError(
                "API generation returned incomplete training pairs: "
                f"expected {expected_outputs}, got {len(positive_outputs)} positive "
                f"and {len(negative_outputs)} negative outputs."
            )
        for pair_id, (prompt, positive_output, negative_output) in enumerate(
            zip(paired_content, positive_outputs, negative_outputs)
        ):
            for label, answer in (("positive", positive_output), ("negative", negative_output)):
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(
                        f"Empty {label} training answer: concept={concept!r}, pair_id={pair_id}."
                    )
            all_examples += [[
                prompt, positive_output, concept, genre, "positive",
                self.dataset_category, pair_id, source_indices[pair_id],
            ], [
                prompt, negative_output, EMPTY_CONCEPT, genre, "negative",
                self.dataset_category, pair_id, source_indices[pair_id],
            ]]

        # update the column definitions of the DataFrame
        df = pd.DataFrame(
            all_examples, 
            columns = [
                'input', 'output', 'output_concept', 'concept_genre', 'category', 'dataset_category'
                , 'pair_id', 'source_instruction_id'
            ])
        self.logger.warning(f"Finished creating current dataframe in {round(time.time() - start, 3)} sec.")
        return df

    def create_dpo_df(self, existing_df, **kwargs):
        start = time.time()
        self.logger.warning("Creating dataframe.")
        output_length = kwargs.get("output_length", 32)
        steer_data_type = kwargs.get("steer_data_type", "concept")

        positive_df = existing_df[existing_df["category"] == "positive"].copy()
        if positive_df.empty:
            raise ValueError("DPO generation requires at least one positive example.")
        positive_prompts = positive_df["input"].tolist()
        concept = positive_df["output_concept"].iloc[0]
        if steer_data_type != "concept":
            raise ValueError(
                "API-only DPO generation currently supports steer_data_type='concept'."
            )
        losing_output_task = response_without_concept(
            self.lm_model,
            self.tokenizer,
            concepts=[concept] * len(positive_prompts),
            content=positive_prompts,
            length=output_length,
        )
        losing_outputs = asyncio.run(run_tasks([losing_output_task]))[0]
        positive_df["losing_output"] = losing_outputs

        self.logger.warning(f"Finished creating current dataframe in {round(time.time() - start, 3)} sec.")
        return positive_df
