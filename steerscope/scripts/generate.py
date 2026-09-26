
import shutil
import sys
import argparse
import time
import os
import pickle
import random
import json
import csv
import atexit
import requests
import tempfile
import hashlib

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from steerscope.utils.dataset import DatasetFactory
from steerscope.scripts.args.dataset_args import DatasetArgs
from pathlib import Path
from openai import AsyncOpenAI
import httpx, asyncio
from transformers import set_seed
from steerscope.utils.constants import *
from steerscope.utils.api_clients import openai_client_credentials

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

model_name_map = {
    "gemma-2-2b": "google/gemma-2-2b-it",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

MAX_RETRIES = 5
RETRY_DELAY = 1  # in seconds
STATE_FILE = "generate_state.pkl"
METADATA_FILE = "metadata.jsonl"


def load_concepts(dump_dir):
    # 加载来源于sae的概念列表
    sae_concepts = []
    if ".txt" in dump_dir:
        with open(dump_dir, 'r') as file:
            concepts = [line.strip() for line in file.readlines()]
        if concepts[0].startswith("http://") or concepts[0].startswith("https://"):
            logger.warning("Detect external links. Pull concept info from the link.")
            for concept in concepts:
                if "www.neuronpedia.org" not in concept:
                    raise ValueError(f"Pulling from {concept} is not supported.")
                sae_path = concept.split("https://www.neuronpedia.org/")[-1]
                sae_url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
                headers = {"X-Api-Key": os.environ.get("NP_API_KEY")}
                response = requests.get(sae_url, headers=headers).json()
                explanation = response["explanations"][0]["description"]
                sae_concepts += [explanation.strip()]
            return sae_concepts, concepts
        return concepts, ["null"]*len(concepts)
    elif ".csv" in dump_dir:
        # for csv, then the format is <concept>,<url>
        # no http connection is needed
        concepts = []
        with open(dump_dir, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sae_concepts += [row[0]]
                concepts += [row[1]]
        return sae_concepts, concepts
    elif ".json" in dump_dir:
        concepts = []
        # this must be a neuropedia export.
        with open(dump_dir, 'r') as file:
            json_concepts = json.load(file)
        seen_index = set()
        for concept in json_concepts:
            model = concept["modelId"]
            sae_model = concept["layer"]
            subspace_id = concept["index"]
            if subspace_id in seen_index:
                continue  # Use the first description.
            seen_index.add(subspace_id)
            sae_concepts += [concept["description"].strip()]
            concepts += [f"https://www.neuronpedia.org/{model}/{sae_model}/{subspace_id}"]
        return sae_concepts, concepts
    else:
        raise ValueError(f"Unsupported file type: {dump_dir}.")


def save_df_to_parquet_safely(df, final_path):
    # Create temporary file in the same directory as the target
    dirname = os.path.dirname(os.path.abspath(final_path))
    with tempfile.NamedTemporaryFile(delete=False, dir=dirname, suffix='.parquet.tmp') as tmp:
        temp_path = tmp.name
        try:
            # Write to temporary file first
            df.to_parquet(temp_path, index=False)
            # Ensure data is written to disk
            os.fsync(tmp.fileno())
        except Exception as e:
            os.unlink(temp_path)  # Clean up temp file
            raise e

    try:
        # Atomic rename operation
        os.rename(temp_path, final_path)
    except Exception as e:
        os.unlink(temp_path)  # Clean up temp file
        raise e


def _atomic_pickle(value, final_path):
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=final_path.parent, suffix=".pkl.tmp"
    ) as tmp:
        temp_path = Path(tmp.name)
        try:
            pickle.dump(value, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, final_path)


def _atomic_jsonl(entries, final_path):
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False,
        dir=final_path.parent, suffix=".jsonl.tmp"
    ) as tmp:
        temp_path = Path(tmp.name)
        try:
            for entry in entries:
                tmp.write(json.dumps(entry) + "\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, final_path)


def load_metadata_flatten(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    metadata = []
    with open(Path(metadata_path) / METADATA_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            concept, ref =data["concept"], data["ref"]
            concept_genres_map = data["concept_genres_map"][concept]
            ref = data["ref"]
            flatten_data = {
                "concept": concept,
                "ref": ref,
                "concept_genres_map": {concept: concept_genres_map},
                "concept_id": data["concept_id"]
            }
            metadata += [flatten_data]  # Return the metadata as is
    return metadata


def save(
    dump_dir, state, concept_id,
    concept, concept_genres_map,
    ref, partition, current_df):
    """
    Save the current state, metadata, and DataFrame using Parquet format.
    """
    # Commit data and metadata before advancing resumable state.
    metadata_path = Path(dump_dir) / METADATA_FILE
    metadata_entry = {
        "concept_id": concept_id,
        "concept": concept,
        "ref": ref,
        "concept_genres_map": concept_genres_map,
    }

    # Save DataFrame using Parquet
    rotation_freq = 500
    file_index = concept_id // rotation_freq
    if file_index == 0:
        df_path = os.path.join(dump_dir, f"{partition}_data.parquet")
    else:
        df_path = os.path.join(dump_dir, f"{partition}_data_{file_index}.parquet")
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        if "concept_id" in existing_df.columns:
            existing_df = existing_df[
                existing_df["concept_id"].astype(int) != int(concept_id)
            ]
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df
    save_df_to_parquet_safely(combined_df, df_path)

    metadata = []
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = [json.loads(line) for line in f if line.strip()]
    metadata = [
        entry for entry in metadata
        if int(entry["concept_id"]) != int(concept_id)
    ]
    metadata.append(metadata_entry)
    metadata.sort(key=lambda entry: int(entry["concept_id"]))
    _atomic_jsonl(metadata, metadata_path)

    _atomic_pickle(state, Path(dump_dir) / STATE_FILE)


def load_state(dump_dir):
    """Load generation state when present."""
    state_path = os.path.join(Path(dump_dir), STATE_FILE)
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state = pickle.load(f)
            return state
    return None


def frozen_training_concepts(concept_path, dump_dir, seed, max_concepts):
    """Persist concept order before API calls and reuse it on every restart."""
    path = Path(dump_dir) / "concept_selection.jsonl"
    if path.is_file():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if any(row["seed"] != seed or row["max_concepts"] != max_concepts for row in rows):
            raise ValueError("Concept selection settings changed; use a new output directory.")
        return [row["concept"] for row in rows], [row["ref"] for row in rows]
    concepts, refs = load_concepts(str(concept_path))
    pairs = list(zip(concepts, refs))
    if max_concepts is not None:
        random.Random(seed).shuffle(pairs)
        pairs = pairs[:max_concepts]
    source_hash = hashlib.sha256(Path(concept_path).read_bytes()).hexdigest()
    rows = [dict(concept_id=i, concept=c, ref=ref, seed=seed,
                 max_concepts=max_concepts, source_sha256=source_hash)
            for i, (c, ref) in enumerate(pairs)]
    metadata = Path(dump_dir) / METADATA_FILE
    if metadata.is_file():
        for line in metadata.read_text().splitlines():
            prior = json.loads(line)
            current = rows[int(prior["concept_id"])]
            if current["concept"] != prior["concept"] or current["ref"] != prior["ref"]:
                raise ValueError("Existing generation metadata disagrees with the concept source.")
    _atomic_jsonl(rows, path)
    return [c for c, _ in pairs], [ref for _, ref in pairs]


def generate_training(args):
    dump_dir = args.dump_dir
    dump_dir = Path(dump_dir) / "generate"
    dump_dir.mkdir(parents=True, exist_ok=True)

    concept_path = args.concept_path
    num_of_examples = args.num_of_examples
    max_concepts = args.max_concepts

    set_seed(args.seed)
    all_concepts, all_refs = frozen_training_concepts(concept_path, dump_dir, args.seed, max_concepts)

    concept2id = {concept: i for i, concept in enumerate(all_concepts)}
    concepts = list(zip(all_concepts, all_refs))

    # Load the state if it exists.
    state = load_state(dump_dir)
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept index: {start_concept_id}")
    if start_concept_id >= len(concepts):
        logger.warning(f"Datasets for all concepts have been generated. Exiting.")
        return

    # Create a new OpenAI client.
    client = AsyncOpenAI(
        **openai_client_credentials("generation"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000
            ),
            headers={"Connection": "close"},
        ),
        # LanguageModel owns retries so attempts are logged and not multiplied.
        max_retries=0,
    )

    # Only the tokenizer is needed to crop API responses to the configured length.
    model_name = args.model_name or model_name_map[all_refs[0].split("/")[3]]
    tokenizer =  AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    tokenizer.padding_side = "right"

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Init the dataset factory.
    dataset_factory = DatasetFactory(
        None, client, tokenizer, args.dataset_category, num_of_examples, args.output_length,
        dump_dir, use_cache=args.lm_use_cache, master_data_dir=args.master_data_dir,
        seed=args.seed, lm_model=args.lm_model, start_concept_id=start_concept_id,
        api_concurrency=args.api_concurrency,
    )
    atexit.register(dataset_factory.close)
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    progress_bar = tqdm(range(start_concept_id, len(concepts)), desc="Processing concept")
    only_one_concept = True if len(concepts) == 1 else False
    data_concept_id = start_concept_id
    for concept_id in progress_bar:
        concept, ref = concepts[concept_id]
        print(f"Generating for concept: {concept}...")

        # prepare concept related data.
        # 判断概念属于哪一类
        concept_genres_map = \
            dataset_factory.prepare_genre_concepts([concept])
        # generate with retry mechanism.
        # 正式运行生成
        # try:
        current_df = dataset_factory.create_train_df(
            concept, num_of_examples, concept_genres_map,
            output_length=args.output_length,
            current_concept_id=data_concept_id,
            only_one_concept=only_one_concept,
        )
        current_df["concept_id"] = data_concept_id
        # except Exception as e:
        #     logger.warning(f"Failed to create training data for group {concept_id}: {e}")
        #     continue # continue to the next group.

        # Save the generated DataFrame, metadata, and current state
        save(
            dump_dir, {"concept_id": concept_id + 1}, data_concept_id,
            concept, concept_genres_map,
            ref, "train", current_df)
        data_concept_id += 1

    logger.warning(f"Finished creating dataset.")


def save_dpo(

    dump_dir, concept_id, partition,
    current_df):
    # This function saves DataFrames per rank per partition (latent or steering)
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Save DataFrame using Parquet
    rotation_freq = 500
    file_index = concept_id // rotation_freq if concept_id != -1 else 0
    if file_index == 0:
        df_path = os.path.join(dump_dir, f"{partition}_train_data.parquet")
    else:
        df_path = os.path.join(dump_dir, f"{partition}_train_data_{file_index}.parquet")
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df
    combined_df.to_parquet(df_path, index=False)


def save_state_dpo(dump_dir, state, partition):
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save state
    state_path = os.path.join(dump_dir, f"{partition}_{STATE_FILE}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)


def load_state_dpo(dump_dir, partition):
    """
    Load the state from a file if it exists.
    """
    state_path = os.path.join(f"{dump_dir}/generate", f"{partition}_{STATE_FILE}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def generate_dpo_training(args):
    dump_dir = args.dump_dir
    dump_dir = Path(dump_dir) / "generate"
    args.data_dir = f"{args.dump_dir}/generate"
    # check the generate directory exists.
    if not os.path.exists(dump_dir):
        raise ValueError(f"Generate directory does not exist: {dump_dir}")
    # check the train_data.parquet exists.
    if not os.path.exists(os.path.join(dump_dir, "train_data.parquet")):
        raise ValueError(f"Train data does not exist: {os.path.join(dump_dir, 'train_data.parquet')}")

    concept_path = args.concept_path
    num_of_examples = args.num_of_examples
    max_concepts = args.max_concepts

    # Load and optionally shuffle concepts
    set_seed(int(args.seed))

    # Configure the logger per rank
    logger.setLevel(logging.WARNING)  # Set the logging level as desired

    # Create a logging formatter that includes the rank
    formatter = logging.Formatter(
        fmt=f'%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S'
    )

    # Create a console handler and set its formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(console_handler)

    # Optionally, create a file handler per rank
    """
    log_file = f'log_rank_{rank}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    """
    data_dir = args.data_dir
    num_of_examples = args.num_of_examples or args.steering_num_of_examples
    if num_of_examples is None:
        raise ValueError("num_of_examples or steering_num_of_examples must be set for DPO generation.")
    metadata = load_metadata_flatten(data_dir)
    # Get list of all concept_ids
    concept_ids = list(range(len(metadata)))
    concepts = [metadata[i]["concept"] for i in concept_ids]

    # Load the state if it exists.
    state = load_state_dpo(args.dump_dir, "dpo")
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept index: {start_concept_id}")
    if start_concept_id >= len(concept_ids):
        logger.warning(f"Datasets for all concepts have been generated. Exiting.")
        return

    # Create a new OpenAI client.
    client = AsyncOpenAI(
        **openai_client_credentials("generation"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000
            ),
            headers={"Connection": "close"},
        ),
        # LanguageModel owns retries so attempts are logged and not multiplied.
        max_retries=0,
    )

    tokenizer_model_name = args.model_name
    if tokenizer_model_name is None:
        tokenizer_model_name = model_name_map[metadata[0]["ref"].split("/")[3]]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_name, model_max_length=512
    )
    tokenizer.padding_side = "right"

    # DPO generation also uses the API only; the tokenizer just crops responses.
    dataset_factory = DatasetFactory(
        None, client, tokenizer, args.dataset_category, num_of_examples, int(args.output_length),
        dump_dir, use_cache=args.lm_use_cache, master_data_dir=args.master_data_dir,
        seed=int(args.seed), lm_model=args.lm_model, start_concept_id=start_concept_id,
        is_inference=True, is_dpo=True, concepts=concepts,
        disable_local_model=args.disable_local_model,
        api_concurrency=args.api_concurrency,
    )
    atexit.register(dataset_factory.close)
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    existing_df = pd.read_parquet(os.path.join(dump_dir, "train_data.parquet"))
    existing_df = existing_df.rename(columns={'output': 'winning_output'})

    progress_bar = tqdm(range(start_concept_id, len(metadata)), desc="Processing concept")
    for start_idx in progress_bar:
        concept_id = metadata[start_idx]["concept_id"]
        concept = metadata[start_idx]["concept"]
        print(f"Generating for concept: {concept}...")
        current_df = existing_df[existing_df["concept_id"] == concept_id].copy()
        dpo_df = dataset_factory.create_dpo_df(
            current_df,
            output_length=int(args.output_length),
            steer_data_type = args.steer_data_type
        )

        save_dpo(dump_dir, concept_id, 'dpo', dpo_df)
        logger.warning(f"Saved dpo dataset for concept {concept_id} to dpo_train_data.parquet")
        # Record the next metadata position after durable output is written.
        current_state = {'concept_id': start_idx + 1}
        save_state_dpo(dump_dir, current_state, 'dpo')

    logger.warning(f"Finished creating DPO dataset.")


def main():
    custom_args = [
        {
            'args': ['--mode'],
            'kwargs': {
                'type': str,
                'default': "training",
                'help': 'The generation mode.'
            }
        }
    ]

    generate_args = DatasetArgs(custom_args=custom_args, section="generate")
    logger.warning("Generating datasets with the following configuration:")
    logger.warning(generate_args)

    if generate_args.mode == "training":
        generate_training(generate_args)
    elif generate_args.mode == "dpo_training":
        generate_dpo_training(generate_args)
    else:
        raise ValueError(f"Invalid mode: {generate_args.mode}")


if __name__ == "__main__":
    main()
