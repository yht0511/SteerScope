import os
import argparse
import hashlib
import yaml
import json
import glob
import pickle
import tempfile
import gc
import torch
import shutil
import requests
import datetime
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from pathlib import Path
from steerscope.scripts.args.training_args import TrainingArgs
from steerscope.scripts.args.dataset_args import DatasetArgs
from steerscope.utils.constants import *
from steerscope.utils.model_utils import get_prefix_length, get_suffix_length
from transformers import set_seed
import torch.distributed as dist
import sys
from torch.utils.data import DataLoader
from steerscope.models.sae import GemmaScopeSAE, save_pruned_sae
from steerscope.evaluation.version import generated_data_identity
from steerscope.utils.training_subset import (
    select_balanced_subset,
    select_unpaired_subset,
)
from steerscope.utils.training_seed import (
    TRAINING_RECIPE_VERSION,
    TRAINING_SEED_DERIVATION,
    derive_training_seed,
)

# all supported methods
import steerscope

import logging

# Initialize the logger
logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"
STATE_FILE = "train_state.pkl"
METADATA_FILE = "metadata.jsonl"
TRAIN_STATE_VERSION = 2
CHECKPOINTS_DIR = "checkpoints"
ARTIFACT_MANIFEST_FILE = "artifact_manifest.json"
METHOD_SCOPED_FINGERPRINT_ARGS = frozenset()


def data_generator(data_dir, use_dpo_loss=False):
    """Yield concept groups from ordered training parquet shards."""
    # Gather all file paths in the directory
    if use_dpo_loss:
        file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) \
            if f.startswith('dpo_train_data') and f.endswith('.parquet') and "combined" not in f]
    else:
        file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) \
            if f.startswith('train_data') and f.endswith('.parquet') and "combined" not in f]

    # Sort files: 'train_data.parquet' comes first, then 'train_data_X.parquet' sorted by X
    def extract_index(file_name):
        if use_dpo_loss:
            if file_name == 'dpo_train_data.parquet':
                return -1  # Ensure 'train_data.parquet' comes first
            else:
                # Extract the number X from 'train_data_X.parquet'
                return int(file_name.split('_')[-1].split('.')[0])
        else:
            if file_name == 'train_data.parquet':
                return -1  # Ensure 'train_data.parquet' comes first
            else:
                # Extract the number X from 'train_data_X.parquet'
                return int(file_name.split('_')[-1].split('.')[0])

    file_paths.sort(key=lambda x: extract_index(os.path.basename(x)))

    for file_path in file_paths:
        df = pd.read_parquet(file_path)
        concept_ids = df['concept_id'].unique()
        concept_ids.sort()
        for concept_id in concept_ids:
            if concept_id >= 0:
                df_subset = df[df['concept_id'] == concept_id]
                yield (concept_id, df_subset)


def select_training_concepts(
    concept_frames,
    *,
    concept_ids=None,
    max_concepts=None,
    num_concepts=None,
    seed=42,
    model_names=(),
):
    """Select an explicit, deterministic sampled, or legacy prefix panel."""
    frames = list(concept_frames)
    configured = sum(
        value is not None
        for value in (concept_ids, max_concepts, num_concepts)
    )
    if configured > 1:
        raise ValueError(
            "train.concept_ids, train.num_concepts, and train.max_concepts "
            "are mutually exclusive."
        )
    if concept_ids is not None:
        if isinstance(concept_ids, (str, bytes)):
            raise TypeError("train.concept_ids must be a list of integers.")
        requested = [int(value) for value in concept_ids]
        if not requested or len(requested) != len(set(requested)):
            raise ValueError(
                "train.concept_ids must be a non-empty list of unique integers."
            )
        unsupported = [
            model_name
            for model_name in model_names
            if getattr(steerscope, model_name).inference_instance_scope
            != "per_concept"
        ]
        if unsupported:
            raise ValueError(
                "Non-prefix train.concept_ids currently requires per-concept "
                f"inference artifacts; unsupported methods: {unsupported}."
            )
        by_id = {int(concept_id): frame for concept_id, frame in frames}
        missing = sorted(set(requested).difference(by_id))
        if missing:
            raise ValueError(
                f"train.concept_ids contains IDs absent from generated data: {missing}."
            )
        return [(concept_id, by_id[concept_id]) for concept_id in requested]
    if num_concepts is not None:
        from steerscope.utils.concept_scope import select_concept_ids

        selected_ids = select_concept_ids(
            (concept_id for concept_id, _ in frames),
            count=int(num_concepts),
            seed=int(seed),
        )
        selected = set(selected_ids)
        return [
            (concept_id, frame)
            for concept_id, frame in frames
            if int(concept_id) in selected
        ]
    if max_concepts:
        return frames[:int(max_concepts)]
    return frames


def load_metadata(metadata_path):
    """
    Load metadata from a JSON lines file.
    """
    metadata = []
    with open(metadata_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            metadata += [data]  # Return the metadata as is
    return metadata


def prepare_concept_training_data(
    original_df, concept, tokenizer,
    binarize, train_on_negative, is_chat_model, output_length, model_name,
    max_num_of_examples=None, subset_seed=None, concept_id=None,
    use_dpo_loss=False, steering_prompt_type="prepend",
    keep_legacy_output_format=False, show_sample=True):

    # assign input and output containing concept with 1, otherwise 0
    positive_df = original_df[
        (original_df["output_concept"] == concept)
        & (original_df["category"] == "positive")
    ].copy()
    paired_negative_df = original_df[
        (original_df["output_concept"] == EMPTY_CONCEPT)
        & (original_df["category"] == "negative")
    ].copy()
    if positive_df.empty:
        raise ValueError(f"Training data for concept '{concept}' has no positive examples.")

    # PSR needs the unformatted question after normal input preparation.
    for frame in (positive_df, paired_negative_df):
        if "raw_input" not in frame.columns:
            frame["raw_input"] = frame["input"]
        if "raw_output" not in frame.columns:
            frame["raw_output"] = frame["output"]

    if use_dpo_loss:
        if binarize:
            raise ValueError("DPO training data cannot be binarized.")
        negative_df = paired_negative_df
    else:
        if paired_negative_df.empty:
            raise ValueError(
                f"Training data for concept '{concept}' requires paired negative "
                "examples generated from the same prompts."
            )
        required_pair_columns = {"pair_id", "input"}
        missing_columns = required_pair_columns - set(original_df.columns)
        if missing_columns:
            raise ValueError(
                f"Training data for concept '{concept}' is missing paired-data "
                f"columns: {sorted(missing_columns)}."
            )
        if (
            positive_df["pair_id"].duplicated().any()
            or paired_negative_df["pair_id"].duplicated().any()
        ):
            raise ValueError(
                f"Paired training data for concept '{concept}' must contain "
                "exactly one positive and one negative row per pair_id."
            )
        positive_pairs = dict(zip(positive_df["pair_id"], positive_df["input"]))
        negative_pairs = dict(zip(
            paired_negative_df["pair_id"], paired_negative_df["input"]
        ))
        if positive_pairs != negative_pairs:
            raise ValueError(
                f"Paired training data for concept '{concept}' must have identical "
                "pair_id and prompt values."
            )
        negative_df = paired_negative_df

    suffix_length, suffix_str = get_suffix_length(tokenizer)
    if show_sample:
        print(
            f"Suffix length for {model_name}: {suffix_length}, "
            f"Suffix string: {suffix_str}"
        )
    subset_group = concept if concept_id is None else concept_id
    if use_dpo_loss:
        positive_limit = (
            None
            if max_num_of_examples is None
            else int(max_num_of_examples) // 2
        )
        positive_df, _ = select_unpaired_subset(
            positive_df,
            max_rows=positive_limit,
            subset_seed=subset_seed,
            group_key=subset_group,
        )
    else:
        positive_df, negative_df, _ = select_balanced_subset(
            positive_df,
            negative_df,
            max_num_of_examples=max_num_of_examples,
            subset_seed=subset_seed,
            group_key=subset_group,
        )
    if binarize:
        if is_chat_model:
            system_messages = []
            if model_name in HAS_SYSTEM_PROMPT_MODELS:
                system_messages = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]

            def apply_chat_template(row):
                prompt_messages = system_messages + [
                    {"role": "user", "content": row["input"]}
                ]
                messages = prompt_messages + [
                    {"role": "assistant", "content": row["output"]}
                ]
                full_tokens = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False
                )[1:-suffix_length]
                prompt_tokens = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=True, add_generation_prompt=True
                )[1:]
                combined = tokenizer.decode(full_tokens)
                prompt = tokenizer.decode(prompt_tokens)
                # Both strings are re-tokenized below by the activation
                # dataloader, including its BOS token.
                assistant_start = len(tokenizer(
                    prompt, add_special_tokens=True
                )["input_ids"])
                return pd.Series({
                    "combined": combined,
                    "assistant_start": assistant_start,
                })

            positive_df = positive_df.copy()
            negative_df = negative_df.copy()
            positive_formatted = positive_df.apply(apply_chat_template, axis=1)
            negative_formatted = negative_df.apply(apply_chat_template, axis=1)
            positive_df[["combined", "assistant_start"]] = positive_formatted
            negative_df[["combined", "assistant_start"]] = negative_formatted
        else:
            positive_df = positive_df.copy()
            negative_df = negative_df.copy()
            positive_df['combined'] = positive_df['input'] + positive_df['output']
            negative_df['combined'] = negative_df['input'] + negative_df['output']
            positive_df["assistant_start"] = positive_df["input"].map(
                lambda value: len(tokenizer(value)["input_ids"])
            )
            negative_df["assistant_start"] = negative_df["input"].map(
                lambda value: len(tokenizer(value)["input_ids"])
            )
        retained_columns = ["combined", "assistant_start"]
        if "pair_id" in positive_df.columns and "pair_id" in negative_df.columns:
            retained_columns.append("pair_id")
        positive_df = pd.DataFrame(positive_df[retained_columns]).rename(columns={'combined': 'input'})
        negative_df = pd.DataFrame(negative_df[retained_columns]).rename(columns={'combined': 'input'})
        positive_df["labels"] = 1
        negative_df["labels"] = 0
        return pd.concat([positive_df, negative_df], axis=0)
    else:
        # Non-binarized data uses the chat template for standard instruction tuning.
        if not use_dpo_loss and train_on_negative:
            all_df = pd.concat([positive_df, negative_df], axis=0)
        else:
            # DPO uses positive examples only.
            all_df = positive_df
        if is_chat_model:
            system_messages = []
            if model_name in HAS_SYSTEM_PROMPT_MODELS:
                system_messages = [{"role": "system", "content": "You are a helpful assistant."}]

            def apply_chat_template(df, column_name):
                def template_function(row):
                    messages = system_messages + [{"role": "user", "content": row[column_name]}]
                    nobos = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)[1:]
                    return tokenizer.decode(nobos)
                df[column_name] = df.apply(template_function, axis=1)

            apply_chat_template(all_df, "input")
            if use_dpo_loss:
                if f"{steering_prompt_type}_steered_input" in all_df.columns:
                    apply_chat_template(all_df, f"{steering_prompt_type}_steered_input")

            # Add EOS prefix tokens by default. The truncation at data collator will take care of the rest.
            def apply_output_template(df, column_name):
                def template_function(row):
                    return row[column_name] + suffix_str
                df[column_name] = df.apply(template_function, axis=1)

            # The legacy format has much shorter outputs.
            if not keep_legacy_output_format:
                # Apply the template to all output columns
                for column in ["output", "winning_output", "losing_output", "prepend_steered_output", "blend_in_steered_output"]:
                    if column in all_df.columns:
                        apply_output_template(all_df, column)

            if show_sample:
                print("\n=== Sample Row Data ===")
                sample_row = all_df.iloc[0]
                for column in sample_row.index:
                    print(f"\n{column}:")
                    print("-" * (len(column) + 1))
                    print(f"{sample_row[column]}")
                print("=====================\n")

        return all_df # do nothing, the task will be standard instruction tuning.


def prepare_df(*args, **kwargs):
    """Backward-compatible alias for single-concept training preparation."""
    return prepare_concept_training_data(*args, **kwargs)


def prepare_all_concepts_training_data(
    original_df,
    selected_concepts,
    tokenizer,
    **kwargs,
):
    """Prepare selected concepts with the same rules used by per-concept runs."""
    prepared = []
    for concept_id, concept in selected_concepts:
        concept_rows = original_df[
            original_df["concept_id"] == int(concept_id)
        ].copy()
        if concept_rows.empty:
            raise ValueError(
                f"Training data is missing selected concept ID {concept_id}."
            )
        prepared.append(
            prepare_concept_training_data(
                concept_rows,
                concept,
                tokenizer,
                concept_id=int(concept_id),
                show_sample=False,
                **kwargs,
            )
        )
    if not prepared:
        raise ValueError("All-concepts training requires at least one concept.")
    return pd.concat(prepared, axis=0, ignore_index=True)


def prepare_training_data(
    original_df,
    tokenizer,
    *,
    training_granularity,
    concept=None,
    concept_id=None,
    selected_concepts=None,
    **kwargs,
):
    """Dispatch dataframe preparation according to a model's training scope."""
    if training_granularity == "per_concept":
        if concept is None or concept_id is None:
            raise ValueError(
                "Per-concept training requires concept and concept_id."
            )
        return prepare_concept_training_data(
            original_df,
            concept,
            tokenizer,
            concept_id=int(concept_id),
            **kwargs,
        )
    if training_granularity == "all_concepts":
        if selected_concepts is None:
            raise ValueError(
                "All-concepts training requires selected_concepts."
            )
        return prepare_all_concepts_training_data(
            original_df,
            selected_concepts,
            tokenizer,
            **kwargs,
        )
    raise ValueError(
        f"Unsupported training granularity: {training_granularity!r}."
    )


def partition_list(lst, n):
    """Split a list into approximately equal contiguous slices."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def _rank_training_assignment(
    df_list, world_size, rank, model_names, lm_model_name
):
    """Assign independent concepts per rank, except for methods whose distributed trainer requires identical rank order."""
    model_names = set(model_names)
    collective_sft = world_size > 1 and "SFT" in model_names
    if collective_sft and "gemma-2-9b" not in str(lm_model_name).lower():
        raise ValueError(
            "Multi-GPU SFT collective training is currently implemented only "
            f"for Gemma-2-9B, not {lm_model_name!r}."
        )
    if collective_sft and model_names != {"SFT"}:
        raise ValueError(
            "Multi-GPU SFT must use an SFT-only train YAML; it cannot share "
            f"a torchrun process group with {sorted(model_names - {'SFT'})}."
        )
    if collective_sft:
        return list(df_list), 0, rank == 0, True
    partitions = partition_list(df_list, world_size)
    return partitions[rank], rank, True, False


def load_state(dump_dir, rank):
    """
    Load the state from a file if it exists.
    """
    state_path = os.path.join(f"{dump_dir}", f"{STATE_FILE}_rank_{rank}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def _atomic_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False, dir=path.parent, suffix=".pkl.tmp"
    ) as temporary:
        temporary_path = Path(temporary.name)
        pickle.dump(value, temporary)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_state(dump_dir, state, rank):
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    _atomic_pickle(dump_dir / f"{STATE_FILE}_rank_{rank}", state)


def _checkpoint_dir(dump_dir, rank, concept_id, model_name):
    return (
        Path(dump_dir)
        / CHECKPOINTS_DIR
        / f"rank_{rank}"
        / f"concept_{int(concept_id)}"
        / model_name
    )


def _joint_checkpoint_dir(dump_dir, model_name):
    return Path(dump_dir) / CHECKPOINTS_DIR / "all_concepts" / model_name


def _partition_training_methods(model_names):
    per_concept = []
    all_concepts = []
    for model_name in sorted(model_names):
        model_class = getattr(steerscope, model_name)
        granularity = model_class.training_granularity
        if granularity == "per_concept":
            per_concept.append(model_name)
        elif granularity == "all_concepts":
            all_concepts.append(model_name)
        else:
            raise ValueError(
                f"Model '{model_name}' has unsupported training granularity "
                f"{granularity!r}."
            )
    return per_concept, all_concepts


def _checkpoint_key(path):
    try:
        rank = int(path.parents[1].name.removeprefix("rank_"))
        concept_id = int(path.parent.name.removeprefix("concept_"))
    except (ValueError, IndexError):
        return None
    return rank, concept_id, path.name


def _completed_training_methods(
    dump_dir,
    rank,
    model_names,
    method_fingerprints=None,
):
    completed = set()
    checkpoint_root = Path(dump_dir) / CHECKPOINTS_DIR / f"rank_{rank}"
    if checkpoint_root.exists():
        for marker in checkpoint_root.glob("concept_*/*/.complete.json"):
            key = _checkpoint_key(marker.parent)
            if key is not None and key[0] == rank:
                expected = (method_fingerprints or {}).get(key[2])
                if expected is not None:
                    with open(marker, encoding="utf-8") as file:
                        actual = json.load(file).get("fingerprint")
                    if actual != expected:
                        raise RuntimeError(
                            f"Checkpoint configuration changed for method "
                            f"'{key[2]}' concept {key[1]}. Use a new dump "
                            "directory instead of mixing training runs."
                        )
                completed.add((key[1], key[2]))
    return completed


def _completed_joint_training_methods(
    dump_dir,
    model_names,
    concept_ids,
    method_fingerprints=None,
):
    completed = set()
    expected_concepts = sorted(int(value) for value in concept_ids)
    for model_name in model_names:
        marker = _joint_checkpoint_dir(
            dump_dir, model_name
        ) / ".complete.json"
        if not marker.exists():
            continue
        with open(marker, encoding="utf-8") as file:
            manifest = json.load(file)
        actual_concepts = sorted(
            int(value) for value in manifest.get("concept_ids", [])
        )
        if actual_concepts != expected_concepts:
            raise RuntimeError(
                f"Checkpoint concept set changed for joint method "
                f"'{model_name}'. Use a new dump directory instead of mixing "
                "training runs."
            )
        expected_fingerprint = (method_fingerprints or {}).get(model_name)
        if (
            expected_fingerprint is not None
            and manifest.get("fingerprint") != expected_fingerprint
        ):
            raise RuntimeError(
                f"Checkpoint configuration changed for joint method "
                f"'{model_name}'. Use a new dump directory instead of mixing "
                "training runs."
            )
        completed.add(model_name)
    return completed


def _training_method_fingerprints(
    args,
    generate_args,
    data_dir,
    world_size,
    model_names,
    generated_data=None,
):
    if generated_data is None:
        generated_data = generated_data_identity(
            data_dir,
            use_dpo_loss=args.use_dpo_loss,
        )
    common = {
        "version": TRAIN_STATE_VERSION,
        "model_name": args.model_name,
        "layer": args.layer,
        "component": args.component,
        "output_length": args.output_length,
        "seed": args.seed,
        "training_recipe_version": TRAINING_RECIPE_VERSION,
        "seed_derivation": TRAINING_SEED_DERIVATION,
        "use_bf16": args.use_bf16,
        "max_concepts": args.max_concepts,
        "concept_ids": (
            None
            if getattr(args, "concept_ids", None) is None
            else [int(value) for value in args.concept_ids]
        ),
        "max_num_of_examples": args.max_num_of_examples,
        "subset_seed": getattr(args, "subset_seed", None),
        "use_dpo_loss": args.use_dpo_loss,
        "world_size": int(world_size),
        "generate_output_length": generate_args.output_length,
        "keep_legacy_output_format": generate_args.keep_legacy_output_format,
        "generated_data": generated_data,
    }
    # Treat an absent scope and num_concepts=None as the same full-pool recipe.
    if getattr(args, "num_concepts", None) is not None:
        common["num_concepts"] = int(args.num_concepts)
    fingerprints = {}
    for model_name in model_names:
        model_class = getattr(steerscope, model_name)
        if model_name in args.models.keys():
            method_args = vars(args.models[model_name]).copy()
        elif not getattr(model_class, "requires_training_args", True):
            method_args = {}
        else:
            raise KeyError(
                f"Training arguments are missing for method '{model_name}'."
            )
        scoped_args = set(
            getattr(model_class, "training_fingerprint_scoped_args", ())
        )
        unknown_scoped_args = scoped_args.difference(
            METHOD_SCOPED_FINGERPRINT_ARGS
        )
        if unknown_scoped_args:
            raise ValueError(
                f"Method '{model_name}' declares unknown scoped fingerprint "
                f"arguments: {sorted(unknown_scoped_args)}"
            )
        for argument in METHOD_SCOPED_FINGERPRINT_ARGS.difference(scoped_args):
            method_args.pop(argument, None)
        if not getattr(model_class, "uses_prompt_generation", False):
            method_args.pop("lm_model", None)
            method_args.pop("prompt_temperature", None)
        payload = {
            **common,
            "method": model_name,
            "method_args": method_args,
        }
        method_context = model_class.training_fingerprint_context()
        if method_context:
            payload["method_context"] = method_context
        encoded = json.dumps(
            payload, sort_keys=True, default=str, separators=(",", ":")
        ).encode("utf-8")
        fingerprints[model_name] = hashlib.sha256(encoded).hexdigest()
    return fingerprints


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_manifest(
    dump_dir,
    *,
    args,
    method_fingerprints,
    concept_ids,
    world_size,
    generated_data=None,
):
    """Write a small immutable identity for potentially huge artifacts."""
    concept_ids = sorted(int(value) for value in concept_ids)
    methods = {}
    for model_name in sorted(method_fingerprints):
        model_class = getattr(steerscope, model_name)
        methods[model_name] = {
            "fingerprint": method_fingerprints[model_name],
            "training_granularity": model_class.training_granularity,
            "inference_instance_scope": model_class.inference_instance_scope,
            "artifact_directory": model_class.artifact_directory,
        }
    payload = {
        "version": 2,
        "base_model": args.model_name,
        "layer": int(args.layer),
        "component": args.component,
        "world_size": int(world_size),
        "concept_ids": concept_ids,
        "methods": methods,
        "generated_data": generated_data,
    }
    _atomic_json(Path(dump_dir) / ARTIFACT_MANIFEST_FILE, payload)
    return payload


def _configured_evaluation_methods(config_path):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    evaluate = config.get("evaluate") or {}
    return set(evaluate.get("models") or ())


def _gemmascope_baseline_requires_refresh(
    dump_dir,
    generated_data,
    fingerprint=None,
):
    """Return whether a cached baseline predates or mismatches this pool."""
    manifest_path = Path(dump_dir) / ARTIFACT_MANIFEST_FILE
    try:
        with open(manifest_path, encoding="utf-8") as file:
            manifest = json.load(file)
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
        return True
    if manifest.get("version") != 2:
        return True
    if manifest.get("generated_data") != generated_data:
        return True
    method = (manifest.get("methods") or {}).get("GemmaScopeSAE")
    if not isinstance(method, dict) or not method.get("fingerprint"):
        return True
    return fingerprint is not None and method.get("fingerprint") != fingerprint


def _final_method_artifact_is_complete(dump_dir, model_name, concept_ids):
    dump_dir = Path(dump_dir)
    concept_ids = [int(concept_id) for concept_id in concept_ids]
    model_class = getattr(steerscope, model_name)
    artifact_directory = model_class.artifact_directory
    if model_class.requires_calibration_scale:
        scale_path = dump_dir / f"{model_name}_scale.pt"
        if not scale_path.exists():
            return False
        scales = torch.load(
            scale_path, map_location="cpu", weights_only=True
        ).reshape(-1)
        selected_scales = (
            scales[torch.as_tensor(concept_ids, dtype=torch.long)]
            if concept_ids and scales.numel() > max(concept_ids)
            else torch.empty(0)
        )
        if (
            scales.numel() <= max(concept_ids, default=-1)
            or selected_scales.numel() != len(concept_ids)
            or not torch.isfinite(selected_scales).all()
            or (selected_scales <= 0).any()
        ):
            return False
    if model_name == "GemmaScopeSAE":
        artifact_path = dump_dir / "GemmaScopeSAE.pt"
        if not artifact_path.exists():
            return False
        try:
            params = torch.load(
                artifact_path, map_location="cpu", weights_only=True
            )
            maximum_id = max(concept_ids, default=-1)
            return (
                isinstance(params, dict)
                and params["W_dec"].shape[0] > maximum_id
                and params["W_enc"].shape[1] > maximum_id
                and params["b_enc"].reshape(-1).numel() > maximum_id
                and params["threshold"].reshape(-1).numel() > maximum_id
            )
        except (KeyError, OSError, RuntimeError, TypeError, AttributeError):
            return False
    if model_class.training_granularity == "all_concepts":
        checkpoint = _joint_checkpoint_dir(dump_dir, model_name)
        return (
            (checkpoint / ".complete.json").exists()
            and artifact_directory is not None
            and (dump_dir / artifact_directory).exists()
        )
    if (
        model_class.inference_instance_scope == "per_concept"
        and artifact_directory is not None
    ):
        root = dump_dir / artifact_directory
        return all((root / str(concept_id)).is_dir() for concept_id in concept_ids)

    if model_name == "PromptSteering":
        path = dump_dir / "PromptSteering_prompts.json"
        if not path.exists():
            return False
        with open(path, encoding="utf-8") as file:
            entries = json.load(file)
        available = {int(entry["concept_id"]) for entry in entries}
        return set(concept_ids).issubset(available)

    weight_path = dump_dir / f"{model_name}_weight.pt"
    bias_path = dump_dir / f"{model_name}_bias.pt"
    if weight_path.exists() and bias_path.exists():
        weight = torch.load(weight_path, map_location="cpu", weights_only=True)
        first = next(iter(weight.values())) if isinstance(weight, dict) else weight
        return first.shape[0] > max(concept_ids, default=-1)

    top_features_path = dump_dir / f"{model_name}_top_features.json"
    if top_features_path.exists() and (dump_dir / f"{model_name}.pt").exists():
        with open(top_features_path, encoding="utf-8") as file:
            return len(json.load(file)) > max(concept_ids, default=-1)
    return False


def _save_method_checkpoint(
    benchmark_model, dump_dir, rank, concept_id, model_name, fingerprint=None
):
    final_dir = _checkpoint_dir(dump_dir, rank, concept_id, model_name)
    if (final_dir / ".complete.json").exists():
        with open(final_dir / ".complete.json", encoding="utf-8") as file:
            actual = json.load(file).get("fingerprint")
        if fingerprint is None or actual == fingerprint:
            return final_dir
        raise RuntimeError(
            f"Checkpoint configuration changed for method '{model_name}' "
            f"concept {concept_id}. Use a new dump directory."
        )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(dump_dir) / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(
        prefix=f"rank_{rank}_concept_{concept_id}_{model_name}_",
        dir=staging_root,
    ))
    try:
        benchmark_model.save(
            staging_dir,
            model_name=model_name,
            concept_id=concept_id,
        )
        with open(staging_dir / ".complete.json", "w", encoding="utf-8") as file:
            json.dump({
                "version": TRAIN_STATE_VERSION,
                "rank": int(rank),
                "concept_id": int(concept_id),
                "model_name": model_name,
                "fingerprint": fingerprint,
            }, file, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        if final_dir.exists():
            if (final_dir / ".complete.json").exists():
                return final_dir
            raise RuntimeError(f"Incomplete checkpoint already exists: {final_dir}")
        os.replace(staging_dir, final_dir)
        return final_dir
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _save_collective_sft_checkpoint(
    benchmark_model, dump_dir, concept_id, fingerprint=None
):
    """Collectively gather one SFT model and atomically publish it on rank 0."""
    rank = dist.get_rank()
    final_dir = _checkpoint_dir(dump_dir, 0, concept_id, "SFT")
    marker = final_dir / ".complete.json"
    if marker.exists():
        with open(marker, encoding="utf-8") as file:
            actual = json.load(file).get("fingerprint")
        if fingerprint is not None and actual != fingerprint:
            raise RuntimeError(
                f"Checkpoint configuration changed for method 'SFT' concept "
                f"{concept_id}. Use a new dump directory."
            )
        dist.barrier()
        return final_dir

    staging_path = None
    if rank == 0:
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(dump_dir) / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging_path = tempfile.mkdtemp(
            prefix=f"collective_concept_{concept_id}_SFT_",
            dir=staging_root,
        )
    shared_path = [staging_path]
    dist.broadcast_object_list(shared_path, src=0)
    staging_dir = Path(shared_path[0])
    try:
        # SFT.save delegates to Trainer.save_model; all FSDP ranks must enter.
        benchmark_model.save(staging_dir, model_name="SFT", concept_id=concept_id)
        dist.barrier()
        if rank == 0:
            with open(staging_dir / ".complete.json", "w", encoding="utf-8") as file:
                json.dump({
                    "version": TRAIN_STATE_VERSION,
                    "rank": 0,
                    "concept_id": int(concept_id),
                    "model_name": "SFT",
                    "fingerprint": fingerprint,
                    "distributed_training": "collective_fsdp_v1",
                }, file, sort_keys=True)
                file.flush()
                os.fsync(file.fileno())
            if final_dir.exists():
                if (final_dir / ".complete.json").exists():
                    shutil.rmtree(staging_dir)
                else:
                    raise RuntimeError(
                        f"Incomplete checkpoint already exists: {final_dir}"
                    )
            else:
                os.replace(staging_dir, final_dir)
        dist.barrier()
        return final_dir
    finally:
        if rank == 0 and staging_dir.exists():
            shutil.rmtree(staging_dir)


def _save_joint_method_checkpoint(
    benchmark_model,
    dump_dir,
    model_name,
    concept_ids,
    fingerprint=None,
):
    final_dir = _joint_checkpoint_dir(dump_dir, model_name)
    marker = final_dir / ".complete.json"
    if marker.exists():
        with open(marker, encoding="utf-8") as file:
            manifest = json.load(file)
        if fingerprint is None or manifest.get("fingerprint") == fingerprint:
            return final_dir
        raise RuntimeError(
            f"Checkpoint configuration changed for joint method '{model_name}'. "
            "Use a new dump directory."
        )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(dump_dir) / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(
        prefix=f"all_concepts_{model_name}_",
        dir=staging_root,
    ))
    try:
        benchmark_model.save(staging_dir, model_name=model_name)
        with open(staging_dir / ".complete.json", "w", encoding="utf-8") as file:
            json.dump({
                "version": TRAIN_STATE_VERSION,
                "training_granularity": "all_concepts",
                "concept_ids": sorted(int(value) for value in concept_ids),
                "model_name": model_name,
                "fingerprint": fingerprint,
            }, file, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        if final_dir.exists():
            if (final_dir / ".complete.json").exists():
                return final_dir
            raise RuntimeError(f"Incomplete checkpoint already exists: {final_dir}")
        os.replace(staging_dir, final_dir)
        return final_dir
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False, dir=path.parent, suffix=".pt.tmp"
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _any_rank_trained(local_training_changed, device):
    """Return whether this invocation trained a new checkpoint on any rank."""
    if not dist.is_available() or not dist.is_initialized():
        return bool(local_training_changed)
    collective_device = (
        device if dist.get_backend() == "nccl" else torch.device("cpu")
    )
    changed = torch.tensor(
        int(bool(local_training_changed)),
        dtype=torch.int32,
        device=collective_device,
    )
    dist.all_reduce(changed, op=dist.ReduceOp.MAX)
    return bool(changed.item())


def _new_rank_zero_artifact(dump_dir, rank, filename, force=False):
    """Only let rank zero create an artifact that is not already materialized."""
    if int(rank) == 0 and (
        force or not (Path(dump_dir) / filename).exists()
    ):
        return filename
    return None


def _concat_checkpoint_values(values):
    if isinstance(values[0], dict):
        return {
            key: torch.cat([value[key] for value in values], dim=0)
            for key in values[0]
        }
    return torch.cat(values, dim=0)


def _index_checkpoint_values_by_concept(
    values,
    concept_ids_by_value,
    *,
    dimension=0,
    fill_value=0,
):
    """Place compact checkpoint rows at their global concept IDs."""
    if not values or len(values) != len(concept_ids_by_value):
        raise ValueError(
            "Checkpoint values and concept-ID groups must be non-empty and "
            "have equal lengths."
        )

    flattened_ids = [
        int(concept_id)
        for group in concept_ids_by_value
        for concept_id in group
    ]
    if (
        not flattened_ids
        or min(flattened_ids) < 0
        or len(flattened_ids) != len(set(flattened_ids))
    ):
        raise ValueError(
            "Checkpoint concept IDs must be non-negative and unique."
        )

    if isinstance(values[0], dict):
        expected_keys = set(values[0])
        if any(set(value) != expected_keys for value in values):
            raise RuntimeError("Checkpoint dictionaries have different keys.")
        return {
            key: _index_checkpoint_values_by_concept(
                [value[key] for value in values],
                concept_ids_by_value,
                dimension=dimension,
                fill_value=fill_value,
            )
            for key in values[0]
        }

    if not all(torch.is_tensor(value) for value in values):
        raise TypeError("Checkpoint values must be tensors or tensor mappings.")
    compact = torch.cat(values, dim=dimension)
    if compact.shape[dimension] != len(flattened_ids):
        raise RuntimeError(
            "Checkpoint row count does not match its concept-ID mapping: "
            f"got {compact.shape[dimension]} rows for "
            f"{len(flattened_ids)} concepts."
        )

    indexed_shape = list(compact.shape)
    indexed_shape[dimension] = max(flattened_ids) + 1
    indexed = torch.full(
        indexed_shape,
        fill_value,
        dtype=compact.dtype,
        device=compact.device,
    )
    indices = torch.as_tensor(
        flattened_ids, dtype=torch.long, device=compact.device
    )
    indexed.index_copy_(dimension, indices, compact)
    return indexed


def _rank_artifacts_with_concept_ids(
    dump_dir,
    concept_ids_by_rank,
    filename_template,
):
    """Return complete per-rank artifact inputs and their row identities."""
    expected = [
        (
            Path(dump_dir) / filename_template.format(rank=rank),
            [int(value) for value in concept_ids],
        )
        for rank, concept_ids in enumerate(concept_ids_by_rank)
        if concept_ids
    ]
    existing = [
        (path, concept_ids)
        for path, concept_ids in expected
        if path.exists()
    ]
    if existing and len(existing) != len(expected):
        missing = [str(path) for path, _ in expected if not path.exists()]
        raise RuntimeError(
            "Per-rank training artifacts are incomplete; missing files: "
            f"{missing}."
        )
    return existing


def _link_checkpoint_directory(source, destination):
    source = Path(source).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        backup_root = destination.parent / ".pre_transactional"
        backup_root.mkdir(parents=True, exist_ok=True)
        backup = backup_root / destination.name
        counter = 1
        while backup.exists() or backup.is_symlink():
            backup = backup_root / f"{destination.name}.{counter}"
            counter += 1
        os.replace(destination, backup)
    temporary = destination.with_name(f".{destination.name}.link.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(
        os.path.relpath(source, destination.parent), target_is_directory=True
    )
    os.replace(temporary, destination)


def _atomic_copy(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False, dir=destination.parent, suffix=".copy.tmp"
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def materialize_joint_artifact(dump_dir, model_name):
    """Expose a completed joint checkpoint through the legacy train layout."""
    dump_dir = Path(dump_dir)
    checkpoint = _joint_checkpoint_dir(dump_dir, model_name)
    if not (checkpoint / ".complete.json").exists():
        return
    for source in checkpoint.iterdir():
        if source.name == ".complete.json":
            continue
        destination = dump_dir / source.name
        if source.is_dir():
            _link_checkpoint_directory(source, destination)
        else:
            _atomic_copy(source, destination)


def materialize_rank_artifacts(dump_dir, rank, concept_ids, model_names):
    """Rebuild append-style rank artifacts deterministically from checkpoints."""
    dump_dir = Path(dump_dir)
    for model_name in model_names:
        checkpoints = [
            _checkpoint_dir(dump_dir, rank, concept_id, model_name)
            for concept_id in concept_ids
        ]
        checkpoints = [
            path for path in checkpoints if (path / ".complete.json").exists()
        ]
        if not checkpoints:
            continue

        prompt_sources = [
            path / "PromptSteering_prompts.json" for path in checkpoints
        ]
        if all(source.exists() for source in prompt_sources):
            prompts = []
            for source in prompt_sources:
                with open(source, encoding="utf-8") as file:
                    prompts.append(json.load(file))
            _atomic_json(
                dump_dir / f"rank_{rank}_PromptSteering_prompts.json",
                prompts,
            )

        for suffix in ("weight.pt", "bias.pt", "scale.pt"):
            sources = [path / f"{model_name}_{suffix}" for path in checkpoints]
            if all(source.exists() for source in sources):
                values = [
                    torch.load(source, map_location="cpu", weights_only=True)
                    for source in sources
                ]
                _atomic_torch_save(
                    _concat_checkpoint_values(values),
                    dump_dir / f"rank_{rank}_{model_name}_{suffix}",
                )

        top_feature_sources = [
            path / f"{model_name}_top_features.json" for path in checkpoints
        ]
        if all(source.exists() for source in top_feature_sources):
            top_features = []
            for source in top_feature_sources:
                with open(source, encoding="utf-8") as file:
                    top_features.extend(json.load(file))
            with open(
                dump_dir / f"rank_{rank}_{model_name}_top_features.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(top_features, file)

        sae_sources = [path / f"{model_name}.pt" for path in checkpoints]
        if all(source.exists() for source in sae_sources):
            values = [
                torch.load(source, map_location="cpu", weights_only=True)
                for source in sae_sources
            ]
            merged = {"b_dec": values[0]["b_dec"]}
            for key in ("W_dec", "W_enc", "b_enc", "threshold"):
                dimension = 1 if key == "W_enc" else 0
                merged[key] = torch.cat(
                    [value[key] for value in values], dim=dimension
                )
            _atomic_torch_save(
                merged, dump_dir / f"rank_{rank}_{model_name}.pt"
            )

        for checkpoint in checkpoints:
            for artifact_root in (
                child for child in checkpoint.iterdir() if child.is_dir()
            ):
                for artifact in artifact_root.iterdir():
                    _link_checkpoint_directory(
                        artifact,
                        dump_dir / artifact_root.name / artifact.name,
                    )


def write_rank_metadata(dump_dir, rank, concept_ids, metadata):
    entries = [metadata[int(concept_id)] for concept_id in concept_ids]
    _atomic_jsonl(
        Path(dump_dir) / f"rank_{rank}_{METADATA_FILE}", entries
    )


def _atomic_jsonl(path, entries):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=path.parent, suffix=".jsonl.tmp", encoding="utf-8"
    ) as temporary:
        temporary_path = Path(temporary.name)
        for entry in entries:
            temporary.write(json.dumps(entry) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=path.parent, suffix=".json.tmp", encoding="utf-8"
    ) as temporary:
        json.dump(value, temporary, indent=2, sort_keys=True)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _model_construction_kwargs(
    args,
    model_name,
    model_instance,
    metadata_path,
    dump_dir,
    sae_params=None,
    concept_id=None,
):
    model_args = args.models[model_name]
    kwargs = {
        "mode": "train",
        "embed_dim": model_instance.config.hidden_size,
        "dtype": torch.bfloat16 if args.use_bf16 else None,
        "intervention_type": model_args.intervention_type,
        "sae_params": sae_params,
        "metadata_path": metadata_path,
        "dump_dir": dump_dir,
        "model_params": model_args,
        "dropout": model_args.dropout,
        "intervention_positions_dropout": (
            model_args.intervention_positions_dropout
        ),
        "preference_pairs": model_args.preference_pairs,
        "hypernet_initialize_from_pretrained": (
            model_args.hypernet_initialize_from_pretrained
        ),
    }
    optional = {
        "concept_id": concept_id,
        "low_rank_dimension": getattr(model_args, "low_rank_dimension", None),
        "num_hidden_layers": model_args.num_hidden_layers,
        "hypernet_name_or_path": model_args.hypernet_name_or_path,
    }
    kwargs.update({key: value for key, value in optional.items() if value is not None})
    return kwargs


def _data_preparation_kwargs(
    args,
    generate_args,
    model_name,
    is_chat_model,
):
    model_args = args.models[model_name]
    return {
        "binarize": model_args.binarize_dataset,
        "train_on_negative": model_args.train_on_negative,
        "use_dpo_loss": args.use_dpo_loss,
        "is_chat_model": is_chat_model,
        "output_length": int(args.output_length),
        "model_name": args.model_name,
        "max_num_of_examples": args.max_num_of_examples,
        "subset_seed": args.subset_seed,
        "steering_prompt_type": model_args.steering_prompt_type,
        "keep_legacy_output_format": generate_args.keep_legacy_output_format,
    }


def train_joint_methods(
    args,
    generate_args,
    model_instance,
    tokenizer,
    selected_df,
    selected_concepts,
    dump_dir,
    rank,
    device,
    world_size,
    model_names,
    method_fingerprints,
    prefix_length,
):
    if not model_names:
        return False
    concept_ids = [int(concept_id) for concept_id, _ in selected_concepts]
    completed = _completed_joint_training_methods(
        dump_dir,
        model_names,
        concept_ids,
        method_fingerprints=method_fingerprints,
    )
    metadata_path = os.path.join(args.data_dir, METADATA_FILE)
    is_chat_model = args.model_name in CHAT_MODELS
    local_training_changed = False

    for model_name in model_names:
        if model_name in completed:
            logger.warning("Skipping completed joint method %s.", model_name)
            if rank == 0:
                materialize_joint_artifact(dump_dir, model_name)
            dist.barrier()
            continue

        logger.warning(
            "Training joint method %s on %s concepts.",
            model_name,
            len(selected_concepts),
        )
        training_seed = derive_training_seed(args.seed, model_name)
        # DDP parameters are broadcast when wrapped, while rank-local random
        # streams (dropout, flow time, and sampling) must remain independent.
        set_seed(training_seed + rank)
        model_class = getattr(steerscope, model_name)
        benchmark_model = model_class(
            model_instance,
            tokenizer,
            layer=args.layer,
            training_args=args.models[model_name],
            lm_model_name=args.model_name,
            device=device,
            seed=training_seed,
            use_wandb=args.use_wandb,
            dump_dir=dump_dir,
        )
        benchmark_model.make_model(**_model_construction_kwargs(
            args,
            model_name,
            model_instance,
            metadata_path,
            dump_dir,
        ))
        prepared_df = prepare_training_data(
            selected_df.copy(),
            tokenizer,
            training_granularity=model_class.training_granularity,
            selected_concepts=selected_concepts,
            **_data_preparation_kwargs(
                args, generate_args, model_name, is_chat_model
            ),
        )
        benchmark_model.train(prepared_df, **{
            "prefix_length": prefix_length,
            "positions": args.models[model_name].intervention_positions,
            "exclude_bos": args.models[model_name].exclude_bos,
            "metadata_path": metadata_path,
            "world_size": world_size,
            "wandb_project": args.wandb_project,
            "wandb_name": args.wandb_name,
            "master_data_dir": generate_args.master_data_dir,
        })
        dist.barrier()
        if rank == 0:
            _save_joint_method_checkpoint(
                benchmark_model,
                dump_dir,
                model_name,
                concept_ids,
                fingerprint=method_fingerprints[model_name],
            )
            materialize_joint_artifact(dump_dir, model_name)
        dist.barrier()
        local_training_changed = True
        del benchmark_model
        torch.cuda.empty_cache()

    return local_training_changed


def main():

    args = TrainingArgs(section="train")
    generate_args = DatasetArgs(section="generate")

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://',
                          timeout=datetime.timedelta(seconds=6000))

    # Get the rank and world_size from environment variables
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Set the device for this process
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # Set a unique seed per rank for reproducibility
    set_seed(args.seed + rank)

    if args.overwrite_data_dir and Path(args.overwrite_data_dir).exists():
        logger.warning(f"Overwriting data directory {args.data_dir}")
        args.data_dir = args.overwrite_data_dir
    else:
        args.data_dir = f"{args.dump_dir}/generate"

    # Configure the logger per rank
    logger.setLevel(logging.WARNING)  # Set the logging level as desired

    # Create a logging formatter that includes the rank
    formatter = logging.Formatter(
        fmt=f'%(asctime)s,%(msecs)03d %(levelname)-8s [Rank {rank}] [%(filename)s:%(lineno)d] %(message)s',
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

    # Load dataset and metadata
    metadata_path = os.path.join(args.data_dir, 'metadata.jsonl')
    metadata = load_metadata(metadata_path)
    df_generator = data_generator(args.data_dir, use_dpo_loss=args.use_dpo_loss)
    df_list = list(df_generator)
    logger.warning(f"Total number of concept df loaded: {len(df_list)}")
    df_list = select_training_concepts(
        df_list,
        concept_ids=getattr(args, "concept_ids", None),
        max_concepts=args.max_concepts,
        num_concepts=getattr(args, "num_concepts", None),
        seed=args.seed,
        model_names=args.models.keys(),
    )
    if getattr(args, "concept_ids", None) is not None:
        logger.warning(
            "Selected explicit concept IDs: %s",
            [int(concept_id) for concept_id, _ in df_list],
        )
    elif getattr(args, "num_concepts", None) is not None:
        logger.warning(
            "Selected %s concepts with concept-scope seed %s: %s",
            args.num_concepts,
            args.seed,
            [int(concept_id) for concept_id, _ in df_list],
        )
    elif args.max_concepts:
        logger.warning(f"All ranks only processing {args.max_concepts} concepts")
    if not df_list:
        raise ValueError("Training requires at least one selected concept.")

    dump_dir = Path(args.dump_dir) / "train"
    dump_dir.mkdir(parents=True, exist_ok=True)

    evaluation_methods = _configured_evaluation_methods(args.config_file)
    generated_data = generated_data_identity(
        args.data_dir,
        use_dpo_loss=args.use_dpo_loss,
    )
    standalone_gemmascope = (
        "GemmaScopeSAE" in evaluation_methods
        and "GemmaScopeSAE" not in args.models
    )
    gemmascope_fingerprint = None
    if standalone_gemmascope:
        gemmascope_fingerprint = _training_method_fingerprints(
            args,
            generate_args,
            args.data_dir,
            world_size,
            ["GemmaScopeSAE"],
            generated_data=generated_data,
        )["GemmaScopeSAE"]
    refresh_gemmascope = bool(
        standalone_gemmascope
        and _gemmascope_baseline_requires_refresh(
            dump_dir,
            generated_data,
            fingerprint=gemmascope_fingerprint,
        )
    )
    if rank == 0 and refresh_gemmascope:
        # Both files are derived from concept metadata.  Keeping either one
        # across a changed pool can silently route the wrong SAE feature or
        # calibration scale to a concept ID.
        (dump_dir / "GemmaScopeSAE_scale.pt").unlink(missing_ok=True)

    # Preserve the original GemmaScope loading path. SAE feature selectors
    # need the full dictionary on every rank, while only rank 0 writes the
    # metadata-selected GemmaScopeSAE baseline.
    sae_params = None
    sae_selector_configured = any(
        model_name.startswith("GemmaScopeSAE")
        and model_name != "GemmaScopeSAE"
        for model_name in args.models.keys()
    )
    if sae_selector_configured:
        sae_params = save_pruned_sae(
            metadata_path,
            dump_dir,
            savefile=_new_rank_zero_artifact(
                dump_dir,
                rank,
                "GemmaScopeSAE.pt",
                force=refresh_gemmascope,
            ),
        )
    elif rank == 0 and (
        refresh_gemmascope or not (dump_dir / "GemmaScopeSAE.pt").exists()
    ):
        try:
            sae_params = save_pruned_sae(
                metadata_path,
                dump_dir,
                savefile=_new_rank_zero_artifact(
                    dump_dir,
                    rank,
                    "GemmaScopeSAE.pt",
                    force=refresh_gemmascope,
                ),
            )
        except Exception:
            sae_params = None

    if rank == 0 and "GemmaScopeSAE" in evaluation_methods:
        if not (dump_dir / "GemmaScopeSAE.pt").exists():
            raise RuntimeError(
                "GemmaScopeSAE evaluation is configured, but training did not "
                "produce GemmaScopeSAE.pt."
            )
        GemmaScopeSAE.prepare_calibration_artifact(
            metadata_path,
            dump_dir,
            [int(concept_id) for concept_id, _ in df_list],
            master_data_dir=generate_args.master_data_dir,
        )
    dist.barrier()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
    tokenizer.padding_side = "right"

    my_df_list, checkpoint_rank, owns_rank_artifacts, collective_sft = (
        _rank_training_assignment(
            df_list,
            world_size,
            rank,
            args.models.keys(),
            args.model_name,
        )
    )
    if collective_sft:
        logger.warning(
            "SFT collective mode: all %s ranks will train the same %s "
            "concepts in lockstep; rank 0 owns checkpoints.",
            world_size,
            len(my_df_list),
        )

    # Load model instance onto device
    if args.use_bf16:
        logger.warning(f"Using bfloat16 for model {args.model_name}")
    model_instance = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16 if args.use_bf16 else None)
    is_chat_model = True if args.model_name in CHAT_MODELS else False
    model_instance = model_instance.eval()
    model_instance.to(device)

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

    prefix_length = 1 # prefix is default to 1 for all models due to theBOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    state = load_state(dump_dir, checkpoint_rank)
    all_model_names = sorted(args.models.keys())
    manifest_model_names = sorted(
        set(all_model_names)
        | ({"GemmaScopeSAE"} if standalone_gemmascope else set())
    )
    per_concept_model_names, joint_model_names = _partition_training_methods(
        all_model_names
    )
    method_fingerprints = _training_method_fingerprints(
        args,
        generate_args,
        args.data_dir,
        world_size,
        manifest_model_names,
        generated_data=generated_data,
    )
    if state and state.get("version") != TRAIN_STATE_VERSION:
        raise RuntimeError(
            "This dump directory uses an unsupported training state format. "
            "Use a new dump directory or remove the old training artifacts."
        )
    completed = _completed_training_methods(
        dump_dir,
        checkpoint_rank,
        per_concept_model_names,
        method_fingerprints=method_fingerprints,
    )
    assigned_concept_ids = (
        [int(concept_id) for concept_id, _ in my_df_list]
        if owns_rank_artifacts
        else []
    )
    logger.warning(
        "Rank %s restored %s completed concept/method checkpoints.",
        rank,
        len(completed),
    )

    # Run training for assigned concept_ids
    # logger.warning(metadata)
    local_training_changed = False
    for concept_id, concept_df in my_df_list:
        concept_id = int(concept_id)
        logger.warning(f"Training models for concept_id {concept_id} on rank {rank}")
        for model_name in per_concept_model_names:
            if (concept_id, model_name) in completed:
                logger.warning(
                    "Rank %s skipping completed concept %s method %s.",
                    rank,
                    concept_id,
                    model_name,
                )
                continue

            concept = metadata[concept_id]["concept"]
            logger.warning(f"Training {model_name} with concept {concept}")
            training_seed = derive_training_seed(
                args.seed, model_name, concept_id
            )
            set_seed(training_seed)
            benchmark_model = getattr(steerscope, model_name)(
                model_instance, tokenizer, layer=args.layer,
                training_args=args.models[model_name],
                lm_model_name=args.model_name,
                device=device, seed=training_seed, use_wandb=args.use_wandb,
                dump_dir=dump_dir,
            )
            benchmark_model.make_model(**_model_construction_kwargs(
                args,
                model_name,
                model_instance,
                metadata_path,
                dump_dir,
                sae_params=sae_params,
                concept_id=concept_id,
            ))
            if (
                model_name not in {"LoReFT", "LoRA", "SFT", "PreferenceLoReFT", "ConceptLoReFT"}
                and args.use_bf16
                and hasattr(benchmark_model, "ax")
            ):
                if isinstance(benchmark_model.ax, list):
                    for ax in benchmark_model.ax:
                        ax.to(torch.bfloat16)
                else:
                    benchmark_model.ax.to(torch.bfloat16)
            kwargs = {
                "prefix_length": prefix_length,
                "positions": args.models[model_name].intervention_positions,
                "exclude_bos": args.models[model_name].exclude_bos,
                "metadata_path": metadata_path,
                "use_dpo_loss": args.use_dpo_loss,
                "logging_metadata": {
                    "concept_id": concept_id,
                    "model_name": model_name,
                    "layer": args.layer,
                },
                "wandb_project": args.wandb_project,
                "wandb_name": args.wandb_name,
                "negative_only": args.models[model_name].negative_only,
                "preference_pairs": args.models[model_name].preference_pairs,
                "steering_prompt_type": args.models[model_name].steering_prompt_type,
                "substraction_type": args.models[model_name].substraction_type,
                "concept": concept,
                "master_data_dir": generate_args.master_data_dir,
            }
            prepared_df = prepare_training_data(
                concept_df.copy(),
                tokenizer,
                training_granularity=(
                    getattr(steerscope, model_name).training_granularity
                ),
                concept=concept,
                concept_id=concept_id,
                **_data_preparation_kwargs(
                    args, generate_args, model_name, is_chat_model
                ),
            )
            benchmark_model.train(prepared_df, **kwargs)
            if (
                benchmark_model.requires_calibration_scale
                and not hasattr(benchmark_model, "calibration_scale")
            ):
                labels_are_binary = (
                    "labels" in prepared_df.columns
                    and set(prepared_df["labels"].dropna().unique()).issubset({0, 1})
                )
                if labels_are_binary:
                    calibration_df = prepared_df
                else:
                    calibration_df = prepare_df(
                        concept_df.copy(),
                        concept,
                        tokenizer,
                        binarize=True,
                        train_on_negative=True,
                        use_dpo_loss=False,
                        is_chat_model=is_chat_model,
                        output_length=int(args.output_length),
                        model_name=args.model_name,
                        max_num_of_examples=args.max_num_of_examples,
                        subset_seed=args.subset_seed,
                        concept_id=concept_id,
                        steering_prompt_type=args.models[model_name].steering_prompt_type,
                        keep_legacy_output_format=generate_args.keep_legacy_output_format,
                    )
                benchmark_model.calibrate(
                    calibration_df,
                    prefix_length=prefix_length,
                    batch_size=args.models[model_name].batch_size,
                    metadata_path=metadata_path,
                    concept_id=concept_id,
                    master_data_dir=generate_args.master_data_dir,
                )
            if collective_sft:
                _save_collective_sft_checkpoint(
                    benchmark_model,
                    dump_dir,
                    concept_id,
                    fingerprint=method_fingerprints[model_name],
                )
            else:
                _save_method_checkpoint(
                    benchmark_model,
                    dump_dir,
                    rank,
                    concept_id,
                    model_name,
                    fingerprint=method_fingerprints[model_name],
                )
            local_training_changed = True
            completed.add((concept_id, model_name))
            if owns_rank_artifacts:
                save_state(dump_dir, {
                    "version": TRAIN_STATE_VERSION,
                    "completed": sorted(completed),
                }, checkpoint_rank)
            if model_name == "SFT":
                # Reload the original model after SFT.
                del benchmark_model
                del model_instance
                gc.collect()
                torch.cuda.empty_cache()
                if args.use_bf16:
                    logger.warning(f"Using bfloat16 for model {args.model_name}")
                model_instance = AutoModelForCausalLM.from_pretrained(
                    args.model_name, torch_dtype=torch.bfloat16 if args.use_bf16 else None)
                is_chat_model = True if args.model_name in CHAT_MODELS else False
                model_instance = model_instance.eval()
                model_instance.to(device)
                benchmark_model = None
            if model_name == "LoRA":
                model_instance = benchmark_model.ax_model.unload()
            logger.warning(f"Saved weights and biases for model {model_name} on rank {rank}")
            # Clean up
            if benchmark_model is not None:
                del benchmark_model
            torch.cuda.empty_cache()

    materialize_rank_artifacts(
        dump_dir, rank, assigned_concept_ids, per_concept_model_names
    )
    write_rank_metadata(dump_dir, rank, assigned_concept_ids, metadata)

    dist.barrier()

    selected_concepts = [
        (int(concept_id), metadata[int(concept_id)]["concept"])
        for concept_id, _ in df_list
    ]
    selected_df = pd.concat(
        [concept_df.copy() for _, concept_df in df_list],
        axis=0,
        ignore_index=True,
    )
    joint_training_changed = train_joint_methods(
        args,
        generate_args,
        model_instance,
        tokenizer,
        selected_df,
        selected_concepts,
        dump_dir,
        rank,
        device,
        world_size,
        joint_model_names,
        method_fingerprints,
        prefix_length,
    )
    dist.barrier()
    any_rank_trained = _any_rank_trained(
        local_training_changed or joint_training_changed, device
    )

    # Rank 0 merges results
    if rank == 0:
        configured_models = manifest_model_names
        configured_concept_ids = [
            int(concept_id) for concept_id, _ in df_list
        ]
        concept_ids_by_rank = [
            [int(concept_id) for concept_id, _ in partition]
            for partition in partition_list(df_list, world_size)
        ]
        missing_final_artifacts = any(
            not _final_method_artifact_is_complete(
                dump_dir, model_name, configured_concept_ids
            )
            for model_name in configured_models
        )
        missing_shared_artifacts = any(
            not (dump_dir / filename).exists()
            for filename in (METADATA_FILE, CONFIG_FILE)
        )
        merge_required = (
            any_rank_trained
            or missing_final_artifacts
            or missing_shared_artifacts
        )
        if not merge_required:
            logger.warning(
                "All training checkpoints were cached; preserving existing "
                "merged artifacts."
            )
        else:
            logger.warning("Rank 0 is merging results.")

        if merge_required:
            # Merging metadata
            metadata_by_id = {}
            for r in range(world_size):
                metadata_path = os.path.join(dump_dir, f"rank_{r}_{METADATA_FILE}")
                with open(metadata_path, "r") as f:
                    for line in f:
                        metadata_entry = json.loads(line)
                        metadata_by_id[int(metadata_entry["concept_id"])] = metadata_entry
            _atomic_jsonl(
                dump_dir / METADATA_FILE,
                [metadata_by_id[concept_id] for concept_id in sorted(metadata_by_id)],
            )

            # Save other config
            config = {"model_name": args.model_name,
                    "layer": args.layer,
                    "component": args.component}
            config_path = dump_dir / CONFIG_FILE
            with open(config_path, 'w') as f:
                json.dump(config, f)

        for model_name in per_concept_model_names if merge_required else ():
            if model_name == "PromptSteering":
                prompts = []
                for r in range(world_size):
                    path = dump_dir / f"rank_{r}_PromptSteering_prompts.json"
                    if not path.exists():
                        continue
                    with open(path, encoding="utf-8") as file:
                        prompts.extend(json.load(file))
                prompts.sort(key=lambda entry: int(entry["concept_id"]))
                expected = set(configured_concept_ids)
                actual = {int(entry["concept_id"]) for entry in prompts}
                if actual != expected:
                    raise RuntimeError(
                        "PromptSteering prompt artifacts are incomplete: "
                        f"expected concepts {sorted(expected)}, got {sorted(actual)}."
                    )
                _atomic_json(
                    dump_dir / "PromptSteering_prompts.json",
                    prompts,
                )
                continue

            # Merge pruned SAEs while preserving their global concept IDs.
            sae_inputs = _rank_artifacts_with_concept_ids(
                dump_dir,
                concept_ids_by_rank,
                f"rank_{{rank}}_{model_name}.pt",
            )
            if not sae_inputs:
                logger.warning(f"No SAE files found for model {model_name}. Skipping.")
            else:
                sae_weights = [
                    torch.load(path, map_location="cpu", weights_only=True)
                    for path, _ in sae_inputs
                ]
                sae_concept_ids = [concept_ids for _, concept_ids in sae_inputs]
                combined_sae_params = {
                    "b_dec": sae_weights[0]["b_dec"],
                    "W_dec": _index_checkpoint_values_by_concept(
                        [value["W_dec"] for value in sae_weights],
                        sae_concept_ids,
                    ),
                    "W_enc": _index_checkpoint_values_by_concept(
                        [value["W_enc"] for value in sae_weights],
                        sae_concept_ids,
                        dimension=1,
                    ),
                    "b_enc": _index_checkpoint_values_by_concept(
                        [value["b_enc"] for value in sae_weights],
                        sae_concept_ids,
                    ),
                    "threshold": _index_checkpoint_values_by_concept(
                        [value["threshold"] for value in sae_weights],
                        sae_concept_ids,
                    ),
                }
                _atomic_torch_save(
                    combined_sae_params, dump_dir / f"{model_name}.pt"
                )
                logger.warning(f"Saved merged SAE weights for model {model_name}")

            # Merge top features into the same global concept-ID layout.
            top_feature_inputs = _rank_artifacts_with_concept_ids(
                dump_dir,
                concept_ids_by_rank,
                f"rank_{{rank}}_{model_name}_top_features.json",
            )
            if not top_feature_inputs:
                logger.warning(f"No top features files found for model {model_name}. Skipping.")
            else:
                combined_top_features = [0] * (max(configured_concept_ids) + 1)
                for top_feature_file, rank_concept_ids in top_feature_inputs:
                    with open(top_feature_file, "r") as f:
                        rank_top_features = json.load(f)
                    if len(rank_top_features) != len(rank_concept_ids):
                        raise RuntimeError(
                            f"{top_feature_file} contains "
                            f"{len(rank_top_features)} features for "
                            f"{len(rank_concept_ids)} concepts."
                        )
                    for concept_id, top_feature in zip(
                        rank_concept_ids, rank_top_features
                    ):
                        combined_top_features[concept_id] = top_feature
                with open(dump_dir / f"{model_name}_top_features.json", "w") as f:
                    json.dump(combined_top_features, f)
                logger.warning(f"Saved merged top features for model {model_name}")

            weight_inputs = _rank_artifacts_with_concept_ids(
                dump_dir,
                concept_ids_by_rank,
                f"rank_{{rank}}_{model_name}_weight.pt",
            )
            bias_inputs = _rank_artifacts_with_concept_ids(
                dump_dir,
                concept_ids_by_rank,
                f"rank_{{rank}}_{model_name}_bias.pt",
            )
            scale_inputs = _rank_artifacts_with_concept_ids(
                dump_dir,
                concept_ids_by_rank,
                f"rank_{{rank}}_{model_name}_scale.pt",
            )

            if scale_inputs:
                merged_scale = _index_checkpoint_values_by_concept(
                    [
                        torch.load(path, map_location="cpu", weights_only=True)
                        .reshape(-1)
                        for path, _ in scale_inputs
                    ],
                    [concept_ids for _, concept_ids in scale_inputs],
                    fill_value=1,
                )
                _atomic_torch_save(
                    merged_scale, dump_dir / f"{model_name}_scale.pt"
                )

            if not weight_inputs or not bias_inputs:
                logger.warning(
                    f"No weight or bias files found for model {model_name}. "
                    "Skipping."
                )
                continue

            # Load weights and biases
            weights = [
                torch.load(path, map_location="cpu", weights_only=True)
                for path, _ in weight_inputs
            ]
            biases = [
                torch.load(path, map_location="cpu", weights_only=True)
                for path, _ in bias_inputs
            ]

            merged_weight = _index_checkpoint_values_by_concept(
                weights,
                [concept_ids for _, concept_ids in weight_inputs],
            )
            merged_bias = _index_checkpoint_values_by_concept(
                biases,
                [concept_ids for _, concept_ids in bias_inputs],
            )

            # Save merged weight and bias files
            weight_file = dump_dir / f"{model_name}_weight.pt"
            bias_file = dump_dir / f"{model_name}_bias.pt"
            _atomic_torch_save(merged_weight, weight_file)
            _atomic_torch_save(merged_bias, bias_file)
            logger.warning(f"Saved merged weights and biases for model {model_name}")

            # Optionally delete per-rank files
            for f, _ in weight_inputs + bias_inputs + scale_inputs:
                try:
                    f.unlink()
                    logger.warning(f"Deleted file {f.name}")
                except Exception as e:
                    logger.error(f"Error deleting file {f.name}: {e}")

        incomplete = [
            model_name
            for model_name in configured_models
            if not _final_method_artifact_is_complete(
                dump_dir, model_name, configured_concept_ids
            )
        ]
        if incomplete:
            raise RuntimeError(
                "Training finished without complete materialized artifacts for "
                f"methods: {incomplete}."
            )
        write_artifact_manifest(
            dump_dir,
            args=args,
            method_fingerprints=method_fingerprints,
            concept_ids=configured_concept_ids,
            world_size=world_size,
            generated_data=generated_data,
        )

    # Finalize the process group
    dist.destroy_process_group()

    # Remove handlers to prevent duplication if the script is run multiple times
    logger.removeHandler(console_handler)
    # If file_handler is used, remove it as well
    # logger.removeHandler(file_handler)


if __name__ == "__main__":
    main()
