import json
import os
from pathlib import Path

from steerscope.utils.constants import CHAT_MODELS
from steerscope.utils.model_utils import get_suffix_length


CONFIG_FILE = "config.json"
METADATA_FILE = "metadata.jsonl"


def load_config(config_path):
    config_file = Path(config_path) / CONFIG_FILE
    if not config_file.exists():
        return None
    with open(config_file) as f:
        return json.load(f)


def load_metadata_flatten(metadata_path):
    metadata = []
    with open(Path(metadata_path) / METADATA_FILE, "r") as f:
        for line in f:
            data = json.loads(line)
            concept = data["concept"]
            concept_genres_map = data["concept_genres_map"][concept]
            metadata.append({
                "concept": concept,
                "ref": data["ref"],
                "concept_genres_map": {concept: concept_genres_map},
                "concept_id": data["concept_id"],
            })
    return metadata


def prepare_df(current_df, tokenizer, is_chat_model, model_name):
    suffix_length, _ = get_suffix_length(tokenizer)
    if is_chat_model:
        if model_name == "meta-llama/Llama-3.1-8B-Instruct":
            def apply_chat_template(row):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]},
                ]
                tokens = tokenizer.apply_chat_template(messages, tokenize=True)[1:-suffix_length]
                return tokenizer.decode(tokens)
        else:
            def apply_chat_template(row):
                messages = [
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]},
                ]
                tokens = tokenizer.apply_chat_template(messages, tokenize=True)[1:-suffix_length]
                return tokenizer.decode(tokens)
        current_df["input"] = current_df.apply(apply_chat_template, axis=1)
    return current_df


def is_chat_model(model_name):
    return model_name in CHAT_MODELS
