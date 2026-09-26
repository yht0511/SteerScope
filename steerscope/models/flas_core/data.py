"""FLAS-specific data preparation copied from the standalone trainer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FLASDataset(Dataset):
    def __init__(self, dataframe):
        required = {"input", "output", "output_concept", "concept_id"}
        missing = required.difference(dataframe.columns)
        if missing:
            raise KeyError(f"FLAS training data is missing columns: {sorted(missing)}")
        self.samples = [
            {
                "input": str(row["input"]),
                "output": str(row["output"]),
                "concept": str(row["output_concept"]),
                "concept_id": int(row["concept_id"]),
            }
            for _, row in dataframe.iterrows()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_flas_batch(
    batch,
    *,
    tokenizer,
    max_length=256,
    concept_max_length=64,
):
    """Tokenize prompt/output separately and mask prompt labels as in FLAS."""
    full_ids = []
    prompt_lengths = []
    concepts = []
    concept_ids = []
    for sample in batch:
        concept = sample["concept"]
        if not concept.strip():
            continue
        prompt_ids = tokenizer(sample["input"], add_special_tokens=True)["input_ids"]
        output_ids = tokenizer(sample["output"], add_special_tokens=False)["input_ids"]
        combined = (list(prompt_ids) + list(output_ids))[: int(max_length)]
        prompt_length = min(len(prompt_ids), len(combined))
        if prompt_length <= 0 or prompt_length >= len(combined):
            continue
        full_ids.append(combined)
        prompt_lengths.append(prompt_length)
        concepts.append(concept)
        concept_ids.append(sample["concept_id"])
    if not full_ids:
        raise RuntimeError(
            "FLAS discarded the entire batch because no output tokens remained."
        )

    encoded = tokenizer.pad({"input_ids": full_ids}, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.clone()
    prompt_mask = torch.zeros_like(attention_mask)
    for index, prompt_length in enumerate(prompt_lengths):
        labels[index, :prompt_length] = -100
        prompt_mask[index, :prompt_length] = 1
    labels[attention_mask == 0] = -100
    prompt_mask = prompt_mask * attention_mask

    concept_encoded = tokenizer(
        concepts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(concept_max_length),
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_mask": prompt_mask,
        "concept_input_ids": concept_encoded["input_ids"],
        "concept_attention_mask": concept_encoded["attention_mask"],
        "concept_ids": torch.tensor(concept_ids, dtype=torch.long),
    }


@dataclass
class FLASCollator:
    tokenizer: object
    max_length: int = 256
    concept_max_length: int = 64

    def __call__(self, batch):
        return collate_flas_batch(
            batch,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            concept_max_length=self.concept_max_length,
        )


def compute_diversity_loss(velocity, concept_ids, attention_mask=None):
    """Penalize cosine similarity between velocities for different concepts."""
    if velocity is None:
        device = concept_ids.device if torch.is_tensor(concept_ids) else "cpu"
        return torch.tensor(0.0, device=device)
    if attention_mask is not None:
        mask = attention_mask.float().unsqueeze(-1)
        pooled = (velocity * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        pooled = velocity.mean(dim=1)
    if pooled.shape[0] < 2:
        return velocity.new_tensor(0.0)
    normalized = F.normalize(pooled, dim=-1)
    similarity = torch.matmul(normalized, normalized.t())
    different = (concept_ids.unsqueeze(0) != concept_ids.unsqueeze(1)).to(
        similarity.dtype
    )
    if different.sum() == 0:
        return velocity.new_tensor(0.0)
    return (similarity * different).sum() / different.sum()


__all__ = [
    "FLASCollator",
    "FLASDataset",
    "collate_flas_batch",
    "compute_diversity_loss",
]
