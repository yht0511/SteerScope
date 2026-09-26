"""Compatibility helpers between the released FLAS core and Transformers 4.45.1."""

from __future__ import annotations

import torch
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2DecoderLayer,
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
)

RMSNorm = Gemma2RMSNorm


def text_config(config):
    config = getattr(config, "text_config", None) or config
    if getattr(config, "model_type", None) != "gemma2":
        raise ValueError(
            "The native SteerScope FLAS integration currently supports only "
            f"Gemma-2; got model_type={getattr(config, 'model_type', None)!r}."
        )
    return config


def text_decoder(model):
    base = model.model
    if hasattr(base, "language_model"):
        return base.language_model
    return base


def rotary_embedding(config):
    return Gemma2RotaryEmbedding(
        config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    )


def flow_self_attention(config, target_layer: int):
    copied = type(config).from_dict(config.to_dict())
    copied._attn_implementation = "eager"
    # Construction parity retains Gemma-2's sliding/global selection. Every
    # FlowBlock owns a dedicated DynamicCache, so its cache index must be zero.
    attention = Gemma2Attention(copied, layer_idx=int(target_layer) % 2)
    attention.layer_idx = 0
    return attention


def attention_forward(
    attention,
    hidden_states,
    attention_mask,
    position_ids,
    past_key_value=None,
    use_cache=False,
):
    cache_position = None
    if position_ids is not None:
        cache_position = position_ids[0]
    return attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=False,
        use_cache=use_cache,
        cache_position=cache_position,
    )


def decoder_layer_forward(layer, hidden_states, attention_mask, position_ids):
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    return layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=cache_position,
    )


__all__ = [
    "Gemma2DecoderLayer",
    "RMSNorm",
    "attention_forward",
    "decoder_layer_forward",
    "flow_self_attention",
    "rotary_embedding",
    "text_config",
    "text_decoder",
]
