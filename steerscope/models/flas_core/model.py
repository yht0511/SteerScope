"""FLAS time-conditioned velocity field, adapted for SteerScope's Gemma-2 stack.

The module follows ``src/flas/model.py`` from the FLAS repository.  The model
structure and parameter names are preserved; only Transformers-version calls
are routed through :mod:`compatibility`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache

from .compatibility import (
    RMSNorm,
    attention_forward,
    decoder_layer_forward,
    flow_self_attention,
    rotary_embedding,
    text_config,
    text_decoder,
)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first = value[..., : value.shape[-1] // 2]
    second = value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    batch, num_kv_heads, sequence_length, head_dim = hidden_states.shape
    if repetitions == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, repetitions, sequence_length, head_dim
    )
    return hidden_states.reshape(
        batch, num_kv_heads * repetitions, sequence_length, head_dim
    )


def _apply_rope_single(value, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (value * cos) + (rotate_half(value) * sin)


class FlowCrossAttention(nn.Module):
    """GQA cross-attention from activation queries to encoded concept tokens."""

    def __init__(
        self,
        *,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        rms_norm_eps,
        rotary_emb,
        attn_bias=False,
        softcap=None,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.softcap = softcap

        self.q_proj = nn.Linear(
            hidden_size, self.num_heads * self.head_dim, bias=attn_bias
        )
        self.k_proj = nn.Linear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, hidden_size, bias=attn_bias
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.rotary_emb = rotary_emb

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask=None,
        q_pos_offset=0,
        q_position_ids=None,
        kv_position_ids=None,
    ):
        batch_size, query_length, _ = hidden_states.shape
        key_length = encoder_hidden_states.shape[1]

        query = self.q_proj(hidden_states).view(
            batch_size, query_length, self.num_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_proj(encoder_hidden_states).view(
            batch_size, key_length, self.num_kv_heads, self.head_dim
        )
        key = self.k_norm(key).transpose(1, 2)
        value = self.v_proj(encoder_hidden_states).view(
            batch_size, key_length, self.num_kv_heads, self.head_dim
        )
        value = value.transpose(1, 2)

        if q_position_ids is None:
            q_position_ids = (
                torch.arange(query_length, device=query.device).unsqueeze(0)
                + q_pos_offset
            )
        if kv_position_ids is None:
            kv_position_ids = torch.arange(key_length, device=key.device).unsqueeze(0)
        q_cos, q_sin = self.rotary_emb(query, q_position_ids)
        k_cos, k_sin = self.rotary_emb(key, kv_position_ids)
        query = _apply_rope_single(query, q_cos, q_sin)
        key = _apply_rope_single(key, k_cos, k_sin)

        key = repeat_kv(key, self.num_kv_groups)
        value = repeat_kv(value, self.num_kv_groups)
        attention_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        if self.softcap is not None:
            attention_weights = (
                torch.tanh(attention_weights / self.softcap) * self.softcap
            )
        if encoder_attention_mask is not None:
            mask = encoder_attention_mask[:, None, None, :]
            attention_weights = attention_weights.masked_fill(mask == 0, -1e4)
        attention_weights = F.softmax(
            attention_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        output = torch.matmul(attention_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, query_length, -1)
        return self.o_proj(output)


def sinusoidal_time_embedding(time: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(0, half, dtype=torch.float32, device=time.device)
        / half
    )
    arguments = time[:, None].float() * frequencies[None]
    embedding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=-1)
    if dimension % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimeEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_dimension: int = 128):
        super().__init__()
        self.freq_dim = int(frequency_dimension)
        self.mlp = nn.Sequential(
            nn.Linear(self.freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, time):
        embedding = sinusoidal_time_embedding(time, self.freq_dim)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class FlowBlock(nn.Module):
    """Time embedding, concept cross-attention, causal self-attention and MLP."""

    def __init__(self, config, rotary_emb, init_gate=0.1, layer_idx=0):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        epsilon = config.rms_norm_eps
        head_dim = getattr(
            config, "head_dim", hidden_size // config.num_attention_heads
        )
        attention_bias = getattr(config, "attention_bias", False)
        softcap = getattr(config, "attn_logit_softcapping", None)

        self.pre_cross_norm = RMSNorm(hidden_size, eps=epsilon)
        self.cross_attn = FlowCrossAttention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=epsilon,
            rotary_emb=rotary_emb,
            attn_bias=attention_bias,
            softcap=softcap,
        )
        self.post_cross_norm = RMSNorm(hidden_size, eps=epsilon)
        self.cross_gate = nn.Parameter(torch.full((hidden_size,), init_gate))

        self.pre_sa_norm = RMSNorm(hidden_size, eps=epsilon)
        self.post_sa_norm = RMSNorm(hidden_size, eps=epsilon)
        self.self_attn = flow_self_attention(config, layer_idx)
        self.self_attn_gate = nn.Parameter(torch.full((hidden_size,), init_gate))

        self.pre_mlp_norm = RMSNorm(hidden_size, eps=epsilon)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.post_mlp_norm = RMSNorm(hidden_size, eps=epsilon)
        activation = getattr(config, "hidden_act", None) or getattr(
            config, "hidden_activation", None
        )
        self.act_fn = ACT2FN[activation]
        self.mlp_gate = nn.Parameter(torch.full((hidden_size,), init_gate))

    def forward(
        self,
        hidden_states,
        concept_hidden,
        concept_mask=None,
        t_emb=None,
        self_attn_cache=None,
        padding_mask=None,
        use_cache=False,
        q_pos_offset=0,
        activation_position_ids=None,
    ):
        if t_emb is not None:
            hidden_states = hidden_states + t_emb[:, None, :]

        normalized = self.pre_cross_norm(hidden_states)
        cross_delta = self.cross_attn(
            normalized,
            concept_hidden,
            concept_mask,
            q_pos_offset=q_pos_offset,
            q_position_ids=activation_position_ids,
        )
        cross_delta = self.post_cross_norm(cross_delta)
        hidden_states = hidden_states + self.cross_gate * cross_delta

        normalized = self.pre_sa_norm(hidden_states)
        query_length = hidden_states.shape[1]
        past_length = (
            self_attn_cache.get_seq_length() if self_attn_cache is not None else 0
        )
        key_length = past_length + query_length
        mask_value = -1e4
        rows = torch.arange(query_length, device=hidden_states.device).unsqueeze(1)
        columns = torch.arange(key_length, device=hidden_states.device).unsqueeze(0)
        causal_mask = (
            torch.where(
                columns <= rows + past_length,
                torch.zeros(1, device=hidden_states.device, dtype=hidden_states.dtype),
                torch.full(
                    (1,),
                    mask_value,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        if padding_mask is not None:
            visible = padding_mask[:, :key_length]
            padding = (
                1.0 - visible[:, None, None, :].to(hidden_states.dtype)
            ) * mask_value
            causal_mask = causal_mask + padding

        attention_output = attention_forward(
            self.self_attn,
            normalized,
            causal_mask,
            activation_position_ids,
            past_key_value=self_attn_cache,
            use_cache=use_cache,
        )
        self_delta = self.post_sa_norm(attention_output[0])
        hidden_states = hidden_states + self.self_attn_gate * self_delta
        new_cache = self_attn_cache if use_cache else None

        value = self.pre_mlp_norm(hidden_states)
        value = self.down_proj(self.act_fn(self.gate_proj(value)) * self.up_proj(value))
        value = self.post_mlp_norm(value)
        hidden_states = hidden_states + self.mlp_gate * value
        return hidden_states, new_cache


class FlowFunction(nn.Module):
    """Velocity field ``v_theta(h, t, concept) = blocks(h, t, c) - h``."""

    def __init__(self, config, num_blocks=1, time_conditioned=True, layer_idx=0):
        super().__init__()
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.num_blocks = int(num_blocks)
        self.time_conditioned = bool(time_conditioned)
        if self.time_conditioned:
            self.time_embed = TimeEmbedder(config.hidden_size)
        self.rotary_emb = rotary_embedding(config)
        self.blocks = nn.ModuleList(
            [
                FlowBlock(config, self.rotary_emb, layer_idx=layer_idx)
                for _ in range(self.num_blocks)
            ]
        )

    def forward(
        self,
        hidden_states,
        concept_hidden,
        concept_mask=None,
        t=None,
        self_attn_caches=None,
        use_cache=False,
        padding_mask=None,
        past_len=0,
        position_ids=None,
    ):
        original = hidden_states
        time_embedding = (
            self.time_embed(t) if self.time_conditioned and t is not None else None
        )
        sequence_length = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(
                past_len,
                past_len + sequence_length,
                device=hidden_states.device,
            ).unsqueeze(0)

        new_caches = [] if use_cache else None
        for index, block in enumerate(self.blocks):
            if self_attn_caches is not None:
                cache = self_attn_caches[index]
            elif use_cache:
                cache = DynamicCache()
            else:
                cache = None
            hidden_states, cache = block(
                hidden_states,
                concept_hidden,
                concept_mask,
                t_emb=time_embedding,
                self_attn_cache=cache,
                padding_mask=padding_mask,
                use_cache=use_cache,
                q_pos_offset=past_len,
                activation_position_ids=position_ids,
            )
            if use_cache:
                new_caches.append(cache)
        return hidden_states - original, new_caches


class ConceptEncoder(nn.Module):
    """Frozen first two Gemma-2 layers used to encode natural-language concepts."""

    def __init__(self, base_model, num_layers=2):
        super().__init__()
        config = text_config(base_model.config)
        decoder = text_decoder(base_model)
        if num_layers < 1 or num_layers > len(decoder.layers):
            raise ValueError("FLAS concept encoder has an invalid layer count.")
        self.embed_tokens = decoder.embed_tokens
        self.layers = nn.ModuleList(list(decoder.layers[:num_layers]))
        self.norm = decoder.norm
        self.hidden_size = int(config.hidden_size)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_base_model(cls, base_model, num_layers=2):
        return cls(base_model, num_layers=num_layers)

    def forward(self, input_ids, attention_mask=None):
        batch_size, sequence_length = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        normalizer = torch.tensor(
            self.hidden_size**0.5,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        hidden_states = hidden_states * normalizer
        position_ids = torch.arange(
            sequence_length, device=hidden_states.device
        ).unsqueeze(0)
        minimum = torch.finfo(hidden_states.dtype).min
        causal = (
            torch.triu(
                torch.full(
                    (sequence_length, sequence_length),
                    minimum,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        if attention_mask is not None:
            padding = (
                1.0 - attention_mask[:, None, None, :].to(hidden_states.dtype)
            ) * minimum
            attention = (causal + padding).clamp(min=minimum)
        else:
            attention = causal.expand(batch_size, -1, -1, -1)
        for layer in self.layers:
            output = decoder_layer_forward(
                layer, hidden_states, attention, position_ids
            )
            hidden_states = output[0] if isinstance(output, tuple) else output
        return self.norm(hidden_states)


def build_flow_model_from_base(
    base_model,
    layer=20,
    num_blocks=1,
    time_conditioned=True,
    init_from_base=True,
):
    """Build FLAS from the base model already owned by the SteerScope runtime."""
    config = text_config(base_model.config)
    decoder = text_decoder(base_model)
    if layer < 0 or layer >= len(decoder.layers):
        raise ValueError(
            f"Invalid FLAS layer {layer}; model has {len(decoder.layers)} layers."
        )
    flow_function = FlowFunction(
        config,
        num_blocks=num_blocks,
        time_conditioned=time_conditioned,
        layer_idx=layer,
    )
    if init_from_base:
        source = decoder.layers[layer]
        norm_mapping = {
            "pre_sa_norm": "input_layernorm",
            "post_sa_norm": "post_attention_layernorm",
            "pre_mlp_norm": "pre_feedforward_layernorm",
            "post_mlp_norm": "post_feedforward_layernorm",
        }
        for block in flow_function.blocks:
            block.gate_proj.load_state_dict(source.mlp.gate_proj.state_dict())
            block.up_proj.load_state_dict(source.mlp.up_proj.state_dict())
            block.down_proj.load_state_dict(source.mlp.down_proj.state_dict())
            block.self_attn.load_state_dict(source.self_attn.state_dict())
            for target_name, source_name in norm_mapping.items():
                getattr(block, target_name).load_state_dict(
                    getattr(source, source_name).state_dict()
                )
    concept_encoder = ConceptEncoder.from_base_model(base_model, num_layers=2)
    return flow_function, concept_encoder


def integrate_euler(
    flow_function,
    hidden_states,
    concept_hidden,
    concept_mask,
    flow_times,
    n_steps,
    *,
    self_attn_caches=None,
    padding_mask=None,
    use_cache=False,
    past_len=0,
    position_ids=None,
):
    """Apply the exact fixed-step Euler integration used by FLAS."""
    batch_size = hidden_states.shape[0]
    # The released generator divides fp32 flow times before casting the Euler
    # step size to the FlowFunction dtype. The ordering matters for bf16.
    flow_times = torch.as_tensor(
        flow_times, device=hidden_states.device, dtype=torch.float32
    ).reshape(-1)
    if flow_times.numel() == 1:
        flow_times = flow_times.expand(batch_size)
    if flow_times.numel() != batch_size:
        raise ValueError("FLAS flow times must contain one value per example.")
    step_size = (flow_times / int(n_steps)).to(hidden_states.dtype)
    caches = [None] * int(n_steps) if self_attn_caches is None else self_attn_caches
    last_velocity = None
    for step in range(int(n_steps)):
        time = step_size * step
        velocity, new_cache = flow_function(
            hidden_states,
            concept_hidden,
            concept_mask,
            t=time,
            self_attn_caches=caches[step],
            padding_mask=padding_mask,
            use_cache=use_cache,
            past_len=past_len,
            position_ids=position_ids,
        )
        if use_cache:
            caches[step] = new_cache
        hidden_states = hidden_states + step_size[:, None, None] * velocity
        last_velocity = velocity
    return hidden_states, last_velocity, caches if use_cache else None


__all__ = [
    "ConceptEncoder",
    "FlowBlock",
    "FlowCrossAttention",
    "FlowFunction",
    "TimeEmbedder",
    "build_flow_model_from_base",
    "integrate_euler",
]
