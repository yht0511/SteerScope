from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple, TypeVar, Union
import wandb
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    Gemma2Config,
    Gemma2PreTrainedModel,
    AutoTokenizer
)

from transformers.activations import FastGELUActivation

from transformers.modeling_utils import PreTrainedModel

from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2Config,
    Gemma2RMSNorm,
    Gemma2DecoderLayer,
    Gemma2RotaryEmbedding,
    Gemma2MLP,
    _prepare_4d_causal_attention_mask_with_cache_position,
)

from transformers.utils import logging

from transformers.cache_utils import Cache, HybridCache
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithCrossAttentions

from .configuration_hypernet import HypernetConfig
from .layers import HypernetDecoderLayer

from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import seaborn as sns
import time
from dataclasses import dataclass

logger = logging.get_logger(__name__)

# T = TypeVar("T", bound="LlamaInterpretor")


class HypernetPreTrainedModel(PreTrainedModel):
    config_class = HypernetConfig
    _keys_to_ignore_on_load_missing = ['target_model']
    _keys_to_ignore_on_save = ['target_model']
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Gemma2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = False
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def _check_and_enable_sdpa(cls, config, hard_check_only: bool = False):
        """
        Overloads `PreTrainedModel._check_and_enable_sdpa` so as to DISABLE torch SDPA by default on Gemma2 models.
        SDPA reduces the model performance on Gemma2 because of the logits softcapping.
        """
        config = super()._check_and_enable_sdpa(config, hard_check_only=hard_check_only)

        # if using the default path -> swap sdpa by eager
        if not hard_check_only and config._attn_implementation == "sdpa":
            config._attn_implementation = "eager"

        return config
                


class UnembeddingMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, output_size, normalize=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.output_size = output_size
        
        self.normalize = normalize
        
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.output_size, bias=True)
        self.act_fn = FastGELUActivation()
        
        self.norm = nn.LayerNorm(self.output_size, eps=1e-5) if self.normalize else None

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        if self.norm is not None:
            down_proj = self.norm(down_proj)
            
        return down_proj


class HypernetModel(HypernetPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Gemma2DecoderLayer`]

    Args:
        config: Gemma2Config
    """

    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HypernetDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.regression_head = nn.Linear(config.hidden_size, config.hidden_size)
        
        
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        base_encoder_hidden_states: Optional[torch.FloatTensor] = None,
        base_encoder_attention_mask: Optional[torch.Tensor] = None,
        base_encoder_position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            batch_size, seq_len, _ = inputs_embeds.shape
            past_key_values = HybridCache(
                self.config,
                batch_size=batch_size,
                max_cache_len=seq_len,
                device=self.device,
                dtype=inputs_embeds.dtype,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            
        if base_encoder_position_ids is None:
            base_encoder_position_ids = torch.arange(
                base_encoder_hidden_states.shape[1], device=base_encoder_hidden_states.device
            ).unsqueeze(0)
            
            
        base_encoder_attention_mask = self._update_encoder_attention_mask(
            base_encoder_hidden_states, base_encoder_attention_mask
        ) if base_encoder_attention_mask is not None else None
        
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds
    
        # normalized
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    base_encoder_hidden_states,
                    base_encoder_attention_mask,
                    base_encoder_position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    base_encoder_hidden_states=base_encoder_hidden_states,
                    base_encoder_attention_mask=base_encoder_attention_mask,
                    base_encoder_position_ids=base_encoder_position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        
        hidden_states = hidden_states[:, -1, :]
        hidden_states = self.regression_head(hidden_states)    

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        
    def _update_encoder_attention_mask(
        self,
        base_encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ):
        dtype, device = base_encoder_hidden_states.dtype, base_encoder_hidden_states.device
        min_dtype = torch.finfo(dtype).min
        # Assign min_dtype to attention_mask = 0 and 0 to attention_mask = 1
        encoder_attention_mask = encoder_attention_mask.to(dtype)
        encoder_attention_mask = torch.where(
            encoder_attention_mask == 0, torch.tensor(min_dtype, dtype=dtype, device=device), 0
        )
        return encoder_attention_mask
        
        
        

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: HybridCache,
        output_attentions: bool,
    ):
        # Flash Attention currently doesn't support static cache but Gemma2 work only with static cache.
        # So we will pass in attention mask as is in any case, not only when ther's padding. Then we'll use its shape
        # to cut out keys/values trailing 0 used in static cache. This workaround should be compile compatible
        # as it doesn't cause dynamic control issues.
        if self.config._attn_implementation == "flash_attention_2":
            return attention_mask

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_length()
        else:
            target_length = attention_mask.shape[-1] if attention_mask is not None else input_tensor.shape[1]

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        return causal_mask

