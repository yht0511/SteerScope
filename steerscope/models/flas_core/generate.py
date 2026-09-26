"""Cached, token-by-token FLAS generation adapted from the released generator."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from transformers.cache_utils import HybridCache

from .compatibility import text_decoder
from .model import integrate_euler


class FLASGenerator:
    """Run the base LM and FLAS self-attention caches in lockstep."""

    def __init__(
        self,
        model,
        tokenizer,
        flow_function,
        concept_encoder,
        layer,
        *,
        n_steps=3,
        concept_max_length=64,
        input_max_length=512,
        device=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.flow_function = flow_function
        self.concept_encoder = concept_encoder
        self.layer = int(layer)
        self.n_steps = int(n_steps)
        self.concept_max_length = int(concept_max_length)
        self.input_max_length = int(input_max_length)
        self.device = torch.device(device or next(model.parameters()).device)
        self.flow_dtype = next(flow_function.parameters()).dtype
        self._hook_handle = None
        self._active = False
        self._flow_times = None
        self._concept_hidden = None
        self._concept_mask = None
        self._padding_mask = None
        self._self_attention_caches = None
        self._is_prefill = True
        self._past_length = 0
        self._position_ids = None

    @torch.no_grad()
    def encode_concepts(self, texts):
        previous_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            encoded = self.tokenizer(
                [str(text) for text in texts],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.concept_max_length,
            ).to(self.device)
        finally:
            self.tokenizer.padding_side = previous_padding
        hidden = self.concept_encoder(encoded["input_ids"], encoded["attention_mask"])
        return hidden.to(self.flow_dtype), encoded["attention_mask"].float()

    def _hook(self, _module, _inputs, output):
        if not self._active:
            return output
        if torch.count_nonzero(self._flow_times).item() == 0:
            return output
        is_tuple = isinstance(output, tuple)
        original = output[0] if is_tuple else output
        hidden = original.to(self.flow_dtype)
        hidden, _, caches = integrate_euler(
            self.flow_function,
            hidden,
            self._concept_hidden[: hidden.shape[0]],
            self._concept_mask[: hidden.shape[0]],
            self._flow_times[: hidden.shape[0]],
            self.n_steps,
            self_attn_caches=self._self_attention_caches,
            padding_mask=self._padding_mask[: hidden.shape[0]],
            use_cache=True,
            past_len=0 if self._is_prefill else self._past_length,
            position_ids=self._position_ids[: hidden.shape[0]],
        )
        self._self_attention_caches = caches
        hidden = hidden.to(original.dtype)
        return (hidden, *output[1:]) if is_tuple else hidden

    @contextmanager
    def _installed_hook(self):
        if self._hook_handle is not None:
            raise RuntimeError("FLAS generation hook is already installed.")
        layer = text_decoder(self.model).layers[self.layer]
        self._hook_handle = layer.register_forward_hook(self._hook)
        self._active = True
        try:
            yield
        finally:
            self._active = False
            self._hook_handle.remove()
            self._hook_handle = None
            self._self_attention_caches = None

    @staticmethod
    def _eos_mask(tokens, eos_token_id):
        token_values = tokens.squeeze(1)
        if isinstance(eos_token_id, (list, tuple)):
            result = torch.zeros_like(token_values, dtype=torch.bool)
            for value in eos_token_id:
                result |= token_values == int(value)
            return result
        return token_values == int(eos_token_id)

    @torch.no_grad()
    def generate_batch(
        self,
        prompts,
        concept_texts,
        flow_times,
        *,
        max_new_tokens=128,
        temperature=1.0,
        do_sample=True,
    ):
        if not (len(prompts) == len(concept_texts) == len(flow_times)):
            raise ValueError("FLAS generation inputs must have matching lengths.")
        if not prompts:
            return []
        concept_hidden, concept_mask = self.encode_concepts(concept_texts)
        previous_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                [str(prompt) for prompt in prompts],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.input_max_length,
                add_special_tokens=True,
            ).to(self.device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            batch_size, prompt_width = input_ids.shape
            self._concept_hidden = concept_hidden
            self._concept_mask = concept_mask
            self._flow_times = torch.as_tensor(
                flow_times, device=self.device, dtype=torch.float32
            )
            self._padding_mask = attention_mask.float()
            self._self_attention_caches = [None] * self.n_steps
            self._is_prefill = True
            self._past_length = 0
            self._position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

            maximum_length = prompt_width + int(max_new_tokens)
            cache_dtype = text_decoder(self.model).embed_tokens.weight.dtype
            base_cache = HybridCache(
                self.model.config,
                batch_size=batch_size,
                max_cache_len=maximum_length,
                device=self.device,
                dtype=cache_dtype,
            )
            generated = input_ids
            unfinished = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id

            with self._installed_hook():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=self._position_ids,
                    past_key_values=base_cache,
                    cache_position=torch.arange(prompt_width, device=self.device),
                    use_cache=True,
                    return_dict=True,
                )
                next_logits = output.logits[:, -1, :]
                self._is_prefill = False
                self._past_length = prompt_width

                for _ in range(int(max_new_tokens)):
                    if do_sample:
                        if float(temperature) <= 0:
                            raise ValueError(
                                "FLAS sampling requires a positive temperature."
                            )
                        probabilities = torch.softmax(
                            next_logits / float(temperature), dim=-1
                        )
                        next_token = torch.multinomial(probabilities, 1)
                    else:
                        next_token = next_logits.argmax(dim=-1, keepdim=True)
                    next_token = next_token.masked_fill(
                        ~unfinished.unsqueeze(1), int(pad_token_id)
                    )
                    generated = torch.cat([generated, next_token], dim=1)
                    attention_mask = torch.cat(
                        [attention_mask, unfinished.unsqueeze(1).long()], dim=1
                    )
                    eos_hit = self._eos_mask(next_token, self.tokenizer.eos_token_id)
                    unfinished = unfinished & ~eos_hit
                    if not unfinished.any():
                        break

                    self._padding_mask = attention_mask.float()
                    self._position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)[
                        :, -1:
                    ]
                    output = self.model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        position_ids=self._position_ids,
                        past_key_values=base_cache,
                        cache_position=torch.tensor(
                            [self._past_length], device=self.device
                        ),
                        use_cache=True,
                        return_dict=True,
                    )
                    next_logits = output.logits[:, -1, :]
                    self._past_length += 1
            return self.tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
        finally:
            self.tokenizer.padding_side = previous_padding
            self._active = False
            self._self_attention_caches = None


__all__ = ["FLASGenerator"]
