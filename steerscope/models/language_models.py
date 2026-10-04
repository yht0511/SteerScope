from ..utils.constants import (
    UNIT_1M,
    PRICING_DOLLAR_PER_1M_TOKEN,
)

import httpx, asyncio
import hashlib
import os, uuid, string, json
import random
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from .language_model_cache import SQLiteLanguageModelCache

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

def is_first_char_punctuation(s):
    if s and s[0] in string.punctuation:
        return True
    return False


class LanguageModelStats(object):
    """Main class for recording language model usage"""

    def __init__(self, model, retain_details=True):
        self.model = model
        self.retain_details = bool(retain_details)
        
        _uuid = str(uuid.uuid4())
        self.key = f"{model}-{_uuid}"
        self.completion_tokens = {}
        self.prompt_tokens = {}
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.prompt_cache = {}
        self.total_call = 0
        self.total_cache_hit = 0

    def record(self, api_name, stats, prompt=None, completion=None):
        self.total_call += 1
        if stats is None:
             self.total_cache_hit += 1
             return
        completion_tokens = int(stats.get(
            "completion_tokens", stats.get("output_tokens", 0)
        ))
        self.total_completion_tokens += completion_tokens
        prompt_tokens = int(stats.get(
            "prompt_tokens", stats.get("input_tokens", 0)
        ))
        self.total_prompt_tokens += prompt_tokens
        if self.retain_details:
            self.completion_tokens.setdefault(api_name, []).append(
                completion_tokens
            )
            self.prompt_tokens.setdefault(api_name, []).append(prompt_tokens)
            self.prompt_cache.setdefault(api_name, [])
        logger.debug(
            f"calling {api_name}, input tokens {prompt_tokens}, "
            f"output tokens {completion_tokens}")
        if self.retain_details and prompt is not None:
            self.prompt_cache[api_name].append({
                "prompt": prompt,
                "completion": completion
            })
    
    def get_total_tokens(self, breakdown=True):
        if breakdown:
            return self.total_prompt_tokens, self.total_completion_tokens
        return self.total_prompt_tokens + self.total_completion_tokens

    def reset(self):
        self.completion_tokens = {}
        self.prompt_tokens = {}
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.prompt_cache = {}
        self.total_call = 0
        self.total_cache_hit = 0

    def get_total_price(self):
        if self.model not in PRICING_DOLLAR_PER_1M_TOKEN:
            return None
        input_tokens, output_tokens = self.get_total_tokens()
        input_price = (input_tokens/UNIT_1M)*\
            PRICING_DOLLAR_PER_1M_TOKEN[self.model]["input"]
        output_price = (output_tokens/UNIT_1M)*\
            PRICING_DOLLAR_PER_1M_TOKEN[self.model]["output"]
        return input_price + output_price
    
    def print_report(self):
        logger.warning("="*20)
        logger.warning(f"Total calls: {self.total_call}, Total cache hits: {self.total_cache_hit}")
        price = self.get_total_price()
        logger.warning("Total price: %s", "unknown" if price is None else f"${price}")
        logger.warning("="*20)

    def get_report(self):
        input_tokens, output_tokens = self.get_total_tokens()
        return {
            "total_calls": self.total_call,
            "network_calls": self.total_call - self.total_cache_hit,
            "total_cache_hits": self.total_cache_hit,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "total_price": self.get_total_price()
        }

class LanguageModel(object):
    """Main class abstract async remote language model access"""

    def __init__(self, model, client, dump_dir=None, use_cache=True, cache_level="api", **kwargs):
        self.model = model
        self.stats = LanguageModelStats(
            model,
            retain_details=kwargs.get("retain_stats_details", True),
        )
        self.client = client
        endpoint = getattr(client, "base_url", "")
        self.cache_endpoint = str(endpoint).rstrip("/") if isinstance(endpoint, (str, httpx.URL)) else ""
        # dump dir
        if dump_dir:
            cur_save_dir = Path(dump_dir) / "lm_cache"
            cur_save_dir.mkdir(parents=True, exist_ok=True)
            self.dump_dir = cur_save_dir
        self.temperature = kwargs.get("temperature", 1.0)
        self.api_batch_size = int(kwargs.get("api_batch_size", 32))
        self.api_max_attempts = int(kwargs.get("api_max_attempts", 8))
        self.api_retry_initial_delay = float(
            kwargs.get("api_retry_initial_delay", 1.0)
        )
        self.api_retry_max_delay = float(kwargs.get("api_retry_max_delay", 30.0))
        if self.api_max_attempts < 1:
            raise ValueError("api_max_attempts must be at least 1")
        if self.api_batch_size < 1:
            raise ValueError("api_batch_size must be at least 1")
        self.cache_dir = None
        self.use_cache = use_cache
        self.cache_level = cache_level
        self.cache = None
        self.api_count = {}
        if self.use_cache:
            assert kwargs.get("master_data_dir", None), "master_data_dir is required for cache"
            legacy_dir = Path(kwargs["master_data_dir"]) / "persist_lm_cache"
            if kwargs.get("cache_tag", None):
                cache_stem = f"{self.model}_{kwargs['cache_tag']}_cache"
            else:
                cache_stem = f"{self.model}_cache"
            # The old pickle remains a read-only migration source.  Retain the
            # compatibility attribute because some callers inspect its path.
            self.cache_file = legacy_dir / f"{cache_stem}.pkl"
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            configured_cache_dir = (
                kwargs.get("lm_cache_dir")
                or os.environ.get("STEERSCOPE_LM_CACHE_DIR")
            )
            if configured_cache_dir:
                self.cache_dir = Path(configured_cache_dir).expanduser()
            else:
                # SQLite WAL must be local.  Namespace the local fallback by
                # data root so unrelated datasets and tests cannot collide.
                master_root = str(
                    Path(kwargs["master_data_dir"]).expanduser().resolve()
                )
                namespace = hashlib.sha256(
                    master_root.encode("utf-8")
                ).hexdigest()[:16]
                cache_root = os.environ.get("STEERSCOPE_CACHE_DIR")
                if cache_root:
                    cache_root = Path(cache_root).expanduser()
                elif os.environ.get("XDG_CACHE_HOME"):
                    cache_root = (
                        Path(os.environ["XDG_CACHE_HOME"]).expanduser()
                        / "steerscope"
                    )
                else:
                    cache_root = Path.home() / ".cache" / "steerscope"
                self.cache_dir = cache_root / "lm_cache" / namespace
            self.cache_db_file = self.cache_dir / f"{cache_stem}.sqlite3"
            self.cache_db_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache = SQLiteLanguageModelCache(
                self.cache_db_file,
                legacy_pickle_path=self.cache_file,
            )

    def normalize(self, text):
        return "" if text is None else text.strip()

    def _get_cache_key(self, prompt, api_count, api_name):
        # Versioned keys do not accept legacy entries with unknown sampling settings.
        return json.dumps({
            "version": 2,
            "endpoint": self.cache_endpoint,
            "model": self.model,
            "temperature": self.temperature,
            "prompt": prompt,
            "purpose": api_name if self.cache_level != "prompt" else None,
        }, sort_keys=True, ensure_ascii=True)
    
    async def chat_completion(
        self, client, prompt, api_name, refresh_cache=False
    ):
        cache_key = self._allocate_cache_key(prompt, api_name)
        if self.use_cache and not refresh_cache:
            cached = self.cache.get_many([cache_key])
            if cache_key in cached:
                return cached[cache_key], None
        completion, usage = await self._network_chat_completion(
            client, prompt, api_name
        )
        if self.use_cache:
            self.cache.upsert_many({cache_key: completion})
        return completion, usage

    def _allocate_cache_key(self, prompt, api_name):
        api_count = self.api_count.get(api_name, 0)
        self.api_count[api_name] = api_count + 1
        return self._get_cache_key(prompt, api_count, api_name)

    async def _network_chat_completion(self, client, prompt, api_name):
        for attempt in range(1, self.api_max_attempts + 1):
            caught_error = None
            try:
                raw_completion = await client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=self.temperature,
                )
                break
            except (APITimeoutError, APIConnectionError, RateLimitError) as error:
                caught_error = error
                retryable = True
            except APIStatusError as error:
                caught_error = error
                retryable = error.status_code in {408, 409, 429} or error.status_code >= 500
            if not retryable or attempt == self.api_max_attempts:
                raise caught_error
            base_delay = min(
                self.api_retry_max_delay,
                self.api_retry_initial_delay * (2 ** (attempt - 1)),
            )
            delay = base_delay * random.SystemRandom().uniform(0.75, 1.25)
            logger.warning(
                "Transient API error for %s (%s), retrying attempt %d/%d in %.1fs.",
                api_name,
                type(caught_error).__name__,
                attempt + 1,
                self.api_max_attempts,
                delay,
            )
            await asyncio.sleep(delay)
        raw_completion = raw_completion.to_dict()
        completion = self.normalize(raw_completion["choices"][0]["message"]["content"])

        usage = raw_completion.get("usage")
        if not usage:
            # Some OpenAI-compatible gateways omit usage metadata even though
            # choices contain a valid completion. Keep the judge result and
            # record zero unknown tokens instead of aborting the evaluation.
            logger.warning(
                "Language-model response for %s omitted usage metadata; "
                "token cost for this call will be recorded as zero.",
                api_name,
            )
            usage = {}
        return (completion, usage)
        
    async def chat_completions(
        self,
        api_names,
        prompts,
        batch_size=None,
        refresh_cache=False,
        progress_callback=None,
    ):
        """handling batched async calls with internal batching mechanism"""
        batch_size = self.api_batch_size if batch_size is None else int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        # Ensure api_names is a list of appropriate length
        if not isinstance(api_names, list):
            api_names = [api_names] * len(prompts)

        # Process in batches
        all_completions = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_api_names = api_names[i:i + batch_size]
            cache_keys = [
                self._allocate_cache_key(prompt, api_name)
                for prompt, api_name in zip(batch_prompts, batch_api_names)
            ]
            cached = (
                {}
                if not self.use_cache or refresh_cache
                else self.cache.get_many(cache_keys)
            )
            raw_completions = [None] * len(batch_prompts)
            misses = []
            for index, (prompt, api_name, cache_key) in enumerate(zip(
                batch_prompts, batch_api_names, cache_keys
            )):
                if cache_key in cached:
                    raw_completions[index] = (cached[cache_key], None)
                else:
                    misses.append((index, prompt, api_name, cache_key))
            network_results = await asyncio.gather(*[
                self._network_chat_completion(self.client, prompt, api_name)
                for _, prompt, api_name, _ in misses
            ])
            writes = {}
            for (index, _, _, cache_key), response in zip(
                misses, network_results
            ):
                raw_completions[index] = response
                writes[cache_key] = response[0]
            if self.use_cache and writes:
                self.cache.upsert_many(writes)
            # post handling for current batch
            for j, (completion, usage) in enumerate(raw_completions):
                all_completions.append(completion)
                self.stats.record(
                    batch_api_names[j], usage,
                    prompt=batch_prompts[j], completion=completion)

            if progress_callback is not None:
                progress_callback(self.stats.get_report())

            # Persist each completed API batch so an interruption loses at most
            # the currently in-flight requests.
            self.save_cache()

        return all_completions

    def dump(self):
        with open(self.dump_dir / "tmp_prompt_cache.json", "w") as outfile:
            json.dump(self.stats.prompt_cache, outfile, indent=4)
        
        with open(self.dump_dir / "cost.jsonl", 'a') as f:
            f.write(json.dumps({"price": self.stats.get_total_price()}) + '\n')

    def save_cache(self):
        if self.cache is not None:
            self.cache.flush()

    async def close(self):
        """Flush/close the cache and then close the HTTP client."""
        if self.cache is not None:
            self.cache.close()
            self.cache = None
        await self.client.close()
