"""Language-model judge capability for evaluators that explicitly need it."""

import asyncio
import logging
import time
import json
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from steerscope.models.language_models import LanguageModel
from steerscope.utils.api_clients import openai_client_credentials


logger = logging.getLogger(__name__)

_COUNT_REPORT_FIELDS = (
    "total_calls",
    "network_calls",
    "total_cache_hits",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)
_EMPTY_REPORT = {
    **{field: 0 for field in _COUNT_REPORT_FIELDS},
    "total_price": 0.0,
}


class JudgeEvaluatorMixin:
    """Own the client, cache, retries, and accounting for an LM judge."""

    def __init__(self, node, context, **params):
        super().__init__(node, context, **params)
        self._judge_client = None
        self._judge_model = None
        self._evaluation_judge_report = dict(_EMPTY_REPORT)
        self._target_judge_report = dict(_EMPTY_REPORT)
        self._judge_progress_last_update = 0.0
        self._judge_fallback_count = 0
        self._target_fallback_start = 0
        self.judge_concurrency = int(self.params.get("judge_concurrency", 32))
        if self.judge_concurrency < 1:
            raise ValueError("judge_concurrency must be at least 1.")
        self.judge_progress_interval = float(
            self.params.get("judge_progress_interval", 1.0)
        )
        if self.judge_progress_interval < 0:
            raise ValueError("judge_progress_interval cannot be negative.")
        self.judge_parse_retries = int(
            self.params.get("judge_parse_retries", 3)
        )
        if self.judge_parse_retries < 0:
            raise ValueError("judge_parse_retries cannot be negative.")

    @classmethod
    def execution_context(cls, node, args):
        return {
            **super().execution_context(node, args),
            "judge_model": getattr(args, "lm_model", None),
        }

    def open_resources(self, models) -> None:
        super().open_resources(models)
        if not self.args.lm_model:
            raise ValueError(
                f"Evaluator '{self.node_id}' requires evaluate.lm_model."
            )
        self._judge_client = AsyncOpenAI(
            **openai_client_credentials("judge"),
            timeout=60.0,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100,
                    max_connections=1000,
                ),
                headers={"Connection": "close"},
            ),
            # LanguageModel owns retries so attempts are logged and not multiplied.
            max_retries=0,
        )
        self._judge_model = LanguageModel(
            self.args.lm_model,
            self._judge_client,
            dump_dir=self.output_dir,
            use_cache=True,
            cache_level="prompt",
            cache_tag=self.node_id,
            master_data_dir=self.args.master_data_dir,
            lm_cache_dir=getattr(self.args, "lm_cache_dir", None),
            retain_stats_details=False,
            temperature=0.0,
        )
        self._evaluation_judge_report = self._judge_report()
        self._update_judge_progress(self._evaluation_judge_report, force=True)

    def checkpoint_resources(self) -> None:
        if self._judge_model is not None:
            self._judge_model.save_cache()
        super().checkpoint_resources()

    def close_resources(self) -> None:
        try:
            if self._judge_model is not None:
                self._update_judge_progress(
                    self._judge_report(), force=True
                )
                self._judge_model.save_cache()
        finally:
            try:
                if self._judge_model is not None:
                    asyncio.run(self._judge_model.close())
                elif self._judge_client is not None:
                    asyncio.run(self._judge_client.close())
            finally:
                self._judge_client = None
                self._judge_model = None
                super().close_resources()

    def begin_target(self, target_id: str) -> None:
        super().begin_target(target_id)
        self._target_judge_report = self._judge_report()
        self._target_fallback_start = self._judge_fallback_count

    def target_metadata(self) -> dict:
        current_report = self._judge_report()
        self._update_judge_progress(current_report, force=True)
        return {
            **super().target_metadata(),
            "judge_parse_fallback_count": self._judge_fallback_count - self._target_fallback_start,
            "language_model": self._report_delta(
                self._target_judge_report,
                current_report,
            ),
        }

    def evaluation_metadata(self) -> dict:
        return {
            **super().evaluation_metadata(),
            "judge_parse_fallback_count": self._judge_fallback_count,
            "language_model": self._report_delta(
                self._evaluation_judge_report,
                self._judge_report(),
            ),
        }

    def combine_metadata(self, target_metadata) -> dict:
        target_metadata = tuple(target_metadata)
        combined = dict(super().combine_metadata(iter(target_metadata)))
        combined["judge_parse_fallback_count"] = sum(
            int(metadata.get("judge_parse_fallback_count", 0)) for metadata in target_metadata
        )
        reports = [
            metadata["language_model"]
            for metadata in target_metadata
            if metadata.get("language_model") is not None
        ]
        combined["language_model"] = {
            **{
                key: sum(int(report.get(key, 0)) for report in reports)
                for key in _COUNT_REPORT_FIELDS
            },
            "total_price": sum(
                float(report.get("total_price", 0.0))
                for report in reports
            ),
        }
        return combined

    def _get_judge_ratings(
        self,
        prompts,
        api_names,
        parser,
        min_rating=0.0,
        max_rating=2.0,
        default_rating=0.0,
    ):
        """Request ratings and refresh only responses that fail validation."""
        if self._judge_model is None:
            raise RuntimeError("The judge model resource is not open.")
        prompts = list(prompts)
        if isinstance(api_names, str):
            api_names = [api_names] * len(prompts)
        else:
            api_names = list(api_names)
        if len(api_names) != len(prompts):
            raise ValueError("api_names and prompts must have the same length.")

        async def process():
            completions = list(
                await self._judge_model.chat_completions(
                    api_names,
                    prompts,
                    batch_size=self.judge_concurrency,
                    progress_callback=self._update_judge_progress,
                )
            )
            if len(completions) != len(prompts):
                raise ValueError(
                    "Judge model returned a different number of completions than prompts."
                )

            ratings = [None] * len(prompts)
            pending = list(range(len(prompts)))
            for retry_index in range(self.judge_parse_retries + 1):
                failed = []
                for index in pending:
                    try:
                        rating = parser(completions[index])
                        if rating is None or not min_rating <= rating <= max_rating:
                            raise ValueError(
                                f"Rating {rating!r} is outside "
                                f"[{min_rating}, {max_rating}]."
                            )
                    except Exception:
                        failed.append(index)
                    else:
                        ratings[index] = rating

                if not failed:
                    break
                if retry_index == self.judge_parse_retries:
                    failure_path = Path(self.output_dir) / "judge_parse_failures.jsonl"
                    failure_path.parent.mkdir(parents=True, exist_ok=True)
                    with failure_path.open("a", encoding="utf-8") as file:
                        for index in failed:
                            file.write(json.dumps({
                                "time": time.time(), "node": self.node_id,
                                "api_name": api_names[index], "prompt": prompts[index],
                                "completion": completions[index],
                                "retries": self.judge_parse_retries,
                                "fallback_rating": default_rating,
                            }, ensure_ascii=False) + "\n")
                    for index in failed:
                        ratings[index] = default_rating
                    self._judge_fallback_count += len(failed)
                    logger.warning(
                        "Judge parsing failed after %d retries for %d responses; "
                        "using rating %s. Details: %s",
                        self.judge_parse_retries, len(failed), default_rating, failure_path,
                    )
                    break

                logger.warning(
                    "Judge rating parsing failed for %d response(s); retrying "
                    "without cache (%d/%d).",
                    len(failed),
                    retry_index + 1,
                    self.judge_parse_retries,
                )
                refreshed = list(
                    await self._judge_model.chat_completions(
                        [api_names[index] for index in failed],
                        [prompts[index] for index in failed],
                        batch_size=self.judge_concurrency,
                        refresh_cache=True,
                        progress_callback=self._update_judge_progress,
                    )
                )
                if len(refreshed) != len(failed):
                    raise ValueError(
                        "Judge model returned a different number of retry "
                        "completions than prompts."
                    )
                for index, completion in zip(failed, refreshed):
                    completions[index] = completion
                pending = failed

            return ratings, completions

        return asyncio.run(process())

    def _judge_report(self):
        if self._judge_model is None:
            return dict(_EMPTY_REPORT)
        return self._judge_model.stats.get_report()

    def _update_judge_progress(self, report, force=False):
        progress_bar = self._model_progress_bar
        if progress_bar is None:
            return
        now = time.monotonic()
        if (
            not force
            and now - self._judge_progress_last_update
            < self.judge_progress_interval
        ):
            return
        self._judge_progress_last_update = now
        progress_bar.set_postfix_str(
            " | ".join((
                f"${float(report.get('total_price', 0.0)):.4f}",
                "judge="
                f"{self._format_token_count(report.get('network_calls', 0))}",
                "tok="
                f"{self._format_token_count(report.get('input_tokens', 0))}"
                "+"
                f"{self._format_token_count(report.get('output_tokens', 0))}",
                "hit="
                f"{self._format_token_count(report.get('total_cache_hits', 0))}",
            )),
            refresh=True,
        )

    @staticmethod
    def _format_token_count(value):
        value = int(value)
        for threshold, suffix in (
            (1_000_000_000, "B"),
            (1_000_000, "M"),
            (1_000, "K"),
        ):
            if value >= threshold:
                return f"{value / threshold:.1f}{suffix}"
        return str(value)

    @staticmethod
    def _report_delta(before, after):
        return {
            **{
                key: int(after.get(key, 0)) - int(before.get(key, 0))
                for key in _COUNT_REPORT_FIELDS
            },
            "total_price": float(after.get("total_price", 0.0))
            - float(before.get("total_price", 0.0)),
        }
