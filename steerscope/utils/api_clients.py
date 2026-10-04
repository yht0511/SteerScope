"""Role-specific credentials for OpenAI-compatible API clients."""

from __future__ import annotations

import os
from copy import deepcopy


_ROLES = {"generation", "judge"}


def apply_api_model_overrides(config: dict) -> dict:
    """Resolve API model overrides before runtime configuration is signed or saved."""
    resolved = deepcopy(config)

    def replace_model(value, model):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "lm_model":
                    value[key] = model
                else:
                    replace_model(child, model)
        elif isinstance(value, list):
            for child in value:
                replace_model(child, model)

    for section, role in (("generate", "GENERATION"), ("train", "GENERATION"), ("evaluate", "JUDGE")):
        model = os.environ.get(f"STEERSCOPE_{role}_MODEL", "").strip()
        if model:
            replace_model(resolved.get(section, {}), model)
    return resolved


def openai_client_credentials(role: str) -> dict[str, str | None]:
    """Return API options for generation or judging.

    Role-specific variables allow the paper's generator and judge models to be
    served by different providers. Standard OpenAI variables remain supported
    as a backwards-compatible fallback.
    """
    normalized = str(role).strip().lower()
    if normalized not in _ROLES:
        raise ValueError(f"Unknown API role {role!r}; expected one of {_ROLES}.")
    prefix = f"STEERSCOPE_{normalized.upper()}"
    api_key = os.environ.get(f"{prefix}_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    base_url = os.environ.get(f"{prefix}_BASE_URL") or os.environ.get(
        "OPENAI_BASE_URL"
    )
    options: dict[str, str | None] = {"api_key": api_key}
    if base_url:
        options["base_url"] = base_url
    return options
