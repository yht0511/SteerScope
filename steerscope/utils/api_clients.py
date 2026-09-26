"""Role-specific credentials for OpenAI-compatible API clients."""

from __future__ import annotations

import os


_ROLES = {"generation", "judge"}


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
