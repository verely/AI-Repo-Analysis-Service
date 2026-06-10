from typing import Protocol

# ── LLM client ─────────────────────────────────────────────────────────────────


# LLM calls are performed via an injected `LLMClient` instance.
# Implementations live in `app.services.llm_providers` and are provided
# to FastAPI endpoints via a dependency (see `llm_factory.py`).


class LLMClient(Protocol):
    """Lightweight protocol for an async LLM client adapter.

    Implementations must provide a non-streaming `complete` method
    that returns the final text
    """

    async def complete(
        self, messages: list[dict]
    ) -> str:  # pragma: no cover - interface
        ...
