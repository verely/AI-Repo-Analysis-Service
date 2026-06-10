"""
config.py — Single source of truth for all tuneable parameters.

Every limit, timeout, and model name lives here.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── OpenAI ──────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_base_url: str = "https://api.studio.nebius.ai/v1"
    # Intentionally empty here so environment / .env is the source of truth.
    openai_model: str = ""

    # ── Nebius AI ──────────────────────────────────────────────────────────────
    nebius_api_key: str = ""
    nebius_base_url: str = "https://api.studio.nebius.ai/v1"
    nebius_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    # ── Token budget ───────────────────────────────────────────────────────────
    # Raised to 32K: covers 99%+ of human-written READMEs without truncation.
    # Prefill is parallelised by the model — a 32K prefill is ~2x slower than
    # 12K, not 2.7x, because attention cost is O(n²) but hardware is pipelined.
    quick_mode_max_tokens: int = 32_000

    # Hard ceiling: never send more than this regardless of config.
    # Protects against generated/auto-doc READMEs that can be 100K+ tokens.
    absolute_max_input_tokens: int = 50_000

    prompt_overhead_tokens: int = 600  # reserved for template boilerplate
    max_output_tokens: int = 800  # max tokens LLM may emit per call

    @property
    def quick_readme_budget(self) -> int:
        """Actual README token budget = total - overhead - output headroom."""
        raw = (
            self.quick_mode_max_tokens
            - self.prompt_overhead_tokens
            - self.max_output_tokens
        )
        # Also respect the absolute ceiling
        ceiling = (
            self.absolute_max_input_tokens
            - self.prompt_overhead_tokens
            - self.max_output_tokens
        )
        return min(raw, ceiling)

    # ── Deep mode ──────────────────────────────────────────────────────────────
    deep_mode_chunk_size: int = 300  # 6_000  # tokens per chunk
    deep_mode_max_chunks: int = 5  # hard cap → max 5 LLM chunk calls

    # Max concurrent LLM calls for chunk processing.
    # Raise if Nebius rate limits allow; lower to 1 if you hit 429s.
    deep_concurrent_llm_calls: int = 3

    # ── Tier 2 fetch (deep mode only) ─────────────────────────────────────────
    # Max files to fetch as entry point snippets
    entry_point_max_files: int = 4
    # Lines to read from each entry point file
    entry_point_max_lines: int = 60
    # Max file tree entries to include in prompt (avoids giant monorepo noise)
    tree_max_entries: int = 120

    # ── HTTP / reliability ─────────────────────────────────────────────────────
    github_timeout: float = 10.0
    llm_timeout: float = 60.0
    max_retries: int = 3
    retry_backoff_base: float = 1.5  # wait = base^attempt  (1.5s, 2.25s, 3.37s)

    # Networking tuning for `httpx.AsyncClient` used by LLM providers.
    # `TIMEOUT_CONFIG` is a float number of seconds for request timeout.
    TIMEOUT_CONFIG: float = 60.0

    # Connection limits for httpx.Limits: provide keys accepted by
    # `httpx.Limits(max_keepalive_connections, max_connections)` as a dict
    # so callers can construct `httpx.Limits(**LIMITS)`.
    LIMITS: dict = {
        "max_keepalive_connections": 20,
        "max_connections": 100,
    }


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — safe to call anywhere without re-parsing .env."""
    return Settings()
