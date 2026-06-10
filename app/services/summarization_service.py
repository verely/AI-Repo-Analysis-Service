"""
summarization_service.py — AI orchestration layer.

Responsibilities:
  - Token estimation (fast heuristic; no tokenizer dependency)
  - Truncation for quick mode with hard ceiling guard
  - Chunking for deep mode with hard cap on number of chunks
  - Calling LLM provider via injected LLMClient
  - Semaphore-bounded concurrent chunk processing
  - Merging partial summaries
  - Parsing structured JSON output into SummaryResult
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from .llm_client import LLMClient

try:  # pragma: no cover
    from config import get_settings  # type: ignore[import]
except ImportError:  # pragma: no cover
    from ..core.config import get_settings

from ..models.schemas import RepoData
from . import prompt_builder

log = logging.getLogger(__name__)


# ── Domain types ───────────────────────────────────────────────────────────────


@dataclass
class SummaryResult:
    # Spec-contract fields — exposed directly in the API response
    summary: str
    technologies: list[str]  # e.g. ["Python", "FastAPI", "httpx"]
    structure: str  # e.g. "src/ holds core logic, tests/ ..."

    # Internal bookkeeping — not exposed in the response
    mode_used: str  # "quick" | "quick-fallback" | "deep"
    chunks_processed: int


class SummarizationError(RuntimeError):
    """Raised when the LLM call fails after all retries."""


# ── Structured output parser ───────────────────────────────────────────────────


def _parse_structured_response(raw: str) -> tuple[str, list[str], str]:
    """
    Parse the LLM JSON response into (summary, technologies, structure).

    The LLM is prompted to return strict JSON. We strip markdown fences
    defensively because some models wrap output in ```json ... ``` blocks
    regardless of instructions.

    Degrades gracefully on malformed JSON — returns the raw text as summary
    with empty technologies/structure so the response contract is always met.
    Never raises to the caller.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)

    try:
        data = json.loads(cleaned)
        return (
            str(data.get("summary", raw)).strip(),
            [str(t) for t in data.get("technologies", [])],
            str(data.get("structure", "")).strip(),
        )
    except json.JSONDecodeError:
        log.warning("LLM returned non-JSON — using raw text as summary")
        return raw.strip(), [], ""


# ── Token estimation ───────────────────────────────────────────────────────────
# Heuristic: ~4 chars per token. Fast, dependency-free, ±15% accurate.
# Swap for tiktoken if tighter budgeting is needed.


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── Truncation ─────────────────────────────────────────────────────────────────


def truncate_to_budget(text: str, token_budget: int) -> tuple[str, bool]:
    """Truncate text at word boundary to fit token_budget. Returns (text, truncated)."""
    if estimate_tokens(text) <= token_budget:
        return text, False
    char_limit = token_budget * 4
    truncated = text[:char_limit]
    last_space = truncated.rfind(" ")
    if last_space > char_limit * 0.8:
        truncated = truncated[:last_space]
    return truncated, True


# ── Chunking ───────────────────────────────────────────────────────────────────


def chunk_text(text: str, chunk_size_tokens: int, max_chunks: int) -> list[str]:
    """Split text into paragraph-aware chunks, hard-capped at max_chunks."""
    chunk_chars = chunk_size_tokens * 4
    chunks: list[str] = []
    start = 0
    while start < len(text) and len(chunks) < max_chunks:
        end = start + chunk_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        snap = text.rfind("\n\n", start, end)
        if snap > start + chunk_chars * 0.5:
            end = snap
        chunks.append(text[start:end])
        start = end
    if start < len(text) and len(chunks) == max_chunks:
        log.warning(
            "max_chunks=%d reached — %d chars dropped", max_chunks, len(text) - start
        )
    return chunks


# ── Semaphore-bounded chunk worker ─────────────────────────────────────────────


async def _bounded_chunk_call(
    sem: asyncio.Semaphore,
    index: int,
    chunk: str,
    total: int,
    metadata: dict,
    llm_client: LLMClient,
) -> tuple[int, str]:
    """
    Process one chunk under the semaphore.
    Returns (original_index, plain_text_summary).

    Chunk prompts intentionally return plain text, not JSON — partial chunks
    lack enough context to populate all three response fields reliably.
    Only the final merge prompt produces JSON.
    """
    async with sem:
        messages = prompt_builder.build_chunk_prompt(chunk, index, total, metadata)
        summary = await llm_client.complete(messages)
    return index, summary


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


async def summarize(
    repo_data: RepoData,
    mode: str,
    llm_client: LLMClient,
) -> SummaryResult:
    """
    Orchestrate the full summarization flow.

    mode: "quick" | "deep"
    Falls back to metadata-only if no README is present regardless of mode.
    """
    cfg = get_settings()

    # ── Fallback: no README ────────────────────────────────────────────────────
    if not repo_data.has_readme or not repo_data.readme:
        messages = prompt_builder.build_metadata_only_prompt(repo_data.metadata)
        raw = await llm_client.complete(messages)
        summary, technologies, structure = _parse_structured_response(raw)
        return SummaryResult(
            summary=summary,
            technologies=technologies,
            structure=structure,
            mode_used="quick-fallback",
            chunks_processed=1,
        )

    # ── Quick mode ─────────────────────────────────────────────────────────────
    if mode == "quick":
        readme_text, truncated = truncate_to_budget(
            repo_data.readme, cfg.quick_readme_budget
        )
        messages = prompt_builder.build_quick_prompt(
            repo_data.metadata, readme_text, truncated
        )
        raw = await llm_client.complete(messages)
        summary, technologies, structure = _parse_structured_response(raw)
        return SummaryResult(
            summary=summary,
            technologies=technologies,
            structure=structure,
            mode_used="quick",
            chunks_processed=1,
        )

    # ── Deep mode ──────────────────────────────────────────────────────────────
    if mode == "deep":
        chunks = chunk_text(
            repo_data.readme, cfg.deep_mode_chunk_size, cfg.deep_mode_max_chunks
        )
        sem = asyncio.Semaphore(cfg.deep_concurrent_llm_calls)
        results = await asyncio.gather(
            *[
                _bounded_chunk_call(
                    sem, i, c, len(chunks), repo_data.metadata, llm_client
                )
                for i, c in enumerate(chunks)
            ]
        )
        # Restore index order — as_completed / gather may return out of order
        partial_summaries = [s for _, s in sorted(results)]
        merge_messages = prompt_builder.build_merge_prompt(
            partial_summaries,
            repo_data.metadata,
            repo_data.tree_structure or None,
            repo_data.entry_point_snippets or None,
        )
        raw = await llm_client.complete(merge_messages)
        summary, technologies, structure = _parse_structured_response(raw)
        return SummaryResult(
            summary=summary,
            technologies=technologies,
            structure=structure,
            mode_used="deep",
            chunks_processed=len(chunks),
        )

    raise ValueError(f"Unknown mode: '{mode}'. Expected 'quick' or 'deep'.")
