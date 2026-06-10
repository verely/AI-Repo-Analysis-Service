"""
prompt_builder.py — All prompt templates live here.

IMPORTANT: As of v1.3 all prompts instruct the model to return strict JSON
matching the spec response contract:
  {
    "summary": "...",
    "technologies": ["...", "..."],
    "structure": "..."
  }

Chunk prompts (deep mode) return plain text partial summaries — only the
final merge prompt requests JSON. This avoids JSON parsing on every chunk
and keeps chunk prompts lean.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.github_service import EntryPointSnippet

PROMPT_VERSION = "v1.3"

_SYSTEM = (
    "You are a senior software engineer writing concise, accurate project "
    "summaries for a technical audience. Be factual. Do not invent features. "
    "If information is missing, say so rather than guessing."
)

# ── JSON output instruction (appended to all final-response prompts) ───────────
_JSON_INSTRUCTION = """
---
Respond with ONLY a JSON object — no markdown fences, no preamble, no explanation.
The JSON must have exactly these three keys:

{
  "summary": "2-4 sentence description of what the project does and who it is for",
  "technologies": ["list", "of", "detected", "languages", "frameworks", "libraries"],
  "structure": "1-2 sentence description of the repository layout and key directories"
}
"""


def _fmt_metadata(metadata: dict) -> str:
    topics = ", ".join(metadata.get("topics", [])) or "none listed"
    return (
        f"Repository : {metadata.get('full_name', 'unknown')}\n"
        f"Language   : {metadata.get('language') or 'not specified'}\n"
        f"Stars      : {metadata.get('stargazers_count', 0):,}\n"
        f"Topics     : {topics}\n"
        f"Description: {metadata.get('description') or 'none'}\n"
    )


def _fmt_tree(paths: list[str]) -> str:
    return "\n".join(paths) if paths else "(not available)"


def _fmt_entry_points(snippets: list[EntryPointSnippet]) -> str:
    return "\n\n".join(s.format() for s in snippets) if snippets else "(none detected)"


# ── Quick mode ─────────────────────────────────────────────────────────────────


def build_quick_prompt(metadata: dict, readme: str, truncated: bool) -> list[dict]:
    truncation_note = (
        "\n\u26a0\ufe0f  Note: The README was truncated to fit the token budget."
        if truncated
        else ""
    )
    user_content = (
        f"## Repository Metadata\n{_fmt_metadata(metadata)}\n"
        f"## README Content\n{readme}{truncation_note}\n"
        f"{_JSON_INSTRUCTION}"
        f"[prompt:{PROMPT_VERSION}/quick]"
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_content},
    ]


def build_metadata_only_prompt(metadata: dict) -> list[dict]:
    user_content = (
        f"## Repository Metadata (no README available)\n{_fmt_metadata(metadata)}\n"
        "Clearly state in the summary that no README was found.\n"
        f"{_JSON_INSTRUCTION}"
        f"[prompt:{PROMPT_VERSION}/metadata-fallback]"
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ── Deep mode — chunk prompts return plain text (not JSON) ─────────────────────


def build_chunk_prompt(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    metadata: dict,
) -> list[dict]:
    """
    Chunk prompts intentionally return plain text, not JSON.
    Reasons:
      1. Partial chunks don't have enough context to fill all three JSON fields
      2. Parsing JSON on every chunk call adds fragility with no benefit
      3. The merge prompt is the only one that produces the final JSON output
    """
    user_content = (
        f"## Repository: {metadata.get('full_name', 'unknown')} "
        f"— README Part {chunk_index + 1} of {total_chunks}\n\n"
        f"{chunk}\n\n"
        "---\n"
        "Write a concise plain-text summary of what is covered in THIS section only. "
        "Note any technologies, frameworks, or architectural patterns you observe. "
        "This will be merged with summaries of other sections — do not repeat the repo name.\n\n"
        f"[prompt:{PROMPT_VERSION}/deep-chunk]"
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ── Deep mode — merge prompt returns JSON ─────────────────────────────────────


def build_merge_prompt(
    partial_summaries: list[str],
    metadata: dict,
    tree_structure: list[str] | None = None,
    entry_point_snippets: list[EntryPointSnippet] | None = None,
) -> list[dict]:
    """
    The merge step is the only deep-mode prompt that returns JSON.
    It receives all partial plain-text summaries plus the tier 2 signals
    (file tree, entry points) for architecture inference.
    """
    parts = "\n\n---\n\n".join(
        f"[Part {i + 1}]\n{s}" for i, s in enumerate(partial_summaries)
    )

    tree_section = ""
    if tree_structure:
        tree_section = (
            f"\n## Repository File Tree ({len(tree_structure)} entries)\n"
            f"{_fmt_tree(tree_structure)}\n"
        )

    entry_section = ""
    if entry_point_snippets:
        entry_section = (
            f"\n## Entry Point Snippets\n"
            f"{_fmt_entry_points(entry_point_snippets)}\n"
        )

    user_content = (
        f"## Repository Metadata\n{_fmt_metadata(metadata)}"
        f"{tree_section}"
        f"{entry_section}"
        f"\n## Partial README Summaries\n{parts}\n"
        "\nMerge the above into a single analysis. "
        "Use the file tree and entry point snippets to validate and enrich "
        "your understanding of the tech stack and structure.\n"
        f"{_JSON_INSTRUCTION}"
        f"[prompt:{PROMPT_VERSION}/deep-merge]"
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_content},
    ]
