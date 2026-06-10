from dataclasses import dataclass, field

# ── Domain types ───────────────────────────────────────────────────────────────


@dataclass
class EntryPointSnippet:
    path: str
    lines: list[str]  # first N lines only

    def format(self) -> str:
        content = "\n".join(self.lines)
        return f"### {self.path}\n```\n{content}\n```"


@dataclass
class RepoData:
    """Internal payload from GitHub used to build the LLM prompt."""

    # Tier 1 — always present
    metadata: dict
    readme: str | None  # raw text; None = not found
    has_readme: bool
    dependency_files: list[str] = field(default_factory=list)

    # Tier 2 — populated in deep mode only
    tree_structure: list[str] = field(default_factory=list)  # filtered file paths
    entry_point_snippets: list[EntryPointSnippet] = field(default_factory=list)
