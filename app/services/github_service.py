"""
github_service.py — All GitHub REST API interactions.

Responsibilities:
  - Validate & parse GitHub URL
  - Tier 1 (always): metadata + README + root dependency detection  [3 concurrent calls]
  - Tier 2 (deep mode): filtered full recursive tree + entry point snippets [concurrent]
  - Retry with exponential backoff on network/timeout errors
  - Enforce per-call timeouts
  - Maps HTTP errors to domain exceptions
"""

import asyncio
import base64
import logging
import re

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from ..models.schemas import EntryPointSnippet, RepoData

log = logging.getLogger(__name__)


class GitHubValidationError(ValueError):
    """Raised when the URL is not a valid public GitHub repo URL."""


class GitHubFetchError(RuntimeError):
    """Raised when the GitHub API is unreachable or returns an unexpected error."""


# ── URL validation ─────────────────────────────────────────────────────────────

_GITHUB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?(?:/.*)?$"
)


def parse_github_url(url: str) -> tuple[str, str]:
    url = url.strip().rstrip("/")
    match = _GITHUB_RE.match(url)
    if not match:
        raise GitHubValidationError(
            f"Not a valid public GitHub repository URL: '{url}'. "
            "Expected format: https://github.com/<owner>/<repo>"
        )
    return match.group("owner"), match.group("repo")


# ── Noise filter for full tree ─────────────────────────────────────────────────

# Directories whose entire subtree we discard — generated artifacts, vendored
# deps, caches. Checked as path prefix so "node_modules/foo" is caught.
_NOISE_DIR_PREFIXES = {
    "node_modules/",
    ".git/",
    "dist/",
    "build/",
    "__pycache__/",
    ".venv/",
    "venv/",
    "vendor/",
    "target/",
    ".next/",
    ".nuxt/",
    "coverage/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".tox/",
    "site-packages/",
    "eggs/",
    ".eggs/",
    "htmlcov/",
}

# File extensions that add no architectural signal
_NOISE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".webp",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".min.js",
    ".min.css",
    ".map",
    ".lock",
    ".sum",
}

# Known dependency/manifest filenames (root-level detection)
_KNOWN_DEPENDENCY_FILES = {
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "package.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "go.mod",
    "go.sum",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Gemfile",
    "composer.json",
    "mix.exs",
    "pubspec.yaml",
}

# Entry point candidates by primary language, in priority order
_ENTRY_POINTS_BY_LANGUAGE: dict[str, list[str]] = {
    "Python": [
        "main.py",
        "app.py",
        "server.py",
        "run.py",
        "wsgi.py",
        "asgi.py",
        "manage.py",
        "__main__.py",
    ],
    "JavaScript": [
        "index.js",
        "app.js",
        "server.js",
        "main.js",
        "src/index.js",
        "src/app.js",
    ],
    "TypeScript": [
        "index.ts",
        "app.ts",
        "server.ts",
        "main.ts",
        "src/index.ts",
        "src/app.ts",
        "src/main.ts",
    ],
    "Go": ["main.go", "cmd/main.go", "cmd/server/main.go"],
    "Rust": ["src/main.rs", "src/lib.rs"],
    "Java": ["src/main/java/Application.java", "src/main/java/App.java"],
    "Ruby": ["app.rb", "config.ru", "Rakefile"],
    "PHP": ["index.php", "public/index.php"],
    "C#": ["Program.cs", "Startup.cs"],
    "Swift": ["Sources/main.swift", "main.swift"],
    "Kotlin": ["src/main/kotlin/Main.kt", "src/main/kotlin/Application.kt"],
}

# Always interesting regardless of language
_UNIVERSAL_ENTRY_POINTS = [
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
]


def _is_noisy_path(path: str) -> bool:
    lower = path.lower()
    for prefix in _NOISE_DIR_PREFIXES:
        if lower.startswith(prefix) or f"/{prefix[:-1]}/" in f"/{lower}":
            return True
    for ext in _NOISE_EXTENSIONS:
        if lower.endswith(ext):
            return True
    return False


def _select_entry_points(
    language: str | None,
    all_paths: set[str],
    max_files: int,
) -> list[str]:
    candidates: list[str] = []
    if language and language in _ENTRY_POINTS_BY_LANGUAGE:
        candidates.extend(_ENTRY_POINTS_BY_LANGUAGE[language])
    candidates.extend(_UNIVERSAL_ENTRY_POINTS)

    selected: list[str] = []
    for c in candidates:
        if c in all_paths and c not in selected:
            selected.append(c)
        if len(selected) >= max_files:
            break
    return selected


# ── HTTP client ────────────────────────────────────────────────────────────────


def _make_client() -> httpx.AsyncClient:
    cfg = get_settings()
    return httpx.AsyncClient(
        base_url="https://api.github.com",
        timeout=cfg.github_timeout,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "repo-summarizer/1.0",
        },
    )


def _github_retry():
    cfg = get_settings()
    return retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(cfg.max_retries),
        wait=wait_exponential(multiplier=1, min=cfg.retry_backoff_base, max=10),
        reraise=True,
    )


# ── Individual fetchers ────────────────────────────────────────────────────────


@_github_retry()
async def _fetch_metadata(client: httpx.AsyncClient, owner: str, repo: str) -> dict:
    resp = await client.get(f"/repos/{owner}/{repo}")
    if resp.status_code == 404:
        raise GitHubValidationError(
            f"Repository '{owner}/{repo}' not found or is private."
        )
    resp.raise_for_status()
    return resp.json()


@_github_retry()
async def _fetch_readme(client: httpx.AsyncClient, owner: str, repo: str) -> str | None:
    resp = await client.get(f"/repos/{owner}/{repo}/readme")
    if resp.status_code == 404:
        log.info("No README found for %s/%s", owner, repo)
        return None
    resp.raise_for_status()
    content = resp.json().get("content", "")
    return base64.b64decode(content).decode("utf-8", errors="replace")


@_github_retry()
async def _fetch_root_files(
    client: httpx.AsyncClient, owner: str, repo: str
) -> list[str]:
    resp = await client.get(f"/repos/{owner}/{repo}/contents/")
    if resp.status_code != 200:
        return []
    return [item["name"] for item in resp.json() if item.get("type") == "file"]


@_github_retry()
async def _fetch_full_tree(
    client: httpx.AsyncClient, owner: str, repo: str, branch: str
) -> list[str]:
    """
    Fetch the complete recursive file tree via GitHub Trees API.
    Returns a filtered, sorted list of meaningful paths capped at tree_max_entries.

    Why recursive=1 over multiple /contents/ calls:
      - Single HTTP round-trip vs O(depth) calls
      - GitHub returns up to 100k entries flat; we filter and cap
      - Tradeoff: response can be large (~MB) for huge monorepos, but
        that's one network payload vs dozens of API calls
    """
    cfg = get_settings()
    resp = await client.get(
        f"/repos/{owner}/{repo}/git/trees/{branch}",
        params={"recursive": "1"},
    )
    if resp.status_code != 200:
        log.warning("Could not fetch full tree (status %d)", resp.status_code)
        return []

    data = resp.json()
    if data.get("truncated"):
        log.warning("GitHub truncated the repository tree (>100k entries)")

    paths = [
        item["path"]
        for item in data.get("tree", [])
        if item.get("type") == "blob" and not _is_noisy_path(item["path"])
    ]
    paths.sort()
    return paths[: cfg.tree_max_entries]


@_github_retry()
async def _fetch_file_snippet(
    client: httpx.AsyncClient, owner: str, repo: str, path: str, max_lines: int
) -> EntryPointSnippet | None:
    resp = await client.get(f"/repos/{owner}/{repo}/contents/{path}")
    if resp.status_code != 200:
        return None
    data = resp.json()
    if data.get("encoding") != "base64":
        return None
    try:
        text = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except Exception:
        return None
    return EntryPointSnippet(path=path, lines=text.splitlines()[:max_lines])


# ── Public API ─────────────────────────────────────────────────────────────────


async def fetch_repo_data(url: str, deep: bool = False) -> RepoData:
    """
    Validate URL then fetch repo data.

    Tier 1 (always, 3 concurrent calls):
      metadata + README + root file list

    Tier 2 (deep=True only, concurrent):
      filtered full recursive tree + detected entry point snippets
    """
    owner, repo = parse_github_url(url)
    cfg = get_settings()

    try:
        async with _make_client() as client:

            # ── Tier 1: always ─────────────────────────────────────────────────
            metadata, readme, root_files = await asyncio.gather(
                _fetch_metadata(client, owner, repo),
                _fetch_readme(client, owner, repo),
                _fetch_root_files(client, owner, repo),
            )
            dep_files = [f for f in root_files if f in _KNOWN_DEPENDENCY_FILES]

            if not deep:
                return RepoData(
                    metadata=metadata,
                    readme=readme,
                    has_readme=readme is not None,
                    dependency_files=dep_files,
                )

            # ── Tier 2: deep mode extras ───────────────────────────────────────
            branch = metadata.get("default_branch", "main")
            tree_paths = await _fetch_full_tree(client, owner, repo, branch)

            entry_point_paths = _select_entry_points(
                metadata.get("language"),
                set(tree_paths),
                cfg.entry_point_max_files,
            )
            log.info(
                "Tier 2 — tree: %d entries, entry points: %s",
                len(tree_paths),
                entry_point_paths,
            )

            snippet_results = await asyncio.gather(
                *[
                    _fetch_file_snippet(
                        client, owner, repo, path, cfg.entry_point_max_lines
                    )
                    for path in entry_point_paths
                ],
                return_exceptions=True,
            )

            snippets = [s for s in snippet_results if isinstance(s, EntryPointSnippet)]

            return RepoData(
                metadata=metadata,
                readme=readme,
                has_readme=readme is not None,
                dependency_files=dep_files,
                tree_structure=tree_paths,
                entry_point_snippets=snippets,
            )

    except GitHubValidationError:
        raise
    except httpx.TimeoutException as exc:
        raise GitHubFetchError(f"GitHub API timed out: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise GitHubFetchError(
            f"GitHub API error {exc.response.status_code}: {exc.response.text[:200]}"
        ) from exc
    except Exception as exc:
        raise GitHubFetchError(f"Unexpected error fetching repo data: {exc}") from exc
