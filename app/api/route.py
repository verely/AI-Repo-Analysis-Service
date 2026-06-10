"""
route.py — FastAPI route definitions.

Endpoints:
  POST /summarize         — standard JSON response

Contract:
  Request:  { "github_url": "https://github.com/owner/repo" }
  Response: { "summary": "...", "technologies": [...], "structure": "..." }

Optional request fields (not in spec, non-breaking):
  "mode": "quick" | "deep"   default: "quick"
"""

import logging
from fastapi import APIRouter, HTTPException, status, Depends
from ..models.request import SummarizeRequest
from ..models.response import RepoMetadata, SummarizeResponse
from ..services.github_service import (
    fetch_repo_data,
    GitHubValidationError,
    GitHubFetchError,
)
from ..services.summarization_service import (
    summarize,
    SummarizationError,
)
from ..services.llm_factory import get_llm_client
from ..services.llm_client import LLMClient

log = logging.getLogger(__name__)
router = APIRouter()


# ── Shared fetch helper ────────────────────────────────────────────────────────


async def _get_repo_data(req: SummarizeRequest):
    """Fetch repo data, selecting tier 1 or tier 2 based on mode."""
    try:
        return await fetch_repo_data(req.github_url, deep=True)
    except GitHubValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    except GitHubFetchError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


# ── POST /summarize — standard JSON ───────────────────────────────────────────


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarise a public GitHub repository",
)
async def summarize_repo(
    req: SummarizeRequest, llm_client: LLMClient = Depends(get_llm_client)
) -> SummarizeResponse:
    """
    Returns a structured summary of the given public GitHub repository.

    The response always contains:
    - **summary** — what the project does and who it's for
    - **technologies** — detected languages, frameworks, and key dependencies
    - **structure** — description of the repository layout
    """
    log.info("Summarize: url=%s mode=%s", req.github_url, "deep")
    repo_data = await _get_repo_data(req)
    try:
        result = await summarize(repo_data, "deep", llm_client)
    except SummarizationError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return SummarizeResponse(
        summary=result.summary,
        technologies=result.technologies,
        structure=result.structure,
    )
