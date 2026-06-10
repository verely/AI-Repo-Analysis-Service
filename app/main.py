"""
main.py — Application entrypoint.

Run with:
    uvicorn main:app --reload
"""

import logging
import logging.handlers
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx

from .core.config import get_settings

from .api.route import router

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Ensure `logs/` exists and add a rotating file handler that captures DEBUG.
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, "repo_summarizer.log")
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=5 * 1024 * 1024, backupCount=5
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
# Make sure the root logger allows DEBUG so the file handler can receive it.
root_logger.setLevel(logging.DEBUG)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Repo Summarizer",
    description=(
        "AI-powered GitHub repository summarizer. "
        "Supports quick (single-pass) and deep (chunked) modes."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def _create_shared_httpx_client() -> None:
    cfg = get_settings()
    limits = httpx.Limits(**cfg.LIMITS)
    timeout = cfg.TIMEOUT_CONFIG
    # store a shared AsyncClient on app.state for reuse by LLM providers
    app.state.httpx_client = httpx.AsyncClient(timeout=timeout, limits=limits)


@app.on_event("shutdown")
async def _close_shared_httpx_client() -> None:
    client = getattr(app.state, "httpx_client", None)
    if client is not None:
        await client.aclose()


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok"}
