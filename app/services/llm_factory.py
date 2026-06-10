import os
from fastapi import Depends, Request

from .llm_client import LLMClient
from .llm_providers.openai_client import OpenAIClient


def get_llm_client(request: Request) -> LLMClient:
    """FastAPI dependency to obtain an LLM client instance."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    shared_client = getattr(request.app.state, "httpx_client", None)
    if provider == "openai":
        return OpenAIClient(httpx_client=shared_client)
    raise RuntimeError(f"Unknown LLM provider: {provider}")
