from __future__ import annotations

from typing import AsyncGenerator
import os
import json
import httpx
import logging

from ..llm_client import LLMClient
from ...core.config import get_settings


logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(self, settings=None, httpx_client=None):
        cfg = settings or get_settings()
        self._settings = cfg
        # Prefer values from Settings (which reads .env) but allow runtime
        # overrides via environment variables.
        api_key = getattr(cfg, "openai_api_key", None)
        if not api_key or api_key == "change-me":
            api_key = os.getenv("OPENAI_API_KEY")
        self._api_key = api_key

        base_url = getattr(cfg, "openai_base_url", None) or os.getenv("OPENAI_BASE_URL")
        self._base_url = base_url or "https://api.openai.com/v1"
        # optional shared httpx.AsyncClient provided by app startup
        self._httpx_client = httpx_client

    async def complete(self, messages: list[dict]) -> str:
        cfg = self._settings
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        # Model selection order:
        # 1. `OPENAI_MODEL` env var
        # 2. `Settings.openai_model` (from .env via pydantic)
        # If missing, fail early to avoid an invalid API call.
        model = os.getenv("OPENAI_MODEL") or cfg.openai_model
        if not model:
            logger.error(
                "No OpenAI model configured. Set OPENAI_MODEL env var or openai_model in .env"
            )
            raise RuntimeError(
                "OpenAI model not configured. Set OPENAI_MODEL or openai_model in settings"
            )

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": cfg.max_output_tokens,
            "temperature": 0.3,
        }

        # logger.debug("OpenAIClient.complete called")
        # logger.debug("openai_model=%s", cfg.openai_model)
        # logger.debug("OPENAI_API_KEY set=%s", bool(self._api_key))
        # logger.debug("POST %s", url)
        # logger.debug(
        #     "payload keys=%s, messages_len=%s",
        #     list(payload.keys()),
        #     len(messages),
        # )

        try:
            if self._httpx_client is not None:
                resp = await self._httpx_client.post(
                    url,
                    json=payload,
                    headers=headers,
                )
            else:
                limits = httpx.Limits(**cfg.LIMITS)
                timeout = cfg.TIMEOUT_CONFIG
                async with httpx.AsyncClient(
                    timeout=timeout,
                    limits=limits,
                ) as client:
                    resp = await client.post(
                        url,
                        json=payload,
                        headers=headers,
                    )

            logger.debug("OpenAI response status=%s", resp.status_code)
            resp.raise_for_status()
            data = resp.json()
            logger.debug(
                "OpenAI response choices=%s",
                len(data.get("choices", [])),
            )
            return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPError:
            logger.exception("HTTP error while calling OpenAI completions endpoint")
            raise
        except Exception:
            logger.exception("Unexpected error in OpenAIClient.complete")
            raise
