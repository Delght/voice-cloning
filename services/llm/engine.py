"""LLM Engine: HTTP connector to Anything-LLM.

Env vars:
    ANYTHINGLLM_BASE_URL       (default: http://localhost:3001)
    ANYTHINGLLM_API_KEY        (required)
    ANYTHINGLLM_WORKSPACE_SLUG (default: voice-assistant)
"""

import logging
import os
import re

import httpx

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

log = logging.getLogger(__name__)

ANYTHINGLLM_BASE_URL = os.environ.get("ANYTHINGLLM_BASE_URL", "http://localhost:3001")
ANYTHINGLLM_API_KEY = os.environ.get("ANYTHINGLLM_API_KEY", "")
ANYTHINGLLM_WORKSPACE_SLUG = os.environ.get("ANYTHINGLLM_WORKSPACE_SLUG", "voice-assistant")


class LLMEngine:
    """HTTP connector to Anything-LLM workspace chat API."""

    def __init__(self) -> None:
        self.base_url = ANYTHINGLLM_BASE_URL
        self.api_key = ANYTHINGLLM_API_KEY
        self.workspace_slug = ANYTHINGLLM_WORKSPACE_SLUG
        self._client: httpx.AsyncClient | None = None

    def load(self) -> None:
        if not self.api_key:
            log.warning("ANYTHINGLLM_API_KEY is not set.")
        self._client = httpx.AsyncClient(
            timeout=180.0,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        log.info("LLM connector: %s/api/v1/workspace/%s/chat", self.base_url, self.workspace_slug)

    async def unload(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def ready(self) -> bool:
        return self._client is not None

    async def chat(self, message: str) -> str:
        """Send a message to the LLM and return the text response.

        Raises:
            RuntimeError: If Anything-LLM is unreachable or returns an error.
        """
        if not self.ready:
            raise RuntimeError("LLM engine not initialized.")

        url = f"{self.base_url}/api/v1/workspace/{self.workspace_slug}/chat"

        try:
            resp = await self._client.post(url, json={"message": message, "mode": "chat"})
        except httpx.ConnectError as e:
            raise RuntimeError(f"Cannot connect to Anything-LLM at {self.base_url}.") from e
        except httpx.TimeoutException as e:
            raise RuntimeError("Anything-LLM timed out.") from e

        if resp.status_code != 200:
            raise RuntimeError(f"Anything-LLM returned HTTP {resp.status_code}: {resp.text[:200]}")

        text_response = resp.json().get("textResponse", "").strip()
        if not text_response:
            raise RuntimeError("Anything-LLM returned empty textResponse.")

        # Some reasoning models emit <redacted_thinking>...</think>; strip before TTS/UI.
        text_response = re.sub(r"<think>.*?</think>", "", text_response, flags=re.DOTALL).strip()
        if not text_response:
            raise RuntimeError("LLM response was empty after stripping think block.")

        return text_response

    async def health_check(self) -> dict:
        if not self.ready:
            return {"status": "not_initialized"}
        try:
            resp = await self._client.get(f"{self.base_url}/api/v1/auth", timeout=5.0)
            return {
                "status": "ok" if resp.status_code == 200 else "degraded",
                "workspace": self.workspace_slug,
                "anythingllm_url": self.base_url,
            }
        except httpx.ConnectError:
            return {
                "status": "down",
                "workspace": self.workspace_slug,
                "anythingllm_url": self.base_url,
            }
