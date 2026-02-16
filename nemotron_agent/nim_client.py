"""Client for NVIDIA NIM API — cloud-hosted Nemotron inference.

Calls the NIM REST API directly via requests.  No OpenAI dependency.
The same client works against NVIDIA's hosted endpoint or a self-hosted
NIM container on VESSL by changing NVIDIA_NIM_BASE_URL.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import requests

from nemotron_agent.config import NVIDIAConfig

logger = logging.getLogger(__name__)


@dataclass
class NIMClient:
    """Thin wrapper around the NVIDIA NIM chat-completions REST endpoint."""

    cfg: NVIDIAConfig = field(default_factory=NVIDIAConfig)
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        })
        logger.info(
            "NIM client ready  model=%s  endpoint=%s",
            self.cfg.model_id,
            self.cfg.base_url,
        )

    @property
    def _completions_url(self) -> str:
        return f"{self.cfg.base_url.rstrip('/')}/chat/completions"

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.6,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat-completion request and return the assistant text."""
        payload = {
            "model": self.cfg.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = self._session.post(self._completions_url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        logger.debug(
            "NIM response (%s tokens): %.120s…",
            usage.get("completion_tokens", "?"),
            content,
        )
        return content

    def chat_structured(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.6,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Chat and parse the response as JSON (for structured actions)."""
        raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        return json.loads(cleaned.strip())
