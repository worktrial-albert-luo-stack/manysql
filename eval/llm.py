"""LLM client for the eval harness.

Single OpenAI-compatible chat-completions client backing three providers:

    * `openai`     - https://api.openai.com/v1, key from $OPENAI_API_KEY
    * `openrouter` - https://openrouter.ai/api/v1, key from $OPENROUTER_API_KEY
    * `vllm`       - any local server speaking the OpenAI chat API
                     (e.g. `vllm serve --port 8000`); base URL is required.

We deliberately do not adopt the `openai` Python SDK to keep the dependency
surface small (httpx is already a project dep). The chat-completions wire
format is stable enough that this fits in <200 lines.

Per-model quirks we paper over:
  * Reasoning-class OpenAI models (o1/o3/o4/gpt-5*) reject the legacy
    `max_tokens` parameter and require `max_completion_tokens`. They also
    reject non-default `temperature`. We auto-detect by model id and adjust.
  * Reasoning models can spend most of their token budget on hidden
    thinking, so we bump the default cap when one is detected.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

# OpenAI's reasoning models (and GPT-5 family) use the new chat-completions
# parameter names. Match a leading model id like 'gpt-5', 'gpt-5-nano',
# 'o1', 'o1-mini', 'o3', 'o3-pro', 'o4-mini', etc.
_OPENAI_REASONING_RE = re.compile(r"^(gpt-5|o[134])(-|$)")

# Provider-specific env var, in lookup order.
PROVIDER_ENV_KEYS = {
    "openai": ["OPENAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "vllm": ["VLLM_API_KEY"],  # most local servers don't enforce auth; optional.
}


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_s": self.duration_s,
            "error": self.error,
        }


class LLMClient:
    """Minimal OpenAI-compatible chat-completions client."""

    def __init__(
        self,
        *,
        provider: str = "openrouter",
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        provider = provider.lower()
        self.provider = provider
        self.model = model
        self.base_url = (base_url or DEFAULT_URLS.get(provider, "")).rstrip("/")
        if not self.base_url:
            raise ValueError(
                f"provider={provider!r} requires base_url= "
                "(e.g. http://localhost:8000/v1 for vLLM)"
            )

        self.api_key = api_key or _lookup_key(provider)
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if provider == "openrouter":
            # Optional but encouraged by OpenRouter for analytics.
            headers.setdefault("HTTP-Referer", "https://github.com/manysql/manysql")
            headers.setdefault("X-Title", "manysql-eval")
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout_s,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def chat(
        self,
        *,
        system: str,
        user: str,
        messages: list[dict[str, str]] | None = None,
    ) -> LLMResponse:
        """One chat round-trip. Either pass `system`/`user`, or override with `messages`."""
        msgs = messages or [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
        }
        is_reasoning = (
            self.provider == "openai"
            and bool(_OPENAI_REASONING_RE.match(self.model))
        )
        if is_reasoning:
            # Reasoning models: only `max_completion_tokens` is accepted, and
            # they reject any non-default temperature. They also need a much
            # larger budget because hidden reasoning tokens count against it.
            body["max_completion_tokens"] = max(self.max_tokens, 4096)
        else:
            body["temperature"] = self.temperature
            body["max_tokens"] = self.max_tokens

        start = time.perf_counter()
        try:
            r = self._client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            return LLMResponse(
                text="",
                model=self.model,
                provider=self.provider,
                duration_s=time.perf_counter() - start,
                error=f"network error: {exc}",
            )
        elapsed = time.perf_counter() - start

        if r.status_code != 200:
            return LLMResponse(
                text="",
                model=self.model,
                provider=self.provider,
                duration_s=elapsed,
                error=f"HTTP {r.status_code}: {r.text[:500]}",
            )

        try:
            payload = r.json()
        except json.JSONDecodeError as exc:
            return LLMResponse(
                text="",
                model=self.model,
                provider=self.provider,
                duration_s=elapsed,
                error=f"invalid JSON response: {exc}",
            )

        choices = payload.get("choices") or []
        if not choices:
            return LLMResponse(
                text="",
                model=self.model,
                provider=self.provider,
                duration_s=elapsed,
                error=f"no choices in response: {payload}",
                raw=payload,
            )

        text = (choices[0].get("message") or {}).get("content") or ""
        usage = payload.get("usage") or {}
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            duration_s=elapsed,
            raw=payload,
        )


def _lookup_key(provider: str) -> str | None:
    for env in PROVIDER_ENV_KEYS.get(provider, []):
        v = os.getenv(env)
        if v:
            return v
    return None


__all__ = ["DEFAULT_URLS", "LLMClient", "LLMResponse"]
