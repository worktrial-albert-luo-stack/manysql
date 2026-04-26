"""Minimal LLM client supporting OpenAI, OpenRouter, and Anthropic.

Why not the official SDKs?
- We only need chat completions; the official clients add a lot of
  surface area we don't use.
- OpenRouter speaks the OpenAI protocol; one HTTP layer covers both.
- Anthropic uses a different schema (`/v1/messages`, top-level
  `system`, content-block responses) so it gets its own request and
  response handlers, but the public `chat` API stays uniform.
- Keeping this thin makes it trivial to inject stubs in tests.

Configuration is from environment by default but can be passed in
explicitly. We respect a `.env` file at the project root.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx


class LLMBackend(str, Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"


# Default cap for Anthropic's required `max_tokens` field when the
# caller doesn't pass one. Generous but bounded.
_ANTHROPIC_DEFAULT_MAX_TOKENS = 4096
_ANTHROPIC_API_VERSION = "2023-06-01"


@dataclass(frozen=True)
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    backend: LLMBackend
    prompt_tokens: int
    completion_tokens: int
    raw: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised when the LLM call fails after retries or returns malformed data."""


@dataclass
class LLMConfig:
    backend: LLMBackend
    api_key: str
    base_url: str
    default_model: str
    timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    @classmethod
    def from_env(
        cls,
        *,
        backend: Optional[LLMBackend] = None,
        default_model: Optional[str] = None,
    ) -> "LLMConfig":
        _maybe_load_dotenv()
        chosen_backend = backend or _backend_from_env()
        if chosen_backend == LLMBackend.OPENAI:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise LLMError("OPENAI_API_KEY not set")
            return cls(
                backend=LLMBackend.OPENAI,
                api_key=key,
                base_url=os.environ.get(
                    "OPENAI_BASE_URL", "https://api.openai.com/v1"
                ),
                default_model=default_model
                or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            )
        if chosen_backend == LLMBackend.ANTHROPIC:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError("ANTHROPIC_API_KEY not set")
            return cls(
                backend=LLMBackend.ANTHROPIC,
                api_key=key,
                base_url=os.environ.get(
                    "ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"
                ),
                default_model=default_model
                or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
            )
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise LLMError("OPENROUTER_API_KEY not set")
        return cls(
            backend=LLMBackend.OPENROUTER,
            api_key=key,
            base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            default_model=default_model
            or os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-pro"),
        )


class LLMClient:
    """Thin chat-completions client. Backend chosen at construction time."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = httpx.Client(timeout=config.timeout_seconds)

    @classmethod
    def from_env(
        cls,
        *,
        backend: Optional[LLMBackend] = None,
        default_model: Optional[str] = None,
    ) -> "LLMClient":
        return cls(LLMConfig.from_env(backend=backend, default_model=default_model))

    def chat(
        self,
        *,
        system: Optional[str] = None,
        user: Optional[str] = None,
        messages: Optional[list[LLMMessage]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send a chat completion. Either pass `messages` or pass `system`+`user`."""
        if messages is None:
            messages = []
            if system:
                messages.append(LLMMessage(role="system", content=system))
            if user is None:
                raise LLMError("LLMClient.chat needs either `messages` or `user`")
            messages.append(LLMMessage(role="user", content=user))

        if self.config.backend == LLMBackend.ANTHROPIC:
            url, payload = self._build_anthropic_request(
                messages, model, temperature, max_tokens, json_mode
            )
            parse = _parse_anthropic_response
        else:
            url, payload = self._build_openai_request(
                messages, model, temperature, max_tokens, json_mode
            )
            parse = _parse_openai_response

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                resp = self._client.post(url, headers=self._headers(), json=payload)
                if resp.status_code >= 400:
                    raise LLMError(
                        f"HTTP {resp.status_code} from {self.config.backend.value}: "
                        f"{resp.text[:500]}"
                    )
                return parse(resp.json(), self.config.backend)
            except (httpx.RequestError, LLMError) as exc:
                last_error = exc
                if attempt == self.config.max_retries - 1:
                    break
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        raise LLMError(
            f"LLM call failed after {self.config.max_retries} attempts: {last_error}"
        )

    def _build_openai_request(
        self,
        messages: list[LLMMessage],
        model: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        json_mode: bool,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": model or self.config.default_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return f"{self.config.base_url}/chat/completions", payload

    def _build_anthropic_request(
        self,
        messages: list[LLMMessage],
        model: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        json_mode: bool,
    ) -> tuple[str, dict[str, Any]]:
        # Anthropic puts `system` at the top level, not in `messages`.
        # Multiple system messages are concatenated to preserve intent.
        system_parts = [m.content for m in messages if m.role == "system"]
        chat_msgs = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        system_text: Optional[str] = (
            "\n\n".join(s for s in system_parts if s) if system_parts else None
        )
        if json_mode:
            # Anthropic has no `response_format=json_object`; nudge via
            # the system prompt. Caller still parses with json.loads.
            instruction = (
                "Respond with a single valid JSON object and nothing else. "
                "Do not wrap it in markdown fences."
            )
            system_text = (
                f"{system_text}\n\n{instruction}" if system_text else instruction
            )

        payload: dict[str, Any] = {
            "model": model or self.config.default_model,
            "messages": chat_msgs,
            "temperature": temperature,
            # `max_tokens` is required by Anthropic.
            "max_tokens": max_tokens
            if max_tokens is not None
            else _ANTHROPIC_DEFAULT_MAX_TOKENS,
        }
        if system_text:
            payload["system"] = system_text
        return f"{self.config.base_url}/messages", payload

    def chat_json(
        self,
        *,
        system: Optional[str] = None,
        user: Optional[str] = None,
        messages: Optional[list[LLMMessage]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Convenience: ask for a JSON object reply and parse it."""
        resp = self.chat(
            system=system,
            user=user,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            return json.loads(resp.text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Reply was not valid JSON: {exc}\n--- reply ---\n{resp.text}")

    def _headers(self) -> dict[str, str]:
        if self.config.backend == LLMBackend.ANTHROPIC:
            # Anthropic uses x-api-key, not Bearer auth, and requires a
            # version header pinned at construction time.
            return {
                "x-api-key": self.config.api_key,
                "anthropic-version": _ANTHROPIC_API_VERSION,
                "Content-Type": "application/json",
            }
        h = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.backend == LLMBackend.OPENROUTER:
            # OpenRouter recommends these headers for analytics; they're optional.
            h.setdefault(
                "HTTP-Referer",
                os.environ.get("OPENROUTER_REFERER", "https://github.com/manysql"),
            )
            h.setdefault(
                "X-Title", os.environ.get("OPENROUTER_TITLE", "manysql codegen")
            )
        return h

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class NullLLMClient(LLMClient):
    """Test double: returns a fixed reply for every call.

    Lets us write codegen tests without hitting the network.
    """

    def __init__(
        self,
        canned_reply: str = "{}",
        *,
        record: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        # Skip parent init; we don't need a real httpx client.
        self.config = LLMConfig(
            backend=LLMBackend.OPENAI,
            api_key="null",
            base_url="null",
            default_model="null",
        )
        self.canned_reply = canned_reply
        self.record: list[dict[str, Any]] = record if record is not None else []

    def chat(  # type: ignore[override]
        self,
        *,
        system: Optional[str] = None,
        user: Optional[str] = None,
        messages: Optional[list[LLMMessage]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.record.append(
            {
                "system": system,
                "user": user,
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "json_mode": json_mode,
            }
        )
        return LLMResponse(
            text=self.canned_reply,
            model=model or self.config.default_model,
            backend=self.config.backend,
            prompt_tokens=0,
            completion_tokens=0,
        )

    def close(self) -> None:
        return None


def _backend_from_env() -> LLMBackend:
    raw = os.environ.get("MANYSQL_LLM_BACKEND")
    if raw:
        try:
            return LLMBackend(raw.lower())
        except ValueError as exc:
            raise LLMError(
                f"MANYSQL_LLM_BACKEND={raw!r} is invalid; "
                "expected 'openai', 'openrouter', or 'anthropic'"
            ) from exc
    # Auto-detection priority: OpenRouter > OpenAI > Anthropic. This
    # preserves prior behavior; users with multiple keys set should
    # pick explicitly via MANYSQL_LLM_BACKEND.
    if os.environ.get("OPENROUTER_API_KEY"):
        return LLMBackend.OPENROUTER
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLMBackend.ANTHROPIC
    if os.environ.get("OPENAI_API_KEY"):
        return LLMBackend.OPENAI
    raise LLMError(
        "No LLM backend configured: set OPENAI_API_KEY, OPENROUTER_API_KEY, "
        "or ANTHROPIC_API_KEY"
    )


def _parse_openai_response(data: dict[str, Any], backend: LLMBackend) -> LLMResponse:
    try:
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            model=data.get("model", "unknown"),
            backend=backend,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            raw=data,
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(
            f"Malformed chat-completions response from {backend.value}: {exc}\n{data}"
        ) from exc


def _parse_anthropic_response(
    data: dict[str, Any], backend: LLMBackend = LLMBackend.ANTHROPIC
) -> LLMResponse:
    # Anthropic responses look like:
    # {"content": [{"type": "text", "text": "..."}], "usage": {...}}
    # Multiple text blocks can appear (e.g. with thinking) so we
    # concatenate every text block in order.
    try:
        blocks = data["content"]
        text = "".join(
            b.get("text", "") for b in blocks if b.get("type") == "text"
        )
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            model=data.get("model", "unknown"),
            backend=backend,
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0),
            raw=data,
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(
            f"Malformed messages response from {backend.value}: {exc}\n{data}"
        ) from exc


def _maybe_load_dotenv() -> None:
    """Best-effort load of project .env without forcing python-dotenv usage.

    If python-dotenv is installed and a .env exists at the project root,
    load it. Otherwise no-op.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:  # pragma: no cover - dotenv is in deps; guard for safety
        return
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for c in candidates:
        if c.exists():
            load_dotenv(c)
            return
