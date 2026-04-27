"""Tests for the LLM client wrapper.

The real backends are not exercised here (they cost money and aren't
deterministic). We test:
- Backend selection from env (priority + override)
- Stub client records calls
- chat_json parses well-formed replies and surfaces parse errors
"""

from __future__ import annotations

import json

import pytest

from manysql.llm import (
    LLMBackend,
    LLMClient,
    LLMConfig,
    LLMError,
    LLMMessage,
    NullLLMClient,
)


def test_backend_selection_prefers_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("MANYSQL_LLM_BACKEND", "openai")
    cfg = LLMConfig.from_env()
    assert cfg.backend == LLMBackend.OPENAI


def test_backend_defaults_to_openrouter_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MANYSQL_LLM_BACKEND", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    cfg = LLMConfig.from_env()
    assert cfg.backend == LLMBackend.OPENROUTER


def test_no_keys_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Suppress dotenv refill — otherwise the project .env at the repo root
    # would re-populate the env after our delenv calls.
    monkeypatch.setattr(
        "manysql.llm.client._maybe_load_dotenv", lambda: None, raising=True
    )
    monkeypatch.delenv("MANYSQL_LLM_BACKEND", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError):
        LLMConfig.from_env()


def test_null_client_records_messages() -> None:
    client = NullLLMClient(canned_reply="hello")
    resp = client.chat(system="be terse", user="hi")
    assert resp.text == "hello"
    assert client.record[0]["system"] == "be terse"
    assert client.record[0]["user"] == "hi"


def test_null_client_chat_json_parses() -> None:
    payload = json.dumps({"a": 1, "b": [1, 2]})
    client = NullLLMClient(canned_reply=payload)
    out = client.chat_json(system=None, user="give json")
    assert out == {"a": 1, "b": [1, 2]}


def test_null_client_chat_json_raises_on_bad_payload() -> None:
    client = NullLLMClient(canned_reply="not json")
    with pytest.raises(LLMError):
        client.chat_json(user="give json")


# Reply-cleanup behavior of chat_json. Anthropic (and occasionally other
# providers) like to wrap JSON in a ```json ... ``` markdown fence even
# when explicitly told to return raw JSON. chat_json must tolerate that.

def test_chat_json_strips_json_code_fence() -> None:
    payload = '```json\n{"a": 1, "b": [1, 2]}\n```'
    client = NullLLMClient(canned_reply=payload)
    assert client.chat_json(user="x") == {"a": 1, "b": [1, 2]}


def test_chat_json_strips_bare_code_fence() -> None:
    payload = '```\n{"k": "v"}\n```'
    client = NullLLMClient(canned_reply=payload)
    assert client.chat_json(user="x") == {"k": "v"}


def test_chat_json_strips_leading_and_trailing_prose() -> None:
    payload = 'Sure, here you go:\n{"k": 1}\n— let me know if you need more.'
    client = NullLLMClient(canned_reply=payload)
    assert client.chat_json(user="x") == {"k": 1}


def test_chat_json_handles_braces_inside_strings() -> None:
    # The brace-balancer must respect string state or it'll close the
    # outer object early on a nested literal that contains "}".
    payload = '```json\n{"text": "she said \\"}{\\"", "n": 2}\n```'
    client = NullLLMClient(canned_reply=payload)
    out = client.chat_json(user="x")
    assert out == {"text": 'she said "}{"', "n": 2}


def test_messages_list_passes_through() -> None:
    client = NullLLMClient(canned_reply="ok")
    msgs = [
        LLMMessage(role="system", content="a"),
        LLMMessage(role="user", content="b"),
        LLMMessage(role="assistant", content="c"),
        LLMMessage(role="user", content="d"),
    ]
    client.chat(messages=msgs)
    assert client.record[-1]["messages"] == msgs


def test_real_client_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("MANYSQL_LLM_BACKEND", "openai")
    client = LLMClient.from_env()
    client.close()
    client.close()  # should not raise


# Regression: a malformed HTTP body (truncated, HTML, SSE keep-alive,
# etc.) used to surface as an uncaught json.JSONDecodeError because the
# retry loop only caught httpx.RequestError + LLMError. We now wrap the
# body decode and recover via the existing retry path.

class _FakeHTTPResponse:
    def __init__(self, *, status: int, body: str, json_payload: object = None) -> None:
        self.status_code = status
        self.text = body
        self._json_payload = json_payload

    def json(self) -> object:
        if self._json_payload is None:
            # Mimic httpx behavior: strict json.loads on the body.
            return json.loads(self.text)
        return self._json_payload


class _FakeHTTPClient:
    """Replays a queued list of (response | exception) on each .post()."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def post(self, *args: object, **kwargs: object) -> _FakeHTTPResponse:
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[return-value]

    def close(self) -> None:
        pass


def _real_client(monkeypatch: pytest.MonkeyPatch) -> LLMClient:
    monkeypatch.setattr(
        "manysql.llm.client._maybe_load_dotenv", lambda: None, raising=True
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MANYSQL_LLM_BACKEND", "openai")
    cfg = LLMConfig.from_env()
    # Speed up the backoff so the test runs in milliseconds.
    cfg = LLMConfig(
        backend=cfg.backend,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        default_model=cfg.default_model,
        max_retries=3,
        retry_backoff_seconds=0.0,
    )
    return LLMClient(cfg)


def test_chat_retries_on_malformed_http_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """Truncated/HTML body on attempt 1 retries and succeeds on attempt 2."""
    good_payload = {
        "choices": [{"message": {"content": "hi"}}],
        "model": "x",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    bad = _FakeHTTPResponse(status=200, body="<html>upstream timeout</html>")
    good = _FakeHTTPResponse(status=200, body="{}", json_payload=good_payload)
    client = _real_client(monkeypatch)
    client._client = _FakeHTTPClient([bad, good])  # type: ignore[assignment]
    resp = client.chat(user="hi")
    assert resp.text == "hi"
    assert client._client.calls == 2  # type: ignore[attr-defined]


def test_chat_raises_llmerror_when_all_attempts_yield_bad_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = _FakeHTTPResponse(status=200, body="not-json-at-all")
    client = _real_client(monkeypatch)
    client._client = _FakeHTTPClient([bad, bad, bad])  # type: ignore[assignment]
    with pytest.raises(LLMError) as ei:
        client.chat(user="hi")
    # The wrapped error mentions both the cause and a body preview.
    msg = str(ei.value)
    assert "HTTP body was not JSON" in msg or "body was not JSON" in msg or "after 3 attempts" in msg
    assert client._client.calls == 3  # type: ignore[attr-defined]
