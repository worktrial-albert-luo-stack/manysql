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
