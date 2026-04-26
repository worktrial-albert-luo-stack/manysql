"""LLM client wrapper.

Codegen agents call LLMs through this layer, not directly. Centralizing the
client lets us:
- Swap backends (OpenAI / OpenRouter / Anthropic) by changing one env var.
- Cache identical requests deterministically (the codegen refine loop is
  iterative, and identical retries are common).
- Inject deterministic stubs in tests.

Usage::

    from manysql.llm import LLMClient
    client = LLMClient.from_env()
    reply = client.chat(
        system="You are a Lark grammar specialist.",
        user="Generate a grammar for ...",
    )
"""

from manysql.llm.client import (
    LLMBackend,
    LLMClient,
    LLMConfig,
    LLMError,
    LLMMessage,
    LLMResponse,
    NullLLMClient,
)

__all__ = [
    "LLMBackend",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "LLMMessage",
    "LLMResponse",
    "NullLLMClient",
]
