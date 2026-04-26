"""LLM SQL-generation eval harness for manysql.

Pluggable along three axes:

1. LLM provider (`eval.llm`): OpenAI, OpenRouter, or any OpenAI-compatible
   server (e.g. a local vLLM serve endpoint).
2. SQL execution backend (`eval.executors`): SQLite (default, hermetic),
   Tinybird (the canonical tinybirdco/llm-benchmark target), or a
   manysql-generated synthetic dialect engine (TODO).
3. Question suite (`eval.dataset`): NL question + reference SQL +
   reference rows. The starter suite mirrors a subset of
   tinybirdco/llm-benchmark's GitHub-events corpus, ported to ANSI/SQLite.

Usage::

    python -m eval --provider openrouter --model anthropic/claude-sonnet-4
    python -m eval --provider vllm --vllm-base-url http://localhost:8000/v1 \
                   --model unsloth/Qwen3-4B-Instruct-2507
    python -m eval --backend tinybird ...   # requires Tinybird credentials
"""

from eval.executors.base import ExecResult, SqlExecutor
from eval.llm import LLMClient, LLMResponse
from eval.runner import BenchmarkRun, run_benchmark
from eval.validator import compare_results

__all__ = [
    "BenchmarkRun",
    "ExecResult",
    "LLMClient",
    "LLMResponse",
    "SqlExecutor",
    "compare_results",
    "run_benchmark",
]
