"""Tinybird execution backend (the canonical tinybirdco/llm-benchmark target).

Hits Tinybird's `/v0/sql` endpoint with the LLM-generated query. Requires:
    TINYBIRD_WORKSPACE_TOKEN
    TINYBIRD_API_HOST     (e.g. https://api.tinybird.co)

The schema prompt mirrors `tinybirdco/llm-benchmark/src/tinybird/datasources/
github_events.datasource`. We do *not* vendor the full ClickHouse function
list here; if you want a fully faithful 1:1 port of the original prompt,
inject `tinybird/...` resources via `--schema-file`.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx

from eval.executors.base import ExecResult, SqlExecutor

if TYPE_CHECKING:
    from eval.dataset.questions import Question

_DEFAULT_SCHEMA_PROMPT = """\
Table: github_events  (200M rows of public GitHub activity)

Columns (ClickHouse types):
  file_time           DateTime
  event_type          LowCardinality(String)  -- e.g. WatchEvent, IssueCommentEvent, PushEvent...
  actor_login         LowCardinality(String)
  repo_name           LowCardinality(String)
  created_at          DateTime
  updated_at          DateTime
  action              LowCardinality(String)  -- 'opened', 'closed', 'created', ...
  comment_id          UInt64
  body                Nullable(String)
  number              Int32
  title               Nullable(String)
  labels              Array(LowCardinality(String))
  state               LowCardinality(String)
  author_association  LowCardinality(String)  -- NONE, CONTRIBUTOR, OWNER, COLLABORATOR, MEMBER
  closed_at           DateTime
  merged_at           DateTime
  merged              UInt8
  commits             UInt32
  additions           UInt32
  deletions           UInt32
  changed_files       UInt32
  push_size           UInt32
  release_tag_name    String
  review_state        LowCardinality(String)

Notes:
- WatchEvent = a star.
- For PRs/issues, filter action = 'opened' to get newly-opened ones.
"""


class TinybirdExecutor(SqlExecutor):
    """ClickHouse-backed Tinybird workspace executor.

    Defaults to the env-var-driven config used by the upstream Node bench.
    """

    name = "tinybird"

    def __init__(
        self,
        *,
        api_host: str | None = None,
        workspace_token: str | None = None,
        timeout_s: float = 30.0,
        schema_prompt: str | None = None,
    ) -> None:
        self.api_host = (
            api_host
            or os.getenv("TINYBIRD_API_HOST")
            or "https://api.tinybird.co"
        )
        self.workspace_token = workspace_token or os.getenv(
            "TINYBIRD_WORKSPACE_TOKEN"
        )
        self.timeout_s = timeout_s
        self._schema_prompt = schema_prompt or _DEFAULT_SCHEMA_PROMPT
        self._client: httpx.Client | None = None

    def setup(self) -> None:
        if not self.workspace_token:
            raise RuntimeError(
                "TinybirdExecutor requires TINYBIRD_WORKSPACE_TOKEN "
                "(set in env or pass workspace_token=)"
            )
        self._client = httpx.Client(
            timeout=self.timeout_s,
            headers={"Authorization": f"Bearer {self.workspace_token}"},
        )

    def execute(self, sql: str, *, question: Question | None = None) -> ExecResult:
        del question  # global schema; per-question pointers are ignored.
        if self._client is None:
            raise RuntimeError("TinybirdExecutor.setup() not called")

        sql = sql.strip().rstrip(";").strip()
        url = f"{self.api_host}/v0/sql?q={quote(sql)} FORMAT JSON"
        start = time.perf_counter()
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            return ExecResult(
                success=False,
                error=f"network error: {exc}",
                execution_time_s=time.perf_counter() - start,
                backend=self.name,
            )

        elapsed = time.perf_counter() - start
        if response.status_code != 200:
            return ExecResult(
                success=False,
                error=response.text[:500],
                execution_time_s=elapsed,
                backend=self.name,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            return ExecResult(
                success=False,
                error=f"invalid JSON: {exc}",
                execution_time_s=elapsed,
                backend=self.name,
            )

        rows: list[dict[str, Any]] = payload.get("data", []) or []
        cols = [m.get("name", "") for m in (payload.get("meta") or [])]
        return ExecResult(
            success=True,
            rows=rows,
            columns=cols,
            execution_time_s=elapsed,
            backend=self.name,
        )

    def schema_prompt(self) -> str:
        return self._schema_prompt

    def dialect_label(self) -> str:
        return "clickhouse (Tinybird)"

    def teardown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


__all__ = ["TinybirdExecutor"]
