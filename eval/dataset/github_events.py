"""SQLite-flavored synthetic GitHub-events dataset.

Schema mirrors the columns the tinybirdco/llm-benchmark prompts assume to
exist, with ClickHouse-only types collapsed to SQLite affinities:

  * LowCardinality(String) / Nullable(String) / Enum*  -> TEXT
  * Array(LowCardinality(String))                      -> TEXT (comma-sep)
  * UInt*/Int*                                         -> INTEGER
  * DateTime                                           -> TEXT (ISO 8601 UTC)
  * UInt8 booleans                                     -> INTEGER (0/1)

Datetimes are ISO-8601 strings so SQLite's date/time functions
(`strftime`, `date`, `julianday`) work out of the box. The LLM is told
this in the prompt.

The seeder is deterministic (`random.Random(seed)`) so reference SQL
results are stable across runs.
"""

from __future__ import annotations

import datetime as _dt
import random
from typing import Any

SCHEMA_DDL = """
DROP TABLE IF EXISTS github_events;
CREATE TABLE github_events (
    file_time           TEXT NOT NULL,
    event_type          TEXT NOT NULL,
    actor_login         TEXT NOT NULL DEFAULT '',
    repo_name           TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    action              TEXT NOT NULL,
    comment_id          INTEGER NOT NULL DEFAULT 0,
    commit_id           TEXT NOT NULL DEFAULT '',
    body                TEXT,
    ref                 TEXT NOT NULL DEFAULT '',
    number              INTEGER NOT NULL DEFAULT 0,
    title               TEXT,
    labels              TEXT NOT NULL DEFAULT '',
    state               TEXT NOT NULL DEFAULT '',
    locked              INTEGER NOT NULL DEFAULT 0,
    assignee            TEXT NOT NULL DEFAULT '',
    comments            INTEGER NOT NULL DEFAULT 0,
    author_association  TEXT NOT NULL DEFAULT 'NONE',
    closed_at           TEXT NOT NULL DEFAULT '',
    merged_at           TEXT NOT NULL DEFAULT '',
    merged              INTEGER NOT NULL DEFAULT 0,
    commits             INTEGER NOT NULL DEFAULT 0,
    additions           INTEGER NOT NULL DEFAULT 0,
    deletions           INTEGER NOT NULL DEFAULT 0,
    changed_files       INTEGER NOT NULL DEFAULT 0,
    push_size           INTEGER NOT NULL DEFAULT 0,
    release_tag_name    TEXT NOT NULL DEFAULT '',
    release_name        TEXT NOT NULL DEFAULT '',
    review_state        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX idx_event_type   ON github_events(event_type);
CREATE INDEX idx_repo_name    ON github_events(repo_name);
CREATE INDEX idx_actor_login  ON github_events(actor_login);
CREATE INDEX idx_created_at   ON github_events(created_at);
"""

SCHEMA_PROMPT = """\
Table: github_events  (synthetic subset of GitHub public activity, 2015-2025)

Columns (SQLite types):
  file_time           TEXT       -- ISO 8601, e.g. '2024-04-26 12:30:00'
  event_type          TEXT       -- WatchEvent, PushEvent, PullRequestEvent, IssueCommentEvent,
                                 --   IssuesEvent, CreateEvent, DeleteEvent, ForkEvent,
                                 --   PullRequestReviewEvent, PullRequestReviewCommentEvent,
                                 --   ReleaseEvent, MemberEvent, GollumEvent, CommitCommentEvent
  actor_login         TEXT       -- GitHub username
  repo_name           TEXT       -- 'owner/repo'
  created_at          TEXT       -- ISO 8601
  updated_at          TEXT       -- ISO 8601
  action              TEXT       -- 'opened', 'closed', 'created', 'merged', 'edited', 'reopened',
                                 --   'labeled', 'assigned', 'none', ...
  comment_id          INTEGER
  commit_id           TEXT       -- 40-char hex (only for CommitCommentEvent rows; '' otherwise)
  body                TEXT       -- comment body (only for *CommentEvent rows)
  ref                 TEXT       -- e.g. 'refs/heads/main' (only for PushEvent rows; '' otherwise)
  number              INTEGER    -- PR/issue number
  title               TEXT
  labels              TEXT       -- comma-separated labels, e.g. 'bug,help wanted'
  state               TEXT       -- 'open' | 'closed'
  locked              INTEGER    -- 0/1
  assignee            TEXT
  comments            INTEGER
  author_association  TEXT       -- NONE, CONTRIBUTOR, OWNER, COLLABORATOR, MEMBER
  closed_at           TEXT       -- ISO 8601 ('' if not closed)
  merged_at           TEXT       -- ISO 8601 ('' if not merged)
  merged              INTEGER    -- 0/1
  commits             INTEGER
  additions           INTEGER
  deletions           INTEGER
  changed_files       INTEGER
  push_size           INTEGER
  release_tag_name    TEXT
  release_name        TEXT
  review_state        TEXT       -- approved, changes_requested, commented, dismissed, pending

Notes:
- WatchEvent represents starring a repo.
- For "newly opened PRs/issues" filter `action = 'opened'`.
- `labels` is a comma-separated string. Use LIKE '%bug%' or split on ',' for matching.
- All datetime columns are TEXT in ISO 8601 UTC. Use `strftime('%Y-%m', created_at)` etc.
- Empty datetime/text columns are '' (not NULL).
- `created_at` years span 2015..2025, biased toward recent years.
"""

# --- seed corpus --------------------------------------------------------

EVENT_TYPES = [
    "WatchEvent",
    "PushEvent",
    "PullRequestEvent",
    "IssueCommentEvent",
    "IssuesEvent",
    "CreateEvent",
    "DeleteEvent",
    "ForkEvent",
    "PullRequestReviewEvent",
    "PullRequestReviewCommentEvent",
    "ReleaseEvent",
    "MemberEvent",
    "GollumEvent",
    "CommitCommentEvent",
]

# Weights chosen so PushEvent / WatchEvent dominate (mirroring real GH archive).
EVENT_TYPE_WEIGHTS = [
    30,  # WatchEvent
    40,  # PushEvent
    10,  # PullRequestEvent
    8,  # IssueCommentEvent
    8,  # IssuesEvent
    6,  # CreateEvent
    2,  # DeleteEvent
    5,  # ForkEvent
    3,  # PullRequestReviewEvent
    3,  # PullRequestReviewCommentEvent
    2,  # ReleaseEvent
    1,  # MemberEvent
    1,  # GollumEvent
    1,  # CommitCommentEvent
]

REPO_POOL = [
    "torvalds/linux",
    "facebook/react",
    "pytorch/pytorch",
    "tensorflow/tensorflow",
    "rust-lang/rust",
    "golang/go",
    "vuejs/vue",
    "kubernetes/kubernetes",
    "django/django",
    "pandas-dev/pandas",
    "numpy/numpy",
    "scikit-learn/scikit-learn",
    "huggingface/transformers",
    "tinybirdco/tinybird",
    "openai/openai-python",
    "anthropics/anthropic-sdk-python",
    "polars-rs/polars",
    "duckdb/duckdb",
    "sqlglot/sqlglot",
    "apache/arrow",
]

ACTOR_POOL = [
    "alice",
    "bob",
    "carol",
    "dave",
    "eve",
    "frank",
    "grace",
    "heidi",
    "ivan",
    "judy",
    "ken",
    "lola",
    "mallory",
    "niaj",
    "olivia",
    "peggy",
    "quinn",
    "rupert",
    "sybil",
    "trent",
]

LABELS_POOL = [
    "bug",
    "enhancement",
    "documentation",
    "good first issue",
    "help wanted",
    "question",
    "wontfix",
    "duplicate",
    "performance",
    "security",
]

AUTHOR_ASSOC = [
    "NONE",
    "CONTRIBUTOR",
    "OWNER",
    "COLLABORATOR",
    "MEMBER",
]

ACTIONS = [
    "none",
    "created",
    "edited",
    "deleted",
    "opened",
    "closed",
    "reopened",
    "labeled",
    "assigned",
    "merged",
]

REVIEW_STATES = ["approved", "changes_requested", "commented", "dismissed", "pending"]


# Year sampling weights: a slight bias toward more recent years so the
# distribution is roughly realistic, but every year (2015..2025) gets a
# meaningful share so YoY-style queries (e.g. 2016 vs 2017) work.
_YEAR_RANGE = list(range(2015, 2026))
_YEAR_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4]

_BRANCHES = [
    "refs/heads/main",
    "refs/heads/master",
    "refs/heads/develop",
    "refs/heads/feature/x",
    "refs/heads/release",
]
_BRANCH_WEIGHTS = [10, 6, 3, 2, 1]


def _hex_id(rng: random.Random, length: int = 40) -> str:
    return "".join(rng.choices("0123456789abcdef", k=length))


def seed_rows(*, seed: int = 0xDB, n: int = 5_000) -> list[dict[str, Any]]:
    """Generate `n` deterministic synthetic event rows."""
    rng = random.Random(seed)

    # Reuse a small pool of commit ids per repo so CommitCommentEvent rows
    # repeat commits enough for "comments per commit" aggregates to surface
    # actual top-K results (rather than every row being a unique commit).
    commit_pool: dict[str, list[str]] = {
        repo: [_hex_id(rng) for _ in range(8)] for repo in REPO_POOL
    }

    rows: list[dict[str, Any]] = []
    for _ in range(n):
        evt = rng.choices(EVENT_TYPES, weights=EVENT_TYPE_WEIGHTS, k=1)[0]
        repo = rng.choice(REPO_POOL)
        actor = rng.choice(ACTOR_POOL)

        # Deterministic timestamp spanning 2015..2025 with mild recency bias
        # so YoY queries (2016/2017) and "year >= 2015" filters all return
        # populated rows.
        year = rng.choices(_YEAR_RANGE, weights=_YEAR_WEIGHTS, k=1)[0]
        day_of_year = rng.randint(0, 364)
        minute_of_day = rng.randint(0, 24 * 60 - 1)
        created = (
            _dt.datetime(year, 1, 1, tzinfo=_dt.UTC)
            + _dt.timedelta(days=day_of_year, minutes=minute_of_day)
        )
        created_iso = created.strftime("%Y-%m-%d %H:%M:%S")

        action = "none"
        state = ""
        merged = 0
        merged_at = ""
        closed_at = ""
        body = None
        number = 0
        title = None
        labels = ""
        push_size = 0
        commits = 0
        additions = 0
        deletions = 0
        changed_files = 0
        review_state = ""
        release_tag = ""
        release_name = ""
        commit_id = ""
        ref = ""

        if evt == "PushEvent":
            push_size = rng.randint(1, 25)
            commits = push_size
            additions = rng.randint(0, 800)
            deletions = rng.randint(0, 400)
            changed_files = rng.randint(1, 30)
            ref = rng.choices(_BRANCHES, weights=_BRANCH_WEIGHTS, k=1)[0]
        elif evt in {"PullRequestEvent", "IssuesEvent"}:
            action = rng.choices(
                ["opened", "closed", "reopened", "edited", "labeled"],
                weights=[50, 30, 5, 10, 5],
                k=1,
            )[0]
            state = "closed" if action == "closed" else "open"
            number = rng.randint(1, 50_000)
            title = f"{evt[:-5]} #{number}"
            labels = ",".join(
                rng.sample(LABELS_POOL, k=rng.randint(0, 3))
            )
            if evt == "PullRequestEvent":
                if action == "closed" and rng.random() < 0.6:
                    merged = 1
                    merged_at = created_iso
                additions = rng.randint(0, 2000)
                deletions = rng.randint(0, 1000)
                changed_files = rng.randint(1, 50)
                commits = rng.randint(1, 20)
            if action == "closed":
                closed_at = created_iso
        elif evt in {
            "IssueCommentEvent",
            "PullRequestReviewCommentEvent",
            "CommitCommentEvent",
        }:
            action = "created"
            # Comments occasionally name another repo from the pool so
            # "comments mentioning X" / body-LIKE queries are non-trivial.
            if rng.random() < 0.35:
                mentioned = rng.choice(REPO_POOL)
                body = f"see {mentioned} for context, cc @{rng.choice(ACTOR_POOL)}"
            else:
                body = "auto-generated comment"
            number = rng.randint(1, 50_000)
            if evt == "CommitCommentEvent":
                commit_id = rng.choice(commit_pool[repo])
        elif evt == "PullRequestReviewEvent":
            action = "created"
            review_state = rng.choice(REVIEW_STATES)
            number = rng.randint(1, 50_000)
        elif evt == "CreateEvent":
            action = "created"
        elif evt == "DeleteEvent":
            action = "deleted"
        elif evt == "ReleaseEvent":
            action = "published"
            release_tag = f"v{rng.randint(0, 5)}.{rng.randint(0, 20)}.{rng.randint(0, 50)}"
            release_name = release_tag
        elif evt == "ForkEvent":
            action = "fork"
        elif evt == "WatchEvent":
            action = "started"
        elif evt == "MemberEvent":
            action = "added"

        rows.append(
            {
                "file_time": created_iso,
                "event_type": evt,
                "actor_login": actor,
                "repo_name": repo,
                "created_at": created_iso,
                "updated_at": created_iso,
                "action": action,
                "comment_id": (
                    rng.randint(0, 10_000_000)
                    if evt.endswith("CommentEvent")
                    else 0
                ),
                "commit_id": commit_id,
                "body": body,
                "ref": ref,
                "number": number,
                "title": title,
                "labels": labels,
                "state": state,
                "locked": 0,
                "assignee": "",
                "comments": rng.randint(0, 20) if number else 0,
                "author_association": rng.choice(AUTHOR_ASSOC),
                "closed_at": closed_at,
                "merged_at": merged_at,
                "merged": merged,
                "commits": commits,
                "additions": additions,
                "deletions": deletions,
                "changed_files": changed_files,
                "push_size": push_size,
                "release_tag_name": release_tag,
                "release_name": release_name,
                "review_state": review_state,
            }
        )
    return rows


__all__ = ["SCHEMA_DDL", "SCHEMA_PROMPT", "seed_rows"]
