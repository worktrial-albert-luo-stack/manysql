"""``python -m train.env``: smoke-test the SQL RL environment end-to-end.

Drives a single episode against a single task using a :class:`FixedSqlPolicy`
(by default replaying the gold SQL, which should match on turn 1) or a
caller-supplied SQL string. Useful for:

* Verifying a freshly-codegened dialect's runtime works in the env.
* Sanity-checking gold-row alignment between a generator's reference
  engine and the candidate dialect's runtime.
* Hand-driving the env to debug reward shaping.

Examples::

    # Replay the first golden task on the aggressive_alien dialect using its
    # canonical SQL; should match on turn 1 and earn the full turn bonus.
    python -m train.env --dialect aggressive_alien --generator golden \\
        --policy gold --task-index 0

    # Try a hand-written candidate query.
    python -m train.env --dialect aggressive_alien --generator golden \\
        --policy fixed --sql "SELECT * FROM employees" --task-index 0

    # Hit the eval-suite NL benchmark instead.
    python -m train.env --dialect mild_postgres_ish --generator eval \\
        --task-name e1_count_total_rows --policy gold
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from train.env import (
    DialectRuntime,
    EvalSuiteTaskConfig,
    EvalSuiteTaskGenerator,
    FixedSqlPolicy,
    GoldenTaskConfig,
    GoldenTaskGenerator,
    RewardConfig,
    SqlEnv,
    SqlTask,
    TaskGenerator,
    run_episode,
)


def _build_generator(args: argparse.Namespace) -> TaskGenerator:
    if args.generator == "golden":
        return GoldenTaskGenerator(
            GoldenTaskConfig(
                target_dialect=args.dialect,
                reference_dialect=args.reference_dialect,
            )
        )
    if args.generator == "eval":
        return EvalSuiteTaskGenerator(
            EvalSuiteTaskConfig(target_dialect=args.dialect)
        )
    raise ValueError(f"unknown generator {args.generator!r}")


def _select_task(gen: TaskGenerator, args: argparse.Namespace) -> SqlTask:
    tasks = gen.all_tasks()
    if not tasks:
        sys.exit(f"no tasks available from {gen.name!r} for dialect {args.dialect!r}")
    if args.task_name is not None:
        for t in tasks:
            if t.meta.task_id == args.task_name:
                return t
        ids = [t.meta.task_id for t in tasks][:10]
        sys.exit(f"no task named {args.task_name!r}; have {ids}...")
    if args.task_index >= len(tasks):
        sys.exit(f"task-index {args.task_index} out of range (have {len(tasks)} tasks)")
    return tasks[args.task_index]


def _build_policy(task: SqlTask, args: argparse.Namespace) -> FixedSqlPolicy:
    if args.policy == "gold":
        # Replay the canonical SQL. For golden tasks this should match the
        # gold rows on turn 1 (same dialect engine, same data, same SQL).
        # For eval tasks the reference is SQLite-flavored and may not parse
        # in the candidate dialect -- expect failure there.
        return FixedSqlPolicy(task.gold_sql)
    if args.policy == "fixed":
        if not args.sql:
            sys.exit("--policy fixed requires --sql")
        return FixedSqlPolicy(args.sql)
    if args.policy == "sequence":
        if not args.sql:
            sys.exit("--policy sequence requires --sql (use ;; to separate)")
        seq = [s.strip() for s in args.sql.split(";;")]
        return FixedSqlPolicy(sequence=seq)
    raise ValueError(f"unknown policy {args.policy!r}")


def _print_summary(result: dict[str, Any]) -> None:
    ep = result["episode"]
    print(f"task           : {ep['task']['task_id']} ({ep['task']['generator']})")
    print(f"dialect        : {ep['task']['dialect']}")
    print(f"matched        : {ep['matched']}")
    print(f"truncated      : {ep['truncated']}")
    print(f"turns_used     : {len(ep['turns'])}")
    print(f"reward         : {ep['reward']:.3f}")
    print("reward_components:")
    for k, v in ep["reward_components"].items():
        print(f"  {k:<14} {v:.3f}")
    print("turns:")
    for t in ep["turns"]:
        sql = t["sql"].replace("\n", " ")
        if len(sql) > 80:
            sql = sql[:77] + "..."
        outcome = (
            "MATCH"
            if t["matched"]
            else (t["error_class"] or "no-match").upper()
        )
        print(f"  [{t['index']}] {outcome:<10} {sql}")
        if not t["matched"] and not t["exec_result"]["success"]:
            err = (t["exec_result"].get("error") or "").splitlines()[0]
            print(f"        err: {err[:100]}")


def main() -> int:
    p = argparse.ArgumentParser(prog="train.env", description=__doc__)
    p.add_argument("--dialect", required=True, help="target dialect name (e.g. aggressive_alien)")
    p.add_argument(
        "--generator",
        choices=["golden", "eval"],
        default="golden",
        help="task generator: golden = SQL-translation, eval = NL->SQL",
    )
    p.add_argument(
        "--reference-dialect",
        default="_reference",
        help="reference dialect for golden gold-row materialization",
    )
    p.add_argument("--task-name", default=None, help="select a specific task by id")
    p.add_argument("--task-index", type=int, default=0, help="select a task by 0-based index")
    p.add_argument(
        "--policy",
        choices=["gold", "fixed", "sequence"],
        default="gold",
        help=(
            "gold = replay gold_sql; fixed = single SQL via --sql; "
            "sequence = ;;-separated SQL list"
        ),
    )
    p.add_argument("--sql", default=None, help="SQL for --policy fixed/sequence")
    p.add_argument("--max-turns", type=int, default=3)
    p.add_argument("--json", action="store_true", help="dump full RolloutResult as JSON")
    args = p.parse_args()

    gen = _build_generator(args)
    gen.build()
    task = _select_task(gen, args)

    runtime = DialectRuntime(dialect=args.dialect, catalog=task.catalog)
    runtime.setup()
    try:
        env = SqlEnv(
            task=task,
            runtime=runtime,
            max_turns=args.max_turns,
            reward_config=RewardConfig(),
        )
        policy = _build_policy(task, args)
        result = run_episode(env=env, policy=policy)
    finally:
        runtime.teardown()

    payload = result.to_dict()
    if args.json:
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        _print_summary(payload)
    return 0 if result.episode.matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
