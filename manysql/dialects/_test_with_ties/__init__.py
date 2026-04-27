"""Synthetic test dialect: exercises the Plan-rewrite passes lane.

This dialect's lowering would, for a SQL like

    SELECT * FROM t ORDER BY x LIMIT 2 WITH TIES

emit a non-canonical IR marker:

    Filter(
      input=Sort(input=Scan(t), keys=[OrderKey(x, ASC)]),
      predicate=FuncCall("__manysql_with_ties", [Literal(2)])
    )

`passes.py` rewrites that marker into canonical IR (Window(rank) +
Filter(rank <= n) + Project) the executor already understands.

The dialect is intentionally minimal: no real grammar, no Lark parsing.
Its purpose is to validate that `PRE_EXECUTION_PASSES` is loaded and
applied between lowering and execution. See
`tests/test_dialect_passes.py`.
"""
