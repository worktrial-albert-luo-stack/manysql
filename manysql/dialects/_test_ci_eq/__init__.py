"""Synthetic test dialect: exercises the executor effects lane.

`effects.py` installs a `text_eq` handler that lowercases both operands
before comparing them, simulating a collation-insensitive default
(roughly T-SQL's `Latin1_General_CI_AS`). The plan IR is unchanged and
the handler does not require any dialect-specific markers — only the
implementation of the `=` decision point swaps.

The dialect ships no real grammar or lowering; tests construct IR
plans directly and execute them with `effects=engine.effects`. See
`tests/test_dialect_effects.py`.
"""
