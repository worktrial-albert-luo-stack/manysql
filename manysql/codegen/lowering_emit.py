"""Emit lowering.py for a generated dialect.

Key observation: as long as the dialect keeps the reference grammar's *rule
names* (we only swap literal keywords/operators), the lowering logic is
identical — the parse tree shape is the same.

For specs that change structural rules (e.g. JoinSyntax.PIPELINED,
CaseSyntax.SWITCH), the lowering needs targeted rewrites. The codegen
refine loop hands those cases off to the LLM with the deterministic
baseline as the starting prompt.
"""

from __future__ import annotations

from importlib import resources

from manysql.spec.dialect import (
    CaseSyntax,
    DialectSpec,
    JoinSyntax,
    LimitSyntax,
)


_REFERENCE_LOWERING_HEADER = '"""Reference-dialect lowering'
_GENERATED_HEADER_TEMPLATE = '"""Generated lowering for the {name} dialect'


def emit_lowering(spec: DialectSpec) -> str:
    """Return the lowering.py source for the dialect.

    The deterministic path returns the reference lowering with the docstring
    swapped to mention the new dialect, plus targeted patches for surface
    knobs whose grammar shape diverges (e.g. LimitSyntax.OFFSET_FETCH puts
    offset before limit in the parse tree). Structural divergences that
    can't be patched mechanically raise so the pipeline knows to fall back
    to the LLM lane.
    """
    if _requires_structural_changes(spec):
        raise NotImplementedError(
            f"Spec {spec.name!r} requires structural lowering changes "
            f"(join_syntax={spec.surface.join_syntax.value}, "
            f"case_syntax={spec.surface.case_syntax.value}). "
            "Use the LLM-refined emitter."
        )

    base = _read_reference_lowering()
    new_header = _GENERATED_HEADER_TEMPLATE.format(name=spec.name)
    out = base.replace(_REFERENCE_LOWERING_HEADER, new_header, 1)
    out = _patch_limit_lowering(out, spec)
    return out


def _read_reference_lowering() -> str:
    return resources.read_text(
        "manysql.dialects._reference", "lowering.py", encoding="utf-8"
    )


def _requires_structural_changes(spec: DialectSpec) -> bool:
    return (
        spec.surface.join_syntax != JoinSyntax.ANSI
        or spec.surface.case_syntax != CaseSyntax.CASE_WHEN
    )


_REFERENCE_LIMIT_BODY = (
    "    def _lower_limit(self, plan: Plan, node: Tree) -> Plan:\n"
    "        ints = [int(t) for t in node.children if isinstance(t, Token)]\n"
    "        limit = ints[0]\n"
    "        offset = ints[1] if len(ints) > 1 else 0\n"
    "        return Limit(input=plan, limit=limit, offset=offset)"
)


def _patch_limit_lowering(text: str, spec: DialectSpec) -> str:
    """Patch `_lower_limit` for limit syntaxes whose tree shape differs from
    the reference's `INT ("OFFSET"i INT)?`.
    """
    syntax = spec.surface.limit_syntax
    if syntax in (LimitSyntax.LIMIT_OFFSET, LimitSyntax.TOP_N):
        return text
    if syntax == LimitSyntax.OFFSET_FETCH:
        new_body = (
            "    def _lower_limit(self, plan: Plan, node: Tree) -> Plan:\n"
            "        # OFFSET m ROWS FETCH NEXT n ROWS ONLY -> offset first, then limit\n"
            "        ints = [int(t) for t in node.children if isinstance(t, Token)]\n"
            "        offset = ints[0]\n"
            "        limit = ints[1]\n"
            "        return Limit(input=plan, limit=limit, offset=offset)"
        )
    elif syntax in (LimitSyntax.SAMPLE_N, LimitSyntax.HEAD_N):
        new_body = (
            "    def _lower_limit(self, plan: Plan, node: Tree) -> Plan:\n"
            "        ints = [int(t) for t in node.children if isinstance(t, Token)]\n"
            "        return Limit(input=plan, limit=ints[0], offset=0)"
        )
    else:  # pragma: no cover - exhaustive
        return text
    return text.replace(_REFERENCE_LIMIT_BODY, new_body, 1)
