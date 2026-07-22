"""Shared Arrow-schema helpers for the scikit-bio worker.

Keeps column-comment plumbing and name sanitisation in one place so every
function exposes consistent, documented schemas to DuckDB.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import pyarrow as pa

# One declared result column: (name, SQL type, description).
ResultColumn = tuple[str, str, str]


def field(
    name: str,
    type: pa.DataType,
    comment: str,
    *,
    nullable: bool = True,
) -> pa.Field:
    """Build a ``pa.Field`` carrying a column comment in its metadata.

    The ``comment`` metadata key is the framework's transport for column
    comments -- DuckDB surfaces it via ``duckdb_columns()`` and ``DESCRIBE``.
    """
    return pa.field(
        name,
        type,
        nullable=nullable,
        metadata={b"comment": comment.encode("utf-8")},
    )


def result_columns_schema(rows: list[ResultColumn]) -> str:
    """Render the ``vgi.result_columns_schema`` tag for a **fixed** result schema.

    DuckDB cannot expose a VGI table function's RETURN columns through its own
    system tables, so the worker declares them itself. The tag is a JSON array of
    ``{name, type, description}`` objects, in output order — machine-readable, so
    a client (or the metadata linter) can check it against what the function
    actually returns. Use ``result_dynamic_columns_md`` instead when the columns
    depend on an argument.
    """
    return json.dumps([{"name": n, "type": t, "description": d} for n, t, d in rows])


def result_dynamic_columns_md(
    variants: list[tuple[str, list[ResultColumn]]],
    *,
    note: str | None = None,
) -> str:
    """Render the ``vgi.result_dynamic_columns_md`` tag for an **argument-dependent** schema.

    Some table functions here change shape with their arguments -- ``id :=`` adds
    a carried id column, ``n_components :=`` adds an axis column per component --
    so no single static schema is truthful. This renders one
    ``Name | Type | Description`` Markdown table per variant, captioned with the
    argument setting that selects it.
    """
    blocks: list[str] = []
    for caption, rows in variants:
        lines = [f"#### {caption}", "", "| Name | Type | Description |", "| --- | --- | --- |"]
        lines += [f"| `{n}` | {t} | {d} |" for n, t, d in rows]
        blocks.append("\n".join(lines))
    md = "\n\n".join(blocks)
    if note:
        md += f"\n\n{note}"
    return md


_NON_IDENT = re.compile(r"[^0-9a-z]+")


def snake_case(name: str) -> str:
    """Normalise a label to a SQL-friendly column name.

    ``"PC 1"`` -> ``"pc_1"``. Collapses any run of non-alphanumeric characters
    to a single underscore and lowercases.
    """
    cleaned = _NON_IDENT.sub("_", name.strip().lower()).strip("_")
    if not cleaned:
        return "col"
    if cleaned[0].isdigit():
        cleaned = f"c_{cleaned}"
    return cleaned


def dedupe_names(names: list[str]) -> list[str]:
    """Ensure column names are unique by suffixing collisions (``_2``, ``_3`` ...)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
    return out


@dataclass(slots=True, frozen=True, kw_only=True)
class NoArgs:
    """Empty argument set for functions that take no user-facing parameters."""
