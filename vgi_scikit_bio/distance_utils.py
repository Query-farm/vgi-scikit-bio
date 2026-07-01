"""Reconstruct a scikit-bio ``DistanceMatrix`` from a long ``(id_1, id_2, distance)`` table.

The worker's distance functions (``beta_diversity``) emit a distance matrix in
long form. The consumers (``pcoa``, ``permanova``, ``anosim``, ``mantel``) read
it back with this helper, which is tolerant of both a full square (n^2 rows) and
a condensed upper/lower triangle: it symmetrizes and zero-fills the diagonal.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from skbio import DistanceMatrix


def resolve_pair_columns(schema: pa.Schema, id1: str, id2: str, dist: str) -> tuple[str, str, str]:
    """Resolve the (id_1, id_2, distance) column names, defaulting to positional 0/1/2."""
    names = list(schema.names)
    a = id1 or (names[0] if len(names) > 0 else "")
    b = id2 or (names[1] if len(names) > 1 else "")
    d = dist or (names[2] if len(names) > 2 else "")
    for label, col in (("id_1", a), ("id_2", b), ("distance", d)):
        if not col or col not in names:
            raise ValueError(f"{label} column {col!r} not found in input; columns: {', '.join(names)}")
    return a, b, d


def distance_matrix_from_long(
    table: pa.Table,
    id1_col: str,
    id2_col: str,
    dist_col: str,
) -> DistanceMatrix:
    """Build a symmetric ``DistanceMatrix`` from long ``(id_1, id_2, distance)`` rows.

    Ids are ordered by first appearance. Both a full square and a condensed
    triangle are accepted; the matrix is symmetrized (``d[j, i] = d[i, j]``) and
    the diagonal is forced to zero.
    """
    id1 = [str(v) for v in table.column(id1_col).to_pylist()]
    id2 = [str(v) for v in table.column(id2_col).to_pylist()]
    dist = table.column(dist_col).to_pylist()

    ids: list[str] = []
    index: dict[str, int] = {}
    for name in (*id1, *id2):
        if name not in index:
            index[name] = len(ids)
            ids.append(name)

    n = len(ids)
    if n < 2:
        raise ValueError("a distance matrix needs at least two distinct ids")
    data = np.zeros((n, n), dtype=np.float64)
    for a, b, d in zip(id1, id2, dist, strict=True):
        if d is None:
            continue
        i, j = index[a], index[b]
        if i == j:
            continue
        data[i, j] = float(d)
        data[j, i] = float(d)
    return DistanceMatrix(data, ids=ids)
