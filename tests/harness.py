"""In-process helpers shared across the worker test suite.

Drive an aggregate function through its ``initial_state`` -> ``update`` ->
``finalize`` lifecycle without spawning a worker process, so tests stay fast and
debuggable. Table/buffering functions are unit-tested via their ``encode``
classmethods; their full lifecycle is covered by the SQL suite.
"""

from __future__ import annotations

import pyarrow as pa


def run_alpha(
    func_cls: type,
    counts: list[float],
    *,
    group_ids: list[int] | None = None,
) -> dict[int, float | None]:
    """Drive an alpha-diversity AggregateFunction over a single count column.

    Returns a ``{group_id: result}`` mapping. With no ``group_ids`` all rows
    fall in group 0, so ``run_alpha(...)[0]`` is the single scalar result.
    """
    n = len(counts)
    gids = group_ids if group_ids is not None else [0] * n
    states = {g: func_cls.initial_state(None) for g in set(gids)}
    func_cls.update(
        states,
        pa.array(gids, type=pa.int64()),
        pa.array(counts, type=pa.float64()),
    )
    ordered = sorted(states)
    batch = func_cls.finalize(pa.array(ordered, type=pa.int64()), states, None)
    return dict(zip(ordered, batch.column("result").to_pylist(), strict=False))
