"""Shared plumbing for table-buffering functions.

Buffering functions (distance matrices, ordination, compositional transforms,
tree construction) all need the whole input before producing output. The sink
phase serializes each input batch to execution-scoped storage; finalize
reassembles the full table. This module holds the serialization, storage, and
assembly helpers plus the single-bucket sink/combine implementation so each
function only writes its finalize logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from vgi.table_buffering_function import TableBufferingFunction, TableBufferingParams
from vgi_rpc import ArrowSerializableDataclass

_DATA_KEY = b"input_batches"


@dataclass(kw_only=True)
class DrainState(ArrowSerializableDataclass):
    """Per-finalize-stream cursor: emit the result once, then finish."""

    done: bool = False


def serialize_batch(batch: pa.RecordBatch) -> bytes:
    """Serialize one record batch to the Arrow IPC stream format."""
    sink = pa.BufferOutputStream()
    # pa.ipc.new_stream is untyped in pyarrow's partial stubs.
    with pa.ipc.new_stream(sink, batch.schema) as writer:  # type: ignore[no-untyped-call]
        writer.write_batch(batch)
    return bytes(sink.getvalue().to_pybytes())


def deserialize_batches(value: bytes) -> list[pa.RecordBatch]:
    """Read back the record batches from an Arrow IPC stream blob."""
    # pa.ipc.open_stream is untyped in pyarrow's partial stubs.
    reader = pa.ipc.open_stream(pa.BufferReader(value))  # type: ignore[no-untyped-call]
    batches: list[pa.RecordBatch] = reader.read_all().to_batches()
    return batches


class SinkBuffer[TArgs, TState](TableBufferingFunction[TArgs, TState]):
    """Single-bucket sink/combine: buffer every input batch under one key.

    Subclasses implement ``on_bind``, ``initial_finalize_state``, and
    ``finalize`` (calling ``buffered_table(params)`` to get the full input).
    """

    @classmethod
    def process(cls, batch: pa.RecordBatch, params: TableBufferingParams[TArgs]) -> bytes:
        """Append the (non-empty) input batch to the single buffer bucket."""
        if batch.num_rows:
            params.storage.state_append(_DATA_KEY, b"", serialize_batch(batch))
        return params.execution_id

    @classmethod
    def combine(cls, state_ids: list[bytes], params: TableBufferingParams[TArgs]) -> list[bytes]:
        """Collapse partial state ids back to the single execution id."""
        return [params.execution_id]

    @classmethod
    def buffered_table(cls, params: TableBufferingParams[TArgs], input_schema: pa.Schema) -> pa.Table | None:
        """Reassemble every buffered batch into the full input table, or None if empty."""
        batches: list[pa.RecordBatch] = []
        for _sid, value in params.storage.state_log_scan(_DATA_KEY, b""):
            batches.extend(deserialize_batches(value))
        if not batches:
            return None
        return pa.Table.from_batches(batches, schema=input_schema)


def input_schema_of(params: Any) -> pa.Schema:
    """Input schema from a process/finalize params object."""
    schema = params.init_call.bind_call.input_schema
    assert schema is not None
    return schema
