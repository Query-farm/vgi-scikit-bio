"""Compositional data transforms: centred (CLR) and isometric (ILR) log-ratios.

Compositional data (e.g. relative abundances) carry only relative information, so
these transforms move it into ordinary real space where standard statistics
apply. Both read a long feature table ``(sample_id, feature_id, value)`` and emit
long:

* ``clr`` -- centred log-ratio: one value per (sample, feature), same shape.
* ``ilr`` -- isometric log-ratio: ``D-1`` orthonormal components per sample,
  emitted as ``(sample_id, component, value)``.

Values must be positive; pass ``pseudocount :=`` to replace zeros before the
transform (the log of zero is undefined).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .schema_utils import columns_md_rows
from .schema_utils import field as sfield


@dataclass(slots=True, frozen=True)
class _CompArgs:
    data: Annotated[TableInput, Arg(0, doc="One row per (sample_id, feature_id, value).")]
    sample: Annotated[str, Arg("sample", default="", doc="Sample-id column (defaults to the first column).")]
    feature: Annotated[str, Arg("feature", default="", doc="Feature-id column (defaults to the second column).")]
    value: Annotated[str, Arg("value", default="", doc="Value column (defaults to the third column).")]
    pseudocount: Annotated[
        float, Arg("pseudocount", default=0.0, doc="Added to every value before the transform (to handle zeros).")
    ]


def _resolve_triple(schema: pa.Schema, sample: str, feature: str, value: str) -> tuple[str, str, str]:
    """Resolve the (sample, feature, value) column names, defaulting to positional 0/1/2."""
    names = list(schema.names)
    s = sample or (names[0] if len(names) > 0 else "")
    f = feature or (names[1] if len(names) > 1 else "")
    v = value or (names[2] if len(names) > 2 else "")
    for label, col in (("sample", s), ("feature", f), ("value", v)):
        if not col or col not in names:
            raise ValueError(f"{label} column {col!r} not found in input; columns: {', '.join(names)}")
    return s, f, v


def _pivot(table: pa.Table, s_col: str, f_col: str, v_col: str, pseudocount: float) -> tuple[list[str], list[str], Any]:
    """Pivot a long feature table to (sample_ids, feature_ids, dense matrix + pseudocount)."""
    samples = [str(v) for v in table.column(s_col).to_pylist()]
    features = [str(v) for v in table.column(f_col).to_pylist()]
    values = table.column(v_col).to_pylist()
    sample_ids = sorted(set(samples))
    feature_ids = sorted(set(features))
    s_index = {s: i for i, s in enumerate(sample_ids)}
    f_index = {f: j for j, f in enumerate(feature_ids)}
    matrix = np.zeros((len(sample_ids), len(feature_ids)), dtype=np.float64)
    for s, f, v in zip(samples, features, values, strict=True):
        if v is None:
            continue
        matrix[s_index[s], f_index[f]] += float(v)
    matrix = matrix + float(pseudocount)
    return sample_ids, feature_ids, matrix


class Clr(SinkBuffer[_CompArgs, DrainState]):
    """Centred log-ratio transform of a long feature table (same long shape)."""

    FunctionArguments: ClassVar[type] = _CompArgs

    class Meta:
        """VGI metadata for the clr function."""

        name = "clr"
        description = "Centred log-ratio (CLR) transform of a long feature table"
        categories = ["stats", "composition"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM skbio.stats.clr((SELECT * FROM "
                    "(VALUES ('s1','a',1),('s1','b',2),('s1','c',3),('s2','a',4),('s2','b',5),('s2','c',6)) "
                    "AS t(sample_id, feature_id, value)))"
                ),
                description="CLR-transform two 3-part compositions",
            )
        ]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    ("sample_id", "VARCHAR", "Sample id."),
                    ("feature_id", "VARCHAR", "Feature id."),
                    ("clr", "DOUBLE", "Centred log-ratio value."),
                ]
            ),
            "vgi.doc_llm": (
                "Table function applying the centred log-ratio (CLR) transform to each sample of a long "
                "feature table, returning the same long shape. The table arg is "
                "`(SELECT sample_id, feature_id, value FROM ...)` (columns default to positional 1/2/3; "
                "override with `sample :=`, `feature :=`, `value :=`). For each sample the values are "
                "closed to proportions and replaced by the log of each part divided by the geometric mean "
                "of the sample — moving compositional data into real space where ordinary statistics "
                "apply. Values must be positive; pass `pseudocount :=` (e.g. 1) to offset zeros before the "
                "log. Returns `(sample_id, feature_id, clr)`."
            ),
            "vgi.doc_md": (
                "**CLR** — centred log-ratio transform of compositional data.\n\n"
                "- Table arg: `(SELECT sample_id, feature_id, value FROM ...)` (positional 1/2/3 by "
                "default)\n"
                "- Per sample: close to proportions, then `log(part / geometric_mean)`\n"
                "- `pseudocount :=` — added to every value first to handle zeros (log 0 is undefined)\n"
                "- Returns `(sample_id, feature_id, clr)` — same long shape as the input"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[_CompArgs]) -> BindResponse:
        """Validate columns and fix the (sample_id, feature_id, clr) output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        _resolve_triple(input_schema, a.sample, a.feature, a.value)
        fields = [
            sfield("sample_id", pa.string(), "Sample id.", nullable=False),
            sfield("feature_id", pa.string(), "Feature id.", nullable=False),
            sfield("clr", pa.float64(), "Centred log-ratio value.", nullable=False),
        ]
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_CompArgs]) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_CompArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Pivot, CLR-transform per sample, and emit long, once."""
        if state.done:
            out.finish()
            return
        state.done = True
        input_schema = input_schema_of(params)
        out_schema = params.output_schema
        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            out.emit(pa.RecordBatch.from_pydict({n: [] for n in out_schema.names}, schema=out_schema))
            return
        out.emit(pa.RecordBatch.from_pydict(cls.encode(table, params.args), schema=out_schema))

    @classmethod
    def encode(cls, table: pa.Table, args: _CompArgs) -> dict[str, list[Any]]:
        """CLR-transform each sample and return the long-format columns."""
        from skbio.stats.composition import closure, clr

        s_col, f_col, v_col = _resolve_triple(table.schema, args.sample, args.feature, args.value)
        sample_ids, feature_ids, matrix = _pivot(table, s_col, f_col, v_col, args.pseudocount)
        transformed = clr(closure(matrix))
        s_out: list[str] = []
        f_out: list[str] = []
        v_out: list[float] = []
        for i, sid in enumerate(sample_ids):
            for j, fid in enumerate(feature_ids):
                s_out.append(sid)
                f_out.append(fid)
                v_out.append(float(transformed[i, j]))
        return {"sample_id": s_out, "feature_id": f_out, "clr": v_out}


class Ilr(SinkBuffer[_CompArgs, DrainState]):
    """Isometric log-ratio transform: D-1 orthonormal components per sample."""

    FunctionArguments: ClassVar[type] = _CompArgs

    class Meta:
        """VGI metadata for the ilr function."""

        name = "ilr"
        description = "Isometric log-ratio (ILR) transform of a long feature table (D-1 components)"
        categories = ["stats", "composition"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM skbio.stats.ilr((SELECT * FROM "
                    "(VALUES ('s1','a',1),('s1','b',2),('s1','c',3),('s2','a',4),('s2','b',5),('s2','c',6)) "
                    "AS t(sample_id, feature_id, value)))"
                ),
                description="ILR-transform two 3-part compositions (2 components each)",
            )
        ]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    ("sample_id", "VARCHAR", "Sample id."),
                    ("component", "BIGINT", "ILR component index (1..D-1)."),
                    ("value", "DOUBLE", "Isometric log-ratio value on that component."),
                ]
            ),
            "vgi.doc_llm": (
                "Table function applying the isometric log-ratio (ILR) transform to each sample of a long "
                "feature table. Unlike CLR, ILR maps a D-part composition onto `D-1` orthonormal "
                "coordinates (no singular covariance), so it emits `(sample_id, component, value)` with "
                "`component` in `1..D-1`. The table arg is `(SELECT sample_id, feature_id, value FROM ...)` "
                "(columns default to positional 1/2/3; override with `sample :=`, `feature :=`, "
                "`value :=`). Values must be positive; pass `pseudocount :=` to offset zeros. Feature ids "
                "are sorted to give the components a stable basis across samples."
            ),
            "vgi.doc_md": (
                "**ILR** — isometric log-ratio transform of compositional data.\n\n"
                "- Table arg: `(SELECT sample_id, feature_id, value FROM ...)` (positional 1/2/3 by "
                "default)\n"
                "- Maps a D-part composition to `D-1` orthonormal coordinates (full-rank, unlike CLR)\n"
                "- `pseudocount :=` — added to every value first to handle zeros\n"
                "- Returns `(sample_id, component, value)` with `component` in `1..D-1`; features sorted "
                "for a stable basis"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[_CompArgs]) -> BindResponse:
        """Validate columns and fix the (sample_id, component, value) output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        _resolve_triple(input_schema, a.sample, a.feature, a.value)
        fields = [
            sfield("sample_id", pa.string(), "Sample id.", nullable=False),
            sfield("component", pa.int64(), "ILR component index (1..D-1).", nullable=False),
            sfield("value", pa.float64(), "Isometric log-ratio value on that component.", nullable=False),
        ]
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_CompArgs]) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_CompArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Pivot, ILR-transform per sample, and emit long, once."""
        if state.done:
            out.finish()
            return
        state.done = True
        input_schema = input_schema_of(params)
        out_schema = params.output_schema
        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            out.emit(pa.RecordBatch.from_pydict({n: [] for n in out_schema.names}, schema=out_schema))
            return
        out.emit(pa.RecordBatch.from_pydict(cls.encode(table, params.args), schema=out_schema))

    @classmethod
    def encode(cls, table: pa.Table, args: _CompArgs) -> dict[str, list[Any]]:
        """ILR-transform each sample and return the long-format columns."""
        from skbio.stats.composition import closure, ilr

        s_col, f_col, v_col = _resolve_triple(table.schema, args.sample, args.feature, args.value)
        sample_ids, feature_ids, matrix = _pivot(table, s_col, f_col, v_col, args.pseudocount)
        if len(feature_ids) < 2:
            raise ValueError("ILR needs at least two features")
        transformed = ilr(closure(matrix))
        s_out: list[str] = []
        comp_out: list[int] = []
        v_out: list[float] = []
        for i, sid in enumerate(sample_ids):
            for k in range(transformed.shape[1]):
                s_out.append(sid)
                comp_out.append(k + 1)
                v_out.append(float(transformed[i, k]))
        return {"sample_id": s_out, "component": comp_out, "value": v_out}


COMPOSITION_FUNCTIONS: list[type] = [Clr, Ilr]
