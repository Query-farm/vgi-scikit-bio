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
from .schema_utils import field as sfield
from .schema_utils import result_columns_schema


@dataclass(slots=True, frozen=True)
class _CompArgs:
    data: Annotated[TableInput, Arg(0, doc="One row per (sample_id, feature_id, value).")]
    sample: Annotated[str, Arg("sample", default="", doc="Sample-id column (defaults to the first column).")]
    feature: Annotated[str, Arg("feature", default="", doc="Feature-id column (defaults to the second column).")]
    value: Annotated[str, Arg("value", default="", doc="Value column (defaults to the third column).")]
    pseudocount: Annotated[
        float, Arg("pseudocount", default=0.0, doc="Added to every value before the transform (to handle zeros).")
    ]


# The shared two-sample, three-part feature table every compositional example
# transforms. s1 and s2 differ in total (6 vs 15) but the transforms only see
# their relative parts -- which is the point these examples are there to make.
_COMP_INPUT = (
    "(SELECT * FROM "
    "(VALUES ('s1','a',1),('s1','b',2),('s1','c',3),('s2','a',4),('s2','b',5),('s2','c',6)) "
    "AS t(sample_id, feature_id, value))"
)


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
                    "SELECT sample_id, feature_id, round(clr, 4) AS clr "
                    f"FROM skbio.stats.clr({_COMP_INPUT}) ORDER BY sample_id, feature_id"
                ),
                description=(
                    "Move a feature table out of the simplex and into real space, where ordinary "
                    "statistics (correlation, regression, distance) are meaningful again. A "
                    "negative clr value means the feature is below its sample's geometric mean, "
                    "positive above — the sign, not the raw count, is what carries the signal."
                ),
            )
        ]
        tags = {
            "vgi.result_columns_schema": result_columns_schema(
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
                    "SELECT sample_id, component, round(value, 4) AS value "
                    f"FROM skbio.stats.ilr({_COMP_INPUT}) ORDER BY sample_id, component"
                ),
                description=(
                    "Turn 3-part compositions into 2 orthonormal coordinates — note the count "
                    "drops from D to D-1, which is the point: unlike clr, the ilr coordinates are "
                    "not linearly dependent, so a covariance matrix built from them is invertible "
                    "and methods like PCA or LDA behave."
                ),
            )
        ]
        tags = {
            "vgi.result_columns_schema": result_columns_schema(
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


# ===========================================================================
# Same-shape transforms (long (sample, feature, value) -> long, same shape)
# ===========================================================================


def _comp_doc(name: str, blurb: str, cols: list[tuple[str, str, str]]) -> dict[str, str]:
    """Build the tags for a compositional function returning a long table."""
    return {
        "vgi.category": "composition",
        "vgi.result_columns_schema": result_columns_schema(cols),
        "vgi.doc_llm": (
            f"Table function applying the **{name}** compositional operation to a long feature table "
            f"`(SELECT sample_id, feature_id, value FROM ...)` (columns default to positional 1/2/3; "
            f"override with `sample :=`, `feature :=`, `value :=`; `pseudocount :=` offsets zeros). {blurb}"
        ),
        "vgi.doc_md": (
            f"**{name}** — {blurb}\n\n"
            "- Table arg: `(SELECT sample_id, feature_id, value FROM ...)` (positional 1/2/3 by default)\n"
            "- `pseudocount :=` — added to every value first to handle zeros"
        ),
    }


def _comp_example(name: str, key_col: str, out_col: str, description: str) -> list[FunctionExample]:
    """One projected, ordered example for a compositional table function.

    Args:
        name: The function's machine name (``skbio.stats.<name>``).
        key_col: The second output column (``feature_id`` or the index column).
        out_col: The value column the transform emits.
        description: What the example teaches -- not a restatement of the SQL.

    Returns:
        A one-entry example list for the function's ``Meta.examples``.
    """
    return [
        FunctionExample(
            sql=(
                f"SELECT sample_id, {key_col}, round({out_col}, 4) AS {out_col} "
                f"FROM skbio.stats.{name}({_COMP_INPUT}) ORDER BY sample_id, {key_col}"
            ),
            description=description,
        )
    ]


class _SameShape(SinkBuffer[_CompArgs, DrainState]):
    """Buffer a long feature table, apply a same-shape transform, emit long."""

    FunctionArguments: ClassVar[type] = _CompArgs
    TRANSFORM: ClassVar[Any]
    OUT_COL: ClassVar[str] = "value"

    @classmethod
    def _apply(cls, matrix: Any, args: _CompArgs) -> Any:
        """Apply the transform to the pivoted matrix (overridable for extra args)."""
        return cls.TRANSFORM(matrix)

    @classmethod
    def on_bind(cls, params: BindParams[_CompArgs]) -> BindResponse:
        """Validate columns and fix the (sample_id, feature_id, <out>) output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        _resolve_triple(input_schema, a.sample, a.feature, a.value)
        fields = [
            sfield("sample_id", pa.string(), "Sample id.", nullable=False),
            sfield("feature_id", pa.string(), "Feature id.", nullable=False),
            sfield(cls.OUT_COL, pa.float64(), "Transformed value.", nullable=True),
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
        """Pivot, transform per sample, and emit long, once."""
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
        """Transform each sample and return the long-format columns."""
        s_col, f_col, v_col = _resolve_triple(table.schema, args.sample, args.feature, args.value)
        sample_ids, feature_ids, matrix = _pivot(table, s_col, f_col, v_col, args.pseudocount)
        # atleast_2d: several scikit-bio composition transforms (power,
        # multi_replace) squeeze a single-sample (1, D) matrix down to (D,), which
        # would make the [i, j] indexing below raise IndexError on a one-sample
        # input. Restore the (samples, features) shape before indexing.
        transformed = np.atleast_2d(np.asarray(cls._apply(matrix, args), dtype=np.float64))
        s_out: list[str] = []
        f_out: list[str] = []
        v_out: list[float | None] = []
        for i, sid in enumerate(sample_ids):
            for j, fid in enumerate(feature_ids):
                s_out.append(sid)
                f_out.append(fid)
                val = float(transformed[i, j])
                v_out.append(None if np.isnan(val) else val)
        return {"sample_id": s_out, "feature_id": f_out, cls.OUT_COL: v_out}


def _make_same_shape(name: str, transform: Any, blurb: str, out_col: str, example_doc: str) -> type:
    """Generate a same-shape compositional transform class."""
    cols = [
        ("sample_id", "VARCHAR", "Sample id."),
        ("feature_id", "VARCHAR", "Feature id."),
        (out_col, "DOUBLE", "Transformed value."),
    ]
    meta = type(
        "Meta",
        (),
        {
            "__doc__": f"VGI metadata for the {name} function.",
            "name": name,
            "description": f"{name} compositional transform of a long feature table",
            "categories": ["stats", "composition"],
            "examples": _comp_example(name, "feature_id", out_col, example_doc),
            "tags": _comp_doc(name, blurb, cols),
        },
    )
    return type(
        name.title().replace("_", ""),
        (_SameShape,),
        {
            "__doc__": f"{name} compositional transform.",
            "TRANSFORM": staticmethod(transform),
            "OUT_COL": out_col,
            "Meta": meta,
        },
    )


def _closure(m: Any) -> Any:
    from skbio.stats.composition import closure

    return closure(m)


def _centralize(m: Any) -> Any:
    from skbio.stats.composition import centralize, closure

    return centralize(closure(m))


def _rclr(m: Any) -> Any:
    from skbio.stats.composition import rclr

    return rclr(m)


def _multi_replace(m: Any) -> Any:
    from skbio.stats.composition import closure, multi_replace

    return multi_replace(closure(m))


_SAME_SHAPE_FUNCTIONS: list[type] = [
    _make_same_shape(
        "closure",
        _closure,
        "It rescales each sample's parts to sum to 1 (proportions) — the closure operation.",
        "proportion",
        "Normalise two samples with different totals (6 and 15 counts) onto a common scale, so "
        "their proportions can be compared directly. Closure is the first step of essentially "
        "every compositional workflow, and running it alone shows what the later transforms "
        "silently assume about their input.",
    ),
    _make_same_shape(
        "centralize",
        _centralize,
        "It centres each sample's composition around the dataset's geometric mean (perturbation to the centre).",
        "centered",
        "Re-express each sample relative to the dataset's average composition rather than its own "
        "total: values above 1 are features enriched compared with the typical sample, below 1 "
        "depleted. This is the compositional equivalent of subtracting the mean before plotting.",
    ),
    _make_same_shape(
        "rclr",
        _rclr,
        "It is the robust centred log-ratio: like CLR but it leaves zeros out of the geometric mean, so a "
        "pseudocount is usually unnecessary (leave `pseudocount :=` at 0).",
        "rclr",
        "Take log-ratios of a feature table without inventing a pseudocount: rclr excludes zeros "
        "from each sample's geometric mean instead of offsetting them. Reach for this rather than "
        "clr whenever the table is sparse, as microbiome count tables almost always are.",
    ),
    _make_same_shape(
        "multi_replace",
        _multi_replace,
        "It replaces zeros with small positive values by multiplicative replacement, preserving each sample's ratios.",
        "value",
        "Fill in zeros with small positive values while holding every non-zero ratio fixed — the "
        "principled alternative to adding an arbitrary pseudocount. Run this before clr, ilr, or "
        "alr, all of which are undefined at zero.",
    ),
]


@dataclass(slots=True, frozen=True)
class _PowerArgs:
    data: Annotated[TableInput, Arg(0, doc="One row per (sample_id, feature_id, value).")]
    sample: Annotated[str, Arg("sample", default="", doc="Sample-id column (defaults to the first column).")]
    feature: Annotated[str, Arg("feature", default="", doc="Feature-id column (defaults to the second column).")]
    value: Annotated[str, Arg("value", default="", doc="Value column (defaults to the third column).")]
    pseudocount: Annotated[float, Arg("pseudocount", default=0.0, doc="Added to every value before the transform.")]
    power: Annotated[float, Arg("power", default=1.0, doc="Exponent applied to each part of the composition.")]


class Power(_SameShape):
    """Compositional power (scalar) transform: raise each part to a power, then close."""

    OUT_COL: ClassVar[str] = "value"
    FunctionArguments: ClassVar[type] = _PowerArgs

    @classmethod
    def _apply(cls, matrix: Any, args: Any) -> Any:
        """Raise the closed composition to ``power`` and re-close."""
        from skbio.stats.composition import closure, power

        return power(closure(matrix), args.power)

    class Meta:
        """VGI metadata for the power function."""

        name = "power"
        description = "Compositional power transform of a long feature table"
        categories = ["stats", "composition"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT sample_id, feature_id, round(value, 4) AS value FROM skbio.stats.power("
                    f"{_COMP_INPUT}, power := 2.0) ORDER BY sample_id, feature_id"
                ),
                description=(
                    "Sharpen a composition by squaring every part and re-closing it: the dominant "
                    "features gain share and the rare ones lose it, while the result still sums "
                    "to 1. An exponent below 1 does the opposite (flattens), which is how "
                    "compositional data is up- or down-weighted without leaving the simplex."
                ),
            )
        ]
        tags = _comp_doc(
            "power",
            "It raises each part of the closed composition to the exponent `power :=` and re-closes it "
            "(compositional scalar multiplication).",
            [
                ("sample_id", "VARCHAR", "Sample id."),
                ("feature_id", "VARCHAR", "Feature id."),
                ("value", "DOUBLE", "Transformed value."),
            ],
        )


# ===========================================================================
# Dimension-changing transforms and inverses (long -> long, index axis)
# ===========================================================================


class _Indexed(SinkBuffer[_CompArgs, DrainState]):
    """Buffer a long table, apply a transform whose width may change, emit indexed long."""

    FunctionArguments: ClassVar[type] = _CompArgs
    KEY_COL: ClassVar[str] = "component"
    KEY_DOC: ClassVar[str] = "Output coordinate index (1-based)."
    VAL_COL: ClassVar[str] = "value"

    @classmethod
    def _apply(cls, matrix: Any, args: _CompArgs) -> Any:
        """Apply the transform to the pivoted matrix."""
        raise NotImplementedError

    @classmethod
    def on_bind(cls, params: BindParams[_CompArgs]) -> BindResponse:
        """Validate columns and fix the (sample_id, <key>, <val>) output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        _resolve_triple(input_schema, a.sample, a.feature, a.value)
        fields = [
            sfield("sample_id", pa.string(), "Sample id.", nullable=False),
            sfield(cls.KEY_COL, pa.int64(), cls.KEY_DOC, nullable=False),
            sfield(cls.VAL_COL, pa.float64(), "Transformed value.", nullable=True),
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
        """Pivot, transform, and emit indexed long, once."""
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
        """Transform each sample and return the indexed long-format columns."""
        s_col, f_col, v_col = _resolve_triple(table.schema, args.sample, args.feature, args.value)
        sample_ids, _keys, matrix = _pivot(table, s_col, f_col, v_col, args.pseudocount)
        transformed = np.atleast_2d(np.asarray(cls._apply(matrix, args), dtype=np.float64))
        s_out: list[str] = []
        k_out: list[int] = []
        v_out: list[float | None] = []
        for i, sid in enumerate(sample_ids):
            for k in range(transformed.shape[1]):
                s_out.append(sid)
                k_out.append(k + 1)
                val = float(transformed[i, k])
                v_out.append(None if np.isnan(val) else val)
        return {"sample_id": s_out, cls.KEY_COL: k_out, cls.VAL_COL: v_out}


@dataclass(slots=True, frozen=True)
class _AlrArgs:
    data: Annotated[TableInput, Arg(0, doc="One row per (sample_id, feature_id, value).")]
    sample: Annotated[str, Arg("sample", default="", doc="Sample-id column (defaults to the first column).")]
    feature: Annotated[str, Arg("feature", default="", doc="Feature-id column (defaults to the second column).")]
    value: Annotated[str, Arg("value", default="", doc="Value column (defaults to the third column).")]
    pseudocount: Annotated[float, Arg("pseudocount", default=0.0, doc="Added to every value before the transform.")]
    ref_idx: Annotated[int, Arg("ref_idx", default=0, doc="Index of the reference part (0-based, sorted features).")]


class Alr(_Indexed):
    """Additive log-ratio: D-1 log-ratios of each part to a reference part."""

    KEY_COL: ClassVar[str] = "component"
    KEY_DOC: ClassVar[str] = "ALR component index (1..D-1)."
    FunctionArguments: ClassVar[type] = _AlrArgs

    @classmethod
    def _apply(cls, matrix: Any, args: Any) -> Any:
        """Apply the ALR transform relative to ``ref_idx``."""
        from skbio.stats.composition import alr, closure

        return alr(closure(matrix), ref_idx=args.ref_idx)

    class Meta:
        """VGI metadata for the alr function."""

        name = "alr"
        description = "Additive log-ratio (ALR) transform of a long feature table (D-1 components)"
        categories = ["stats", "composition"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT sample_id, component, round(value, 4) AS value "
                    f"FROM skbio.stats.alr({_COMP_INPUT}) ORDER BY sample_id, component"
                ),
                description=(
                    "Express every part as a log-ratio against one chosen reference feature "
                    "(here the first by sorted order, via the default ref_idx := 0). Use alr "
                    "rather than clr or ilr when a genuine reference exists — a spike-in, a "
                    "housekeeping gene — because then the coordinates stay directly interpretable."
                ),
            )
        ]
        tags = _comp_doc(
            "alr",
            "It maps a D-part composition to D-1 log-ratios of each part against a reference part "
            "(`ref_idx :=`, default the first feature by sorted order).",
            [
                ("sample_id", "VARCHAR", "Sample id."),
                ("component", "BIGINT", "ALR component index (1..D-1)."),
                ("value", "DOUBLE", "Log-ratio value."),
            ],
        )


def _make_inverse(name: str, fn_name: str, blurb: str, ref: bool, example_doc: str) -> type:
    """Generate an inverse-transform class (clr_inv / ilr_inv / alr_inv)."""

    def _apply(cls: type, matrix: Any, args: Any) -> Any:
        import skbio.stats.composition as comp

        fn = getattr(comp, fn_name)
        return fn(matrix, ref_idx=args.ref_idx) if ref else fn(matrix)

    meta = type(
        "Meta",
        (),
        {
            "__doc__": f"VGI metadata for the {name} function.",
            "name": name,
            "description": f"{name}: inverse compositional transform back to proportions",
            "categories": ["stats", "composition"],
            "examples": [
                FunctionExample(
                    sql=(
                        "SELECT sample_id, feature, round(value, 6) AS value FROM "
                        f"skbio.stats.{name}((SELECT * FROM "
                        "(VALUES ('s1',1,0.1),('s1',2,-0.2),('s2',1,0.3),('s2',2,-0.1)) "
                        "AS t(sample_id, coordinate, value))) ORDER BY sample_id, feature"
                    ),
                    description=example_doc,
                )
            ],
            "tags": _comp_doc(
                name,
                blurb,
                [
                    ("sample_id", "VARCHAR", "Sample id."),
                    ("feature", "BIGINT", "Output part index (1-based)."),
                    ("value", "DOUBLE", "Proportion (parts sum to 1 per sample)."),
                ],
            ),
        },
    )
    args_cls = _AlrArgs if ref else _CompArgs
    return type(
        name.title().replace("_", ""),
        (_Indexed,),
        {
            "__doc__": f"{name} inverse transform.",
            "_apply": classmethod(_apply),
            "KEY_COL": "feature",
            "KEY_DOC": "Output part index (1-based).",
            "FunctionArguments": args_cls,
            "Meta": meta,
        },
    )


_INVERSE_FUNCTIONS: list[type] = [
    _make_inverse(
        "clr_inv",
        "clr_inv",
        "It inverts a centred log-ratio back to a composition (softmax of the CLR coordinates).",
        ref=False,
        example_doc=(
            "Map log-ratio coordinates back to readable proportions that sum to 1 per sample. "
            "This is the step that makes a model fitted in clr space reportable: predictions come "
            "out as log-ratios, and stakeholders want percentages."
        ),
    ),
    _make_inverse(
        "ilr_inv",
        "ilr_inv",
        "It inverts D-1 isometric log-ratio coordinates back to a D-part composition.",
        ref=False,
        example_doc=(
            "Rebuild a full composition from its D-1 isometric coordinates — note that 2 input "
            "coordinates come back as 3 parts. Pair it with ilr to round-trip: transform, model "
            "or perturb in real space, then invert to get proportions again."
        ),
    ),
    _make_inverse(
        "alr_inv",
        "alr_inv",
        "It inverts D-1 additive log-ratio coordinates back to a D-part composition.",
        ref=True,
        example_doc=(
            "Recover the proportions behind D-1 additive log-ratios, restoring the reference part "
            "that alr divided everything by. Use the same ref_idx := you transformed with, or the "
            "parts come back permuted."
        ),
    ),
]


# ===========================================================================
# Pairwise variance and differential abundance (grouping-based)
# ===========================================================================


class PairwiseVlr(SinkBuffer[_CompArgs, DrainState]):
    """Feature-by-feature variance of log-ratios (compositional association)."""

    FunctionArguments: ClassVar[type] = _CompArgs

    class Meta:
        """VGI metadata for the pairwise_vlr function."""

        name = "pairwise_vlr"
        description = "Pairwise variance of log-ratios between features (long matrix)"
        categories = ["stats", "composition"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT feature_1, feature_2, round(vlr, 4) AS vlr FROM "
                    "skbio.stats.pairwise_vlr((SELECT * FROM "
                    "(VALUES ('s1','a',1),('s1','b',2),('s1','c',3),('s2','a',4),('s2','b',5),('s2','c',6),"
                    "('s3','a',2),('s3','b',1),('s3','c',7)) AS t(sample_id, feature_id, value))) "
                    "WHERE feature_1 < feature_2 ORDER BY vlr"
                ),
                description=(
                    "Rank feature pairs by how tightly they track each other across samples, "
                    "smallest variance first — the pair at the top moves proportionally and is a "
                    "candidate for co-occurring taxa. Filtering to feature_1 < feature_2 drops "
                    "the zero diagonal and the mirrored half of the symmetric matrix."
                ),
            )
        ]
        tags = {
            "vgi.category": "composition",
            "vgi.result_columns_schema": result_columns_schema(
                [
                    ("feature_1", "VARCHAR", "First feature id."),
                    ("feature_2", "VARCHAR", "Second feature id."),
                    ("vlr", "DOUBLE", "Variance of log(feature_1 / feature_2) across samples."),
                ]
            ),
            "vgi.doc_llm": (
                "Table function computing the variance of the log-ratio between every pair of features "
                "across the samples of a long feature table `(SELECT sample_id, feature_id, value FROM ...)`. "
                "A low variance means two features move together (are proportional); it is a "
                "compositionally-coherent measure of feature association. Returns the full feature-by-feature "
                "matrix long as `(feature_1, feature_2, vlr)`; `pseudocount :=` offsets zeros."
            ),
            "vgi.doc_md": (
                "**pairwise_vlr** — variance of log-ratios between every pair of features.\n\n"
                "- Table arg: `(SELECT sample_id, feature_id, value FROM ...)`; `pseudocount :=` offsets zeros\n"
                "- Low variance = the two features are proportional (associated)\n"
                "- Returns the feature-by-feature matrix long: `feature_1`, `feature_2`, `vlr`"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[_CompArgs]) -> BindResponse:
        """Validate columns and fix the long feature-by-feature output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        _resolve_triple(input_schema, a.sample, a.feature, a.value)
        fields = [
            sfield("feature_1", pa.string(), "First feature id.", nullable=False),
            sfield("feature_2", pa.string(), "Second feature id.", nullable=False),
            sfield("vlr", pa.float64(), "Variance of the log-ratio.", nullable=False),
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
        """Pivot, compute the pairwise VLR matrix, and emit long, once."""
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
        """Compute the pairwise VLR matrix and return the long-format columns."""
        from skbio.stats.composition import closure, pairwise_vlr

        s_col, f_col, v_col = _resolve_triple(table.schema, args.sample, args.feature, args.value)
        _samples, feature_ids, matrix = _pivot(table, s_col, f_col, v_col, args.pseudocount)
        dm = pairwise_vlr(closure(matrix), ids=feature_ids)
        data = dm.data
        f1: list[str] = []
        f2: list[str] = []
        vlr: list[float] = []
        for i, a in enumerate(feature_ids):
            for j, b in enumerate(feature_ids):
                f1.append(a)
                f2.append(b)
                vlr.append(float(data[i, j]))
        return {"feature_1": f1, "feature_2": f2, "vlr": vlr}


@dataclass(slots=True, frozen=True)
class _DiffArgs:
    data: Annotated[TableInput, Arg(0, doc="One row per (sample_id, feature_id, count, group).")]
    sample: Annotated[str, Arg("sample", default="", doc="Sample-id column (defaults to the first column).")]
    feature: Annotated[str, Arg("feature", default="", doc="Feature-id column (defaults to the second column).")]
    value: Annotated[str, Arg("value", default="", doc="Count column (defaults to the third column).")]
    group: Annotated[str, Arg("group", default="", doc="Group label of the sample (defaults to the fourth column).")]


def _resolve_quad(schema: pa.Schema, args: _DiffArgs) -> tuple[str, str, str, str]:
    """Resolve (sample, feature, value, group) column names, defaulting to positional 0-3."""
    names = list(schema.names)
    s = args.sample or (names[0] if len(names) > 0 else "")
    f = args.feature or (names[1] if len(names) > 1 else "")
    v = args.value or (names[2] if len(names) > 2 else "")
    g = args.group or (names[3] if len(names) > 3 else "")
    for label, col in (("sample", s), ("feature", f), ("value", v), ("group", g)):
        if not col or col not in names:
            raise ValueError(f"{label} column {col!r} not found in input; columns: {', '.join(names)}")
    return s, f, v, g


def _feature_frame(table: pa.Table, s_col: str, f_col: str, v_col: str, g_col: str) -> tuple[Any, Any]:
    """Build a samples x features DataFrame and an aligned grouping Series."""
    import pandas as pd

    samples = [str(x) for x in table.column(s_col).to_pylist()]
    features = [str(x) for x in table.column(f_col).to_pylist()]
    values = table.column(v_col).to_pylist()
    groups = table.column(g_col).to_pylist()
    sample_ids = sorted(set(samples))
    feature_ids = sorted(set(features))
    s_index = {s: i for i, s in enumerate(sample_ids)}
    f_index = {f: j for j, f in enumerate(feature_ids)}
    mat = np.zeros((len(sample_ids), len(feature_ids)), dtype=np.float64)
    grp: dict[str, Any] = {}
    for s, f, v, g in zip(samples, features, values, groups, strict=True):
        if v is not None:
            mat[s_index[s], f_index[f]] += float(v)
        grp.setdefault(s, g)
    frame = pd.DataFrame(mat, index=sample_ids, columns=feature_ids)
    grouping = pd.Series([grp[s] for s in sample_ids], index=sample_ids, name="group")
    return frame, grouping


class Ancom(SinkBuffer[_DiffArgs, DrainState]):
    """ANCOM differential abundance: which features differ between groups."""

    FunctionArguments: ClassVar[type] = _DiffArgs

    class Meta:
        """VGI metadata for the ancom function."""

        name = "ancom"
        description = "ANCOM differential-abundance test per feature between groups"
        categories = ["stats", "composition"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT feature_id, w, significant FROM skbio.stats.ancom("
                    "(SELECT s.sample_id, s.feature_id, s.count, g.grp FROM "
                    "(VALUES ('s1','b1',12),('s1','b2',11),('s2','b1',9),('s2','b2',11),('s3','b1',1),"
                    "('s3','b2',11),('s4','b1',22),('s4','b2',21),('s5','b1',20),('s5','b2',22),"
                    "('s6','b1',23),('s6','b2',21)) AS s(sample_id, feature_id, count) "
                    "JOIN (VALUES ('s1','x'),('s2','x'),('s3','x'),('s4','y'),('s5','y'),('s6','y')) "
                    "AS g(sample, grp) ON s.sample_id = g.sample)) ORDER BY w DESC"
                ),
                description=(
                    "Find which features differ between two groups of samples, strongest evidence "
                    "first. The join is the part worth copying: a table function gets one input "
                    "relation, so the per-sample group label has to ride in as a fourth column "
                    "rather than a second argument."
                ),
            )
        ]
        tags = {
            "vgi.category": "composition",
            "vgi.result_columns_schema": result_columns_schema(
                [
                    ("feature_id", "VARCHAR", "Feature id."),
                    ("w", "BIGINT", "ANCOM W statistic (number of sub-hypotheses rejected)."),
                    ("significant", "BOOLEAN", "Whether the feature is declared differentially abundant."),
                ]
            ),
            "vgi.doc_llm": (
                "Table function running the ANCOM (Analysis of Composition of Microbiomes) differential-"
                "abundance test on a long count table with a per-sample grouping. The table arg is "
                "`(SELECT sample_id, feature_id, count, group FROM ...)` where `group` is the label of the "
                "sample (columns default to positional 1-4). For each feature it reports the ANCOM `w` "
                "statistic (how many log-ratio sub-hypotheses were rejected) and whether it is declared "
                "significantly differentially abundant. Build the input by joining a grouping onto a "
                "feature-count table."
            ),
            "vgi.doc_md": (
                "**ANCOM** — differential abundance of each feature between groups.\n\n"
                "- Table arg: `(SELECT sample_id, feature_id, count, group FROM ...)` (positional 1-4)\n"
                "- Returns per feature: `w` (rejected sub-hypotheses) and `significant`\n"
                "- Join a per-sample grouping onto a count table to build the input"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[_DiffArgs]) -> BindResponse:
        """Validate columns and fix the per-feature ANCOM result schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        _resolve_quad(input_schema, a)
        fields = [
            sfield("feature_id", pa.string(), "Feature id.", nullable=False),
            sfield("w", pa.int64(), "ANCOM W statistic.", nullable=False),
            sfield("significant", pa.bool_(), "Declared differentially abundant.", nullable=False),
        ]
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_DiffArgs]) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_DiffArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Run ANCOM and emit one row per feature, once."""
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
    def encode(cls, table: pa.Table, args: _DiffArgs) -> dict[str, list[Any]]:
        """Run ANCOM and return the per-feature result columns."""
        from skbio.stats.composition import ancom

        s, f, v, g = _resolve_quad(table.schema, args)
        frame, grouping = _feature_frame(table, s, f, v, g)
        result, _percentiles = ancom(frame + 1, grouping)
        return {
            "feature_id": [str(i) for i in result.index],
            "w": [int(x) for x in result["W"]],
            "significant": [bool(x) for x in result["Signif"]],
        }


class DirmultTtest(SinkBuffer[_DiffArgs, DrainState]):
    """Dirichlet-multinomial t-test differential abundance (effect sizes + p-values)."""

    FunctionArguments: ClassVar[type] = _DiffArgs

    class Meta:
        """VGI metadata for the dirmult_ttest function."""

        name = "dirmult_ttest"
        description = "Dirichlet-multinomial t-test for differential abundance per feature"
        categories = ["stats", "composition"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT feature_id, round(log2_fold_change, 4) AS log2_fold_change, "
                    "round(qvalue, 4) AS qvalue, significant FROM skbio.stats.dirmult_ttest("
                    "(SELECT s.sample_id, s.feature_id, s.count, g.grp FROM "
                    "(VALUES ('s1','b1',12),('s1','b2',11),('s2','b1',9),('s2','b2',11),('s3','b1',1),"
                    "('s3','b2',11),('s4','b1',22),('s4','b2',21),('s5','b1',20),('s5','b2',22),"
                    "('s6','b1',23),('s6','b2',21)) AS s(sample_id, feature_id, count) "
                    "JOIN (VALUES ('s1','x'),('s2','x'),('s3','x'),('s4','y'),('s5','y'),('s6','y')) "
                    "AS g(sample, grp) ON s.sample_id = g.sample)) ORDER BY qvalue"
                ),
                description=(
                    "Report an effect size and a multiple-testing-corrected q-value per feature, "
                    "most significant first. Prefer this over ancom when you need a direction and "
                    "magnitude (the log2 fold change) rather than just a ranked W statistic; the "
                    "seed is fixed, so the numbers are reproducible."
                ),
            )
        ]
        tags = {
            "vgi.category": "composition",
            "vgi.result_columns_schema": result_columns_schema(
                [
                    ("feature_id", "VARCHAR", "Feature id."),
                    ("t_statistic", "DOUBLE", "T-statistic for the group difference."),
                    ("log2_fold_change", "DOUBLE", "Estimated log2 fold change between groups."),
                    ("pvalue", "DOUBLE", "Raw p-value."),
                    ("qvalue", "DOUBLE", "Multiple-comparison-adjusted p-value (Holm)."),
                    ("significant", "BOOLEAN", "Whether the adjusted p-value is significant."),
                ]
            ),
            "vgi.doc_llm": (
                "Table function running scikit-bio's Dirichlet-multinomial t-test for differential abundance "
                "on a long count table with a per-sample grouping `(SELECT sample_id, feature_id, count, "
                "group FROM ...)`. It estimates, per feature, the log2 fold change between the two groups "
                "with a t-statistic, raw and Holm-adjusted p-values, and a significance flag. The two group "
                "labels are taken as reference and treatment in sorted order; the posterior is sampled with a "
                "fixed seed so results are reproducible. Build the input by joining a grouping onto a count "
                "table."
            ),
            "vgi.doc_md": (
                "**dirmult_ttest** — Dirichlet-multinomial differential abundance per feature.\n\n"
                "- Table arg: `(SELECT sample_id, feature_id, count, group FROM ...)` (positional 1-4)\n"
                "- Returns per feature: `t_statistic`, `log2_fold_change`, `pvalue`, `qvalue`, `significant`\n"
                "- Two group labels become reference/treatment (sorted); fixed seed for reproducibility"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[_DiffArgs]) -> BindResponse:
        """Validate columns and fix the per-feature result schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        _resolve_quad(input_schema, a)
        fields = [
            sfield("feature_id", pa.string(), "Feature id.", nullable=False),
            sfield("t_statistic", pa.float64(), "T-statistic.", nullable=True),
            sfield("log2_fold_change", pa.float64(), "Log2 fold change.", nullable=True),
            sfield("pvalue", pa.float64(), "Raw p-value.", nullable=True),
            sfield("qvalue", pa.float64(), "Adjusted p-value.", nullable=True),
            sfield("significant", pa.bool_(), "Significant after adjustment.", nullable=True),
        ]
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_DiffArgs]) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_DiffArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Run the test and emit one row per feature, once."""
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
    def encode(cls, table: pa.Table, args: _DiffArgs) -> dict[str, list[Any]]:
        """Run the Dirichlet-multinomial t-test and return the per-feature result columns."""
        from skbio.stats.composition import dirmult_ttest

        s, f, v, g = _resolve_quad(table.schema, args)
        frame, grouping = _feature_frame(table, s, f, v, g)
        labels = sorted({str(x) for x in grouping})
        if len(labels) != 2:
            raise ValueError(f"dirmult_ttest needs exactly two groups (got {len(labels)})")
        reference, treatment = labels[0], labels[1]
        result = dirmult_ttest(frame, grouping, treatment, reference, seed=0)
        return {
            "feature_id": [str(i) for i in result.index],
            "t_statistic": [float(x) for x in result["T-statistic"]],
            "log2_fold_change": [float(x) for x in result["Log2(FC)"]],
            "pvalue": [float(x) for x in result["pvalue"]],
            "qvalue": [float(x) for x in result["qvalue"]],
            "significant": [bool(x) for x in result["Signif"]],
        }


COMPOSITION_FUNCTIONS: list[type] = [
    Clr,
    Ilr,
    *_SAME_SHAPE_FUNCTIONS,
    Power,
    Alr,
    *_INVERSE_FUNCTIONS,
    PairwiseVlr,
    Ancom,
    DirmultTtest,
]
