"""Principal Coordinates Analysis (PCoA) over a distance matrix.

``pcoa`` reads a long ``(id_1, id_2, distance)`` distance matrix (as produced by
``skbio.diversity.beta_diversity``), runs classical multidimensional scaling, and
emits each sample's coordinates on the leading principal axes:

    SELECT * FROM skbio.stats.pcoa(
      (SELECT * FROM skbio.diversity.beta_diversity((SELECT * FROM feature_table))),
      n_components := 2);

The output width is fixed at bind time from ``n_components`` (default 3), so the
schema is ``(sample_id, pc_1, ..., pc_k)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .distance_utils import distance_matrix_from_long, resolve_pair_columns
from .schema_utils import field as sfield
from .schema_utils import result_dynamic_columns_md


def _axis_result_cols(prefix: str, axis_doc: str) -> str:
    """Result-schema variants for an ordination whose width follows ``n_components``.

    The output is ``(sample_id, <prefix>_1 .. <prefix>_k)`` where ``k`` is
    ``n_components :=`` (default 3), so the schema is argument-dependent: one
    variant per commonly-requested width rather than a single static schema.
    """
    sample = ("sample_id", "VARCHAR", "Sample id.")
    return result_dynamic_columns_md(
        [
            (
                f"`n_components := {k}`",
                [sample] + [(f"{prefix}_{i}", "DOUBLE", axis_doc.format(i=i)) for i in range(1, k + 1)],
            )
            for k in (2, 3)
        ],
        note=(
            f"The pattern generalises: `n_components := k` returns `sample_id` plus `{prefix}_1` "
            f"through `{prefix}_{{k}}` (default `k` = 3). Axes beyond what the data can support "
            "are returned as NULL."
        ),
    )


@dataclass(slots=True, frozen=True)
class _PcoaArgs:
    data: Annotated[TableInput, Arg(0, doc="A long distance matrix: (id_1, id_2, distance).")]
    n_components: Annotated[int, Arg("n_components", default=3, doc="Number of principal coordinate axes to return.")]
    id_1: Annotated[str, Arg("id_1", default="", doc="First-id column (defaults to the first column).")]
    id_2: Annotated[str, Arg("id_2", default="", doc="Second-id column (defaults to the second column).")]
    distance: Annotated[str, Arg("distance", default="", doc="Distance column (defaults to the third column).")]


class Pcoa(SinkBuffer[_PcoaArgs, DrainState]):
    """Principal Coordinates Analysis: distance matrix -> per-sample coordinates."""

    FunctionArguments: ClassVar[type] = _PcoaArgs

    class Meta:
        """VGI metadata for the pcoa function."""

        name = "pcoa"
        description = "Principal Coordinates Analysis of a distance matrix (per-sample coordinates)"
        categories = ["stats", "ordination"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT sample_id, round(pc_1, 4) AS pc_1, round(pc_2, 4) AS pc_2 FROM "
                    "skbio.stats.pcoa((SELECT * FROM "
                    "(VALUES ('a','a',0.0),('a','b',0.5),('a','c',0.7),('b','a',0.5),('b','b',0.0),"
                    "('b','c',0.6),('c','a',0.7),('c','b',0.6),('c','c',0.0)) AS d(id_1, id_2, distance)), "
                    "n_components := 2) ORDER BY pc_1"
                ),
                description=(
                    "Collapse a 3-sample distance matrix into two plottable coordinates and order "
                    "the samples along the leading axis. Rounding keeps the output readable; "
                    "sorting by pc_1 reads off the dominant gradient — which samples sit at the "
                    "extremes of the largest source of variation."
                ),
            )
        ]
        tags = {
            "vgi.result_dynamic_columns_md": _axis_result_cols(
                "pc", "Coordinate on principal axis {i} (axes are ordered by variance explained)."
            ),
            "vgi.doc_llm": (
                "Table function running Principal Coordinates Analysis (classical MDS) on a long distance "
                "matrix and returning each sample's coordinates on the leading axes. The table arg is "
                "`(SELECT id_1, id_2, distance FROM ...)` — typically the output of "
                "`skbio.diversity.beta_diversity` (columns default to positional 1/2/3; override with "
                "`id_1 :=`, `id_2 :=`, `distance :=`). `n_components :=` sets how many principal-coordinate "
                "axes to return (default 3), fixing the output width at `(sample_id, pc_1, ..., pc_k)`. Use "
                "it to embed samples in a low-dimensional space for plotting or clustering; the axes are "
                "ordered by variance explained."
            ),
            "vgi.doc_md": (
                "**PCoA** — principal coordinates (classical MDS) of a distance matrix.\n\n"
                "- Table arg: `(SELECT id_1, id_2, distance FROM ...)` (e.g. `beta_diversity` output; "
                "positional 1/2/3 by default)\n"
                "- `n_components :=` — number of axes to return (default 3)\n"
                "- Returns `(sample_id, pc_1, ..., pc_k)`; axes ordered by variance explained\n"
                "- Feed a beta-diversity matrix in to embed samples for plotting/clustering"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[_PcoaArgs]) -> BindResponse:
        """Validate columns and n_components, and fix the (sample_id, pc_1..k) output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        resolve_pair_columns(input_schema, a.id_1, a.id_2, a.distance)
        if a.n_components < 1:
            raise ValueError(f"n_components must be >= 1 (got {a.n_components})")
        fields = [sfield("sample_id", pa.string(), "Sample id.", nullable=False)]
        for k in range(1, a.n_components + 1):
            fields.append(sfield(f"pc_{k}", pa.float64(), f"Coordinate on principal axis {k}.", nullable=True))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_PcoaArgs]) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_PcoaArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Reconstruct the matrix, run PCoA, and emit per-sample coordinates, once."""
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        input_schema = input_schema_of(params)
        out_schema = params.output_schema
        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            out.emit(pa.RecordBatch.from_pydict({n: [] for n in out_schema.names}, schema=out_schema))
            return
        out.emit(pa.RecordBatch.from_pydict(cls.encode(table, a), schema=out_schema))

    @classmethod
    def encode(cls, table: pa.Table, args: _PcoaArgs) -> dict[str, list[Any]]:
        """Run PCoA and return the (sample_id, pc_1..k) columns."""
        from skbio.stats.ordination import pcoa

        id1, id2, dist = resolve_pair_columns(table.schema, args.id_1, args.id_2, args.distance)
        dm = distance_matrix_from_long(table, id1, id2, dist)
        # scikit-bio cannot return more axes than samples; compute what it can
        # and NULL-pad any requested components beyond that below.
        dims = min(args.n_components, dm.shape[0])
        result = pcoa(dm, number_of_dimensions=dims)
        samples = result.samples  # DataFrame indexed by sample id
        available = samples.shape[1]

        columns: dict[str, list[Any]] = {"sample_id": [str(i) for i in samples.index]}
        for k in range(1, args.n_components + 1):
            if k <= available:
                columns[f"pc_{k}"] = [float(v) for v in samples.iloc[:, k - 1].to_numpy()]
            else:
                columns[f"pc_{k}"] = [None] * samples.shape[0]
        return columns


# ===========================================================================
# Feature-table ordination (PCA / CA over a long feature table)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class _FeatureOrdArgs:
    data: Annotated[TableInput, Arg(0, doc="One row per (sample_id, feature_id, value).")]
    n_components: Annotated[int, Arg("n_components", default=3, doc="Number of ordination axes to return.")]
    sample: Annotated[str, Arg("sample", default="", doc="Sample-id column (defaults to the first column).")]
    feature: Annotated[str, Arg("feature", default="", doc="Feature-id column (defaults to the second column).")]
    value: Annotated[str, Arg("value", default="", doc="Value column (defaults to the third column).")]


def _resolve_feature_triple(schema: pa.Schema, sample: str, feature: str, value: str) -> tuple[str, str, str]:
    """Resolve the (sample, feature, value) column names, defaulting to positional 0/1/2."""
    names = list(schema.names)
    s = sample or (names[0] if len(names) > 0 else "")
    f = feature or (names[1] if len(names) > 1 else "")
    v = value or (names[2] if len(names) > 2 else "")
    for label, col in (("sample", s), ("feature", f), ("value", v)):
        if not col or col not in names:
            raise ValueError(f"{label} column {col!r} not found in input; columns: {', '.join(names)}")
    return s, f, v


class _FeatureOrdination(SinkBuffer[_FeatureOrdArgs, DrainState]):
    """Buffer a long feature table, run an ordination, emit per-sample coordinates."""

    FunctionArguments: ClassVar[type] = _FeatureOrdArgs
    PREFIX: ClassVar[str] = "axis"

    @classmethod
    def _ordinate(cls, frame: Any, n_components: int) -> Any:
        """Run the ordination and return an object with a ``.samples`` DataFrame."""
        raise NotImplementedError

    @classmethod
    def on_bind(cls, params: BindParams[_FeatureOrdArgs]) -> BindResponse:
        """Validate columns/n_components and fix the (sample_id, <prefix>_1..k) schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        _resolve_feature_triple(input_schema, a.sample, a.feature, a.value)
        if a.n_components < 1:
            raise ValueError(f"n_components must be >= 1 (got {a.n_components})")
        fields = [sfield("sample_id", pa.string(), "Sample id.", nullable=False)]
        for k in range(1, a.n_components + 1):
            fields.append(sfield(f"{cls.PREFIX}_{k}", pa.float64(), f"Coordinate on axis {k}.", nullable=True))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[_FeatureOrdArgs]
    ) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_FeatureOrdArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Pivot the feature table, ordinate, and emit per-sample coordinates, once."""
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
    def encode(cls, table: pa.Table, args: _FeatureOrdArgs) -> dict[str, list[Any]]:
        """Ordinate the feature table and return the (sample_id, <prefix>_1..k) columns."""
        import numpy as np
        import pandas as pd

        s_col, f_col, v_col = _resolve_feature_triple(table.schema, args.sample, args.feature, args.value)
        samples = [str(x) for x in table.column(s_col).to_pylist()]
        features = [str(x) for x in table.column(f_col).to_pylist()]
        values = table.column(v_col).to_pylist()
        sample_ids = sorted(set(samples))
        feature_ids = sorted(set(features))
        s_index = {s: i for i, s in enumerate(sample_ids)}
        f_index = {f: j for j, f in enumerate(feature_ids)}
        mat = np.zeros((len(sample_ids), len(feature_ids)), dtype=np.float64)
        for s, f, v in zip(samples, features, values, strict=True):
            if v is not None:
                mat[s_index[s], f_index[f]] += float(v)

        frame = pd.DataFrame(mat, index=sample_ids, columns=feature_ids)
        result = cls._ordinate(frame, args.n_components)
        coords = result.samples
        available = coords.shape[1]
        columns: dict[str, list[Any]] = {"sample_id": [str(i) for i in coords.index]}
        for k in range(1, args.n_components + 1):
            if k <= available:
                columns[f"{cls.PREFIX}_{k}"] = [float(x) for x in coords.iloc[:, k - 1].to_numpy()]
            else:
                columns[f"{cls.PREFIX}_{k}"] = [None] * coords.shape[0]
        return columns


class Pca(_FeatureOrdination):
    """Principal Components Analysis of a long feature table (per-sample scores)."""

    PREFIX: ClassVar[str] = "pc"

    @classmethod
    def _ordinate(cls, frame: Any, n_components: int) -> Any:
        """Run PCA over the samples x features frame."""
        from skbio.stats.ordination import pca

        return pca(frame)

    class Meta:
        """VGI metadata for the pca function."""

        name = "pca"
        description = "Principal Components Analysis of a long feature table (per-sample scores)"
        categories = ["stats", "ordination"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT sample_id, round(pc_1, 4) AS pc_1, round(pc_2, 4) AS pc_2 FROM "
                    "skbio.stats.pca((SELECT * FROM "
                    "(VALUES ('s1','a',4),('s1','b',2),('s2','a',1),('s2','b',9),('s3','a',0),('s3','b',5)) "
                    "AS t(sample_id, feature_id, value)), n_components := 2) ORDER BY pc_1"
                ),
                description=(
                    "Score three samples on the two leading principal components straight from a "
                    "long feature table — no distance matrix needed, unlike pcoa. Ordering by "
                    "pc_1 shows which samples the dominant feature gradient separates."
                ),
            )
        ]
        tags = {
            "vgi.category": "ordination",
            "vgi.result_dynamic_columns_md": _axis_result_cols(
                "pc", "Score on principal component {i} (components are ordered by variance explained)."
            ),
            "vgi.doc_llm": (
                "Table function running Principal Components Analysis on a long feature table and returning "
                "each sample's scores on the leading components. The table arg is "
                "`(SELECT sample_id, feature_id, value FROM ...)` (columns default to positional 1/2/3; "
                "override with `sample :=`, `feature :=`, `value :=`). `n_components :=` sets how many "
                "principal-component axes to return (default 3), fixing the output width at "
                "`(sample_id, pc_1, ..., pc_k)`. Unlike PCoA (which needs a distance matrix), PCA works "
                "directly on the feature values, decomposing their covariance; axes are ordered by variance "
                "explained."
            ),
            "vgi.doc_md": (
                "**PCA** — principal components of a feature table.\n\n"
                "- Table arg: `(SELECT sample_id, feature_id, value FROM ...)` (positional 1/2/3 by "
                "default)\n"
                "- `n_components :=` — number of axes to return (default 3)\n"
                "- Returns `(sample_id, pc_1, ..., pc_k)`; axes ordered by variance explained\n"
                "- Works directly on feature values (no distance matrix needed, unlike `pcoa`)"
            ),
        }


class Ca(_FeatureOrdination):
    """Correspondence Analysis of a long feature (count) table (per-sample scores)."""

    PREFIX: ClassVar[str] = "ca"

    @classmethod
    def _ordinate(cls, frame: Any, n_components: int) -> Any:
        """Run correspondence analysis over the samples x features frame."""
        from skbio.stats.ordination import ca

        return ca(frame)

    class Meta:
        """VGI metadata for the ca function."""

        name = "ca"
        description = "Correspondence Analysis of a long feature (count) table (per-sample scores)"
        categories = ["stats", "ordination"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT sample_id, round(ca_1, 4) AS ca_1, round(ca_2, 4) AS ca_2 FROM "
                    "skbio.stats.ca((SELECT * FROM "
                    "(VALUES ('s1','a',4),('s1','b',2),('s2','a',1),('s2','b',9),('s3','a',3),('s3','b',5)) "
                    "AS t(sample_id, feature_id, value)), n_components := 2) ORDER BY ca_1"
                ),
                description=(
                    "Ordinate a count table with correspondence analysis, the chi-square "
                    "counterpart of PCA for abundance data. Sorting by ca_1 orders the samples "
                    "along the strongest compositional gradient, which is what the axis means "
                    "for count data."
                ),
            )
        ]
        tags = {
            "vgi.category": "ordination",
            "vgi.result_dynamic_columns_md": _axis_result_cols(
                "ca", "Score on correspondence axis {i} (axes are ordered by inertia explained)."
            ),
            "vgi.doc_llm": (
                "Table function running Correspondence Analysis on a long feature count table and returning "
                "each sample's scores on the leading axes. The table arg is "
                "`(SELECT sample_id, feature_id, value FROM ...)` (positional 1/2/3 by default; override "
                "with `sample :=`, `feature :=`, `value :=`). `n_components :=` sets the number of axes "
                "(default 3), fixing the output at `(sample_id, ca_1, ..., ca_k)`. CA is the ordination of "
                "choice for count/abundance data with a unimodal (chi-square) response; values must be "
                "non-negative. Axes are ordered by inertia explained."
            ),
            "vgi.doc_md": (
                "**CA** — correspondence analysis of a feature count table.\n\n"
                "- Table arg: `(SELECT sample_id, feature_id, value FROM ...)` (positional 1/2/3 by "
                "default)\n"
                "- `n_components :=` — number of axes to return (default 3)\n"
                "- Returns `(sample_id, ca_1, ..., ca_k)`; values must be non-negative counts\n"
                "- Chi-square ordination for abundance data; axes ordered by inertia explained"
            ),
        }


ORDINATION_FUNCTIONS: list[type] = [Pcoa, Pca, Ca]
