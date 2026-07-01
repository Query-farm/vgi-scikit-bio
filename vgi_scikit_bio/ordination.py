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
from .schema_utils import columns_md_rows
from .schema_utils import field as sfield


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
                    "SELECT * FROM skbio.stats.pcoa((SELECT * FROM "
                    "(VALUES ('a','a',0.0),('a','b',0.5),('a','c',0.7),('b','a',0.5),('b','b',0.0),"
                    "('b','c',0.6),('c','a',0.7),('c','b',0.6),('c','c',0.0)) AS d(id_1, id_2, distance)), "
                    "n_components := 2)"
                ),
                description="2-axis PCoA of a 3-sample distance matrix",
            )
        ]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    ("sample_id", "VARCHAR", "Sample id."),
                    ("pc_1", "DOUBLE", "Coordinate on the first principal axis (further axes follow)."),
                ]
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


ORDINATION_FUNCTIONS: list[type] = [Pcoa]
