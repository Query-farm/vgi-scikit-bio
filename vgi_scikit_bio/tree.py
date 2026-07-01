"""Phylogenetic trees: neighbour-joining construction and Newick inspection.

* ``neighbor_joining`` builds a tree from a long ``(id_1, id_2, distance)``
  distance matrix and returns it as a single Newick string:

      SELECT newick FROM skbio.tree.neighbor_joining((SELECT * FROM distances));

* ``tip_count`` / ``total_branch_length`` are scalar functions over a Newick
  string column, so trees stored in a table can be inspected in bulk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pyarrow as pa
from vgi.arguments import Arg, Param, Returns, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.scalar_function import ScalarFunction
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .distance_utils import distance_matrix_from_long, resolve_pair_columns
from .schema_utils import columns_md_rows
from .schema_utils import field as sfield


@dataclass(slots=True, frozen=True)
class _NjArgs:
    data: Annotated[TableInput, Arg(0, doc="A long distance matrix: (id_1, id_2, distance).")]
    id_1: Annotated[str, Arg("id_1", default="", doc="First-id column (defaults to the first column).")]
    id_2: Annotated[str, Arg("id_2", default="", doc="Second-id column (defaults to the second column).")]
    distance: Annotated[str, Arg("distance", default="", doc="Distance column (defaults to the third column).")]


class NeighborJoining(SinkBuffer[_NjArgs, DrainState]):
    """Build a neighbour-joining tree from a distance matrix; return one Newick row."""

    FunctionArguments: ClassVar[type] = _NjArgs

    class Meta:
        """VGI metadata for the neighbor_joining function."""

        name = "neighbor_joining"
        description = "Build a neighbour-joining tree from a distance matrix (Newick output)"
        categories = ["tree", "phylogenetics"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT newick FROM skbio.tree.neighbor_joining((SELECT * FROM "
                    "(VALUES ('a','b',5),('a','c',9),('a','d',9),('b','c',10),('b','d',10),('c','d',8)) "
                    "AS d(id_1, id_2, distance)))"
                ),
                description="Neighbour-joining tree of a 4-taxon distance matrix",
            )
        ]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [("newick", "VARCHAR", "The neighbour-joining tree in Newick format.")]
            ),
            "vgi.doc_llm": (
                "Table function building a neighbour-joining phylogenetic tree from a long distance matrix "
                "and returning it as a single-row Newick string. The table arg is "
                "`(SELECT id_1, id_2, distance FROM ...)` — typically a `beta_diversity` matrix (columns "
                "default to positional 1/2/3; override with `id_1 :=`, `id_2 :=`, `distance :=`). "
                "Neighbour-joining reconstructs an unrooted tree whose pairwise path lengths approximate "
                "the input distances. Returns one row with the `newick` string, which you can store and "
                "inspect with `skbio.tree.tip_count` / `total_branch_length` or render with any Newick "
                "viewer."
            ),
            "vgi.doc_md": (
                "**Neighbour joining** — a tree from a distance matrix.\n\n"
                "- Table arg: `(SELECT id_1, id_2, distance FROM ...)` (e.g. `beta_diversity` output; "
                "positional 1/2/3 by default)\n"
                "- Reconstructs an unrooted tree approximating the pairwise distances\n"
                "- Returns one row: `newick` (a Newick-format `VARCHAR`)\n"
                "- Inspect with `tip_count` / `total_branch_length`"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[_NjArgs]) -> BindResponse:
        """Validate columns and fix the single-column Newick output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        resolve_pair_columns(input_schema, a.id_1, a.id_2, a.distance)
        return BindResponse(
            output_schema=pa.schema([sfield("newick", pa.string(), "Neighbour-joining tree (Newick).", nullable=False)])
        )

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_NjArgs]) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_NjArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Reconstruct the matrix, run neighbour joining, and emit one Newick row, once."""
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
    def encode(cls, table: pa.Table, args: _NjArgs) -> dict[str, list[Any]]:
        """Run neighbour joining and return the single Newick string column."""
        from skbio.tree import nj

        id1, id2, dist = resolve_pair_columns(table.schema, args.id_1, args.id_2, args.distance)
        dm = distance_matrix_from_long(table, id1, id2, dist)
        newick = str(nj(dm)).strip()
        return {"newick": [newick]}


# ===========================================================================
# Newick inspection scalars
# ===========================================================================


def _read_tree(newick: str) -> Any:
    """Parse a Newick string into a scikit-bio TreeNode."""
    import warnings

    from skbio.tree import TreeNode

    with warnings.catch_warnings():
        # A malformed string trips a noisy format-sniffing warning before the
        # parse fails; the caller already turns that failure into NULL.
        warnings.simplefilter("ignore")
        return TreeNode.read([newick], format="newick")


class TipCount(ScalarFunction):
    """Number of tips (leaves) in a Newick tree."""

    class Meta:
        """VGI metadata for the tip_count scalar."""

        name = "tip_count"
        description = "Number of tips (leaves) in a Newick tree"
        categories = ["tree", "phylogenetics"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.tree.tip_count('((a:2,b:3):3,d:4,c:4);')",
                description="Count the tips of an inline Newick tree",
            )
        ]
        tags = {
            "vgi.doc_llm": (
                "Scalar function returning the number of tips (leaf nodes / taxa) in a Newick-format tree "
                "as a `BIGINT`. Pass one `VARCHAR` column of Newick strings (e.g. the output of "
                "`skbio.tree.neighbor_joining`). NULL or unparseable input returns NULL. Use it to size "
                "trees stored per row without leaving SQL."
            ),
            "vgi.doc_md": (
                "**tip_count** — number of leaves in a Newick tree.\n\n"
                "- Input: one `VARCHAR` Newick column\n"
                "- Returns: `BIGINT` tip count (NULL for NULL/unparseable input)"
            ),
        }

    @classmethod
    def compute(
        cls,
        newick: Annotated[pa.StringArray, Param(doc="A Newick-format tree")],
    ) -> Annotated[pa.Int64Array, Returns(pa.int64())]:
        """Return each tree's tip count (NULL for unparseable input)."""
        out: list[int | None] = []
        for raw in newick.to_pylist():
            if raw is None:
                out.append(None)
                continue
            try:
                out.append(int(_read_tree(str(raw)).count(tips=True)))
            except Exception:
                out.append(None)
        return pa.array(out, type=pa.int64())


class TotalBranchLength(ScalarFunction):
    """Total branch length (sum of all branch lengths) of a Newick tree."""

    class Meta:
        """VGI metadata for the total_branch_length scalar."""

        name = "total_branch_length"
        description = "Total branch length (sum of all branch lengths) of a Newick tree"
        categories = ["tree", "phylogenetics"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.tree.total_branch_length('((a:2,b:3):3,d:4,c:4);')",
                description="Total branch length of an inline Newick tree",
            )
        ]
        tags = {
            "vgi.doc_llm": (
                "Scalar function returning the total branch length of a Newick-format tree as a `DOUBLE`: "
                "the sum of every branch length in the tree. Pass one `VARCHAR` column of Newick strings "
                "(e.g. from `skbio.tree.neighbor_joining`). Branches without an explicit length count as "
                "zero; NULL or unparseable input returns NULL. This is Faith's phylogenetic-diversity "
                "measure when the tree spans a sample's features."
            ),
            "vgi.doc_md": (
                "**total_branch_length** — sum of all branch lengths in a Newick tree.\n\n"
                "- Input: one `VARCHAR` Newick column\n"
                "- Returns: `DOUBLE` total length (NULL for NULL/unparseable input)\n"
                "- Missing branch lengths count as 0; equals Faith's PD over a feature tree"
            ),
        }

    @classmethod
    def compute(
        cls,
        newick: Annotated[pa.StringArray, Param(doc="A Newick-format tree")],
    ) -> Annotated[pa.DoubleArray, Returns(pa.float64())]:
        """Return each tree's total branch length (NULL for unparseable input)."""
        out: list[float | None] = []
        for raw in newick.to_pylist():
            if raw is None:
                out.append(None)
                continue
            try:
                tree = _read_tree(str(raw))
                out.append(float(sum(n.length for n in tree.traverse() if n.length is not None)))
            except Exception:
                out.append(None)
        return pa.array(out, type=pa.float64())


TREE_FUNCTIONS: list[type] = [NeighborJoining, TipCount, TotalBranchLength]
