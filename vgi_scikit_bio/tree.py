"""Phylogenetic trees: build from distances, and inspect/compare Newick strings.

* **Builders** -- ``neighbor_joining`` / ``upgma`` / ``gme`` / ``bme`` build a
  tree from a long ``(id_1, id_2, distance)`` distance matrix and return a single
  Newick string.
* **Inspection scalars** -- ``tip_count`` / ``total_branch_length`` /
  ``tree_height`` read properties of a Newick string, per row.
* **Comparison scalars** -- ``robinson_foulds`` / ``weighted_robinson_foulds`` /
  ``cophenetic_distance`` compare two Newick trees, per row.
"""

from __future__ import annotations

from collections.abc import Callable
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
from .schema_utils import field as sfield
from .schema_utils import result_columns_schema


def _read_tree(newick: str) -> Any:
    """Parse a Newick string into a scikit-bio TreeNode."""
    import warnings

    from skbio.tree import TreeNode

    with warnings.catch_warnings():
        # A malformed string trips a noisy format-sniffing warning before the
        # parse fails; the caller already turns that failure into NULL.
        warnings.simplefilter("ignore")
        return TreeNode.read([newick], format="newick")


# ===========================================================================
# Tree builders (long distance matrix -> one Newick row)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class _BuildArgs:
    data: Annotated[TableInput, Arg(0, doc="A long distance matrix: (id_1, id_2, distance).")]
    id_1: Annotated[str, Arg("id_1", default="", doc="First-id column (defaults to the first column).")]
    id_2: Annotated[str, Arg("id_2", default="", doc="Second-id column (defaults to the second column).")]
    distance: Annotated[str, Arg("distance", default="", doc="Distance column (defaults to the third column).")]


class _TreeBuilder(SinkBuffer[_BuildArgs, DrainState]):
    """Build a tree from a distance matrix and return one Newick row (subclasses set BUILDER)."""

    FunctionArguments: ClassVar[type] = _BuildArgs
    BUILDER: ClassVar[Callable[..., Any]]

    @classmethod
    def on_bind(cls, params: BindParams[_BuildArgs]) -> BindResponse:
        """Validate columns and fix the single-column Newick output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        resolve_pair_columns(input_schema, a.id_1, a.id_2, a.distance)
        return BindResponse(
            output_schema=pa.schema([sfield("newick", pa.string(), "The tree in Newick format.", nullable=False)])
        )

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_BuildArgs]) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_BuildArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Reconstruct the matrix, build the tree, and emit one Newick row, once."""
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
    def encode(cls, table: pa.Table, args: _BuildArgs) -> dict[str, list[Any]]:
        """Build the tree and return the single Newick string column."""
        id1, id2, dist = resolve_pair_columns(table.schema, args.id_1, args.id_2, args.distance)
        dm = distance_matrix_from_long(table, id1, id2, dist)
        return {"newick": [str(cls.BUILDER(dm)).strip()]}


def _make_builder(name: str, builder: Any, blurb: str, example_doc: str) -> type:
    """Generate a distance-matrix tree-builder class."""
    example = FunctionExample(
        sql=(
            f"SELECT newick, skbio.tree.tip_count(newick) AS tips FROM skbio.tree.{name}("
            "(SELECT * FROM "
            "(VALUES ('a','b',5),('a','c',9),('a','d',9),('b','c',10),('b','d',10),('c','d',8)) "
            "AS d(id_1, id_2, distance)))"
        ),
        description=example_doc,
    )
    meta = type(
        "Meta",
        (),
        {
            "__doc__": f"VGI metadata for the {name} function.",
            "name": name,
            "description": f"Build a {name} tree from a distance matrix (Newick output)",
            "categories": ["tree", "phylogenetics"],
            "examples": [example],
            "tags": {
                "vgi.category": "construction",
                "vgi.result_columns_schema": result_columns_schema(
                    [("newick", "VARCHAR", "The tree in Newick format.")]
                ),
                "vgi.doc_llm": (
                    f"Table function building a phylogenetic tree from a long distance matrix and returning "
                    f"it as a single-row Newick string. {blurb} The table arg is "
                    f"`(SELECT id_1, id_2, distance FROM ...)` — typically a `beta_diversity` matrix "
                    f"(columns default to positional 1/2/3). Inspect the result with `tip_count` / "
                    f"`total_branch_length` / `tree_height` or compare trees with `robinson_foulds`."
                ),
                "vgi.doc_md": (
                    f"**{name}** — build a tree from a distance matrix.\n\n"
                    f"{blurb}\n\n"
                    "- Table arg: `(SELECT id_1, id_2, distance FROM ...)` (positional 1/2/3 by default)\n"
                    "- Returns one row: `newick` (a Newick-format `VARCHAR`)"
                ),
            },
        },
    )

    def _builder(dm: Any) -> Any:
        import skbio.tree as tree_mod

        return getattr(tree_mod, builder)(dm)

    return type(
        name.title().replace("_", ""),
        (_TreeBuilder,),
        {"__doc__": f"{name} tree builder.", "BUILDER": staticmethod(_builder), "Meta": meta},
    )


_BUILDER_FUNCTIONS: list[type] = [
    _make_builder(
        "neighbor_joining",
        "nj",
        "Neighbour joining reconstructs an unrooted tree whose pairwise path lengths approximate the input "
        "distances — the standard distance-based method.",
        "Turn a pairwise distance matrix into an actual tree, then check it round-tripped by "
        "counting its tips. Neighbour joining is the default choice when the distances come "
        "from real data and no molecular clock can be assumed; feed it a beta_diversity matrix "
        "to get a sample dendrogram straight out of SQL.",
    ),
    _make_builder(
        "upgma",
        "upgma",
        "UPGMA builds a rooted, ultrametric tree by repeatedly joining the closest clusters (average "
        "linkage); it assumes a molecular clock.",
        "Build a rooted, ultrametric tree — every tip ends up equidistant from the root, so "
        "tree_height is the same for all of them. Reach for UPGMA when you want a clustering "
        "dendrogram to cut at a threshold, and for neighbor_joining when you want a phylogeny.",
    ),
    _make_builder(
        "gme",
        "gme",
        "Greedy minimum evolution builds an unrooted tree by greedily minimising total tree length — fast "
        "for large matrices.",
        "Build a minimum-evolution tree greedily, which is the one to reach for when the matrix "
        "is large enough that neighbour joining's cubic cost hurts. On a small matrix like this "
        "one it agrees with the others; the tip count confirms no taxa were dropped.",
    ),
    _make_builder(
        "bme",
        "bme",
        "Balanced minimum evolution builds an unrooted tree minimising a balanced tree-length criterion "
        "(the objective FastME optimises).",
        "Build the balanced-minimum-evolution tree (the criterion FastME optimises), which is "
        "usually more accurate than greedy ME at a similar cost. Compare its output against "
        "neighbor_joining with robinson_foulds to see whether the method choice changed the "
        "topology at all.",
    ),
]


# ===========================================================================
# Inspection scalars (properties of one Newick string)
# ===========================================================================


class TipCount(ScalarFunction):
    """Number of tips (leaves) in a Newick tree."""

    class Meta:
        """VGI metadata for the tip_count scalar."""

        name = "tip_count"
        description = "Number of tips (leaves) in a Newick tree"
        categories = ["tree", "phylogenetics"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.tree.tip_count('((a:2,b:3):3,d:4,c:4);') AS tips",
                description=(
                    "Size a tree without leaving SQL — the quickest sanity check that a builder "
                    "kept every taxon, and the denominator for anything you normalise per tip. "
                    "Point it at a column of Newick strings to profile a whole table of trees."
                ),
            )
        ]
        tags = {
            "vgi.category": "inspection",
            "vgi.doc_llm": (
                "Scalar function returning the number of tips (leaf nodes / taxa) in a Newick-format tree "
                "as a `BIGINT`. Pass one `VARCHAR` column of Newick strings (e.g. the output of a tree "
                "builder). NULL or unparseable input returns NULL. Use it to size trees stored per row "
                "without leaving SQL."
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
        newick: Annotated[pa.StringArray, Param(doc="A Newick tree")],
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
                sql="SELECT skbio.tree.total_branch_length('((a:2,b:3):3,d:4,c:4);') AS branch_length",
                description=(
                    "Measure how much evolutionary change a tree encodes in total. Over a tree "
                    "pruned to one sample's features this is exactly Faith's PD, which makes it "
                    "the way to compute phylogenetic diversity when you already hold the subtree "
                    "rather than a feature table."
                ),
            )
        ]
        tags = {
            "vgi.category": "inspection",
            "vgi.doc_llm": (
                "Scalar function returning the total branch length of a Newick-format tree as a `DOUBLE`: "
                "the sum of every branch length. Pass one `VARCHAR` column of Newick strings. Branches "
                "without an explicit length count as zero; NULL or unparseable input returns NULL. This is "
                "Faith's phylogenetic-diversity measure when the tree spans a sample's features."
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
        newick: Annotated[pa.StringArray, Param(doc="A Newick tree")],
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


class TreeHeight(ScalarFunction):
    """Height of a Newick tree (maximum root-to-tip distance)."""

    class Meta:
        """VGI metadata for the tree_height scalar."""

        name = "tree_height"
        description = "Height of a Newick tree (maximum root-to-tip branch-length distance)"
        categories = ["tree", "phylogenetics"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.tree.tree_height('((a:2,b:3):3,d:4,c:4);') AS height",
                description=(
                    "Read the deepest root-to-tip distance, i.e. how far back the tree reaches. "
                    "On an ultrametric tree (anything from upgma) every tip sits at this height, "
                    "so comparing it with a tip's own depth is a quick test of ultrametricity."
                ),
            )
        ]
        tags = {
            "vgi.category": "inspection",
            "vgi.doc_llm": (
                "Scalar function returning the height of a Newick-format tree as a `DOUBLE`: the maximum "
                "branch-length distance from the root to any tip. Pass one `VARCHAR` column of Newick "
                "strings. Missing branch lengths count as zero; NULL or unparseable input returns NULL. For "
                "an ultrametric tree (e.g. from `upgma`) every tip is at this height."
            ),
            "vgi.doc_md": (
                "**tree_height** — deepest root-to-tip distance in a Newick tree.\n\n"
                "- Input: one `VARCHAR` Newick column\n"
                "- Returns: `DOUBLE` height (NULL for NULL/unparseable input)\n"
                "- For ultrametric trees (`upgma`) all tips sit at this height"
            ),
        }

    @classmethod
    def compute(
        cls,
        newick: Annotated[pa.StringArray, Param(doc="A Newick tree")],
    ) -> Annotated[pa.DoubleArray, Returns(pa.float64())]:
        """Return each tree's height (NULL for unparseable input)."""
        out: list[float | None] = []
        for raw in newick.to_pylist():
            if raw is None:
                out.append(None)
                continue
            try:
                tree = _read_tree(str(raw))
                out.append(float(max((tree.distance(tip) for tip in tree.tips()), default=0.0)))
            except Exception:
                out.append(None)
        return pa.array(out, type=pa.float64())


# ===========================================================================
# Comparison scalars (distance between two Newick trees)
# ===========================================================================


def _compare(newick1: pa.Array, newick2: pa.Array, fn: Callable[[Any, Any], float]) -> pa.Array:
    """Apply a two-tree comparison to each row (NULL on NULL/unparseable)."""
    out: list[float | None] = []
    for a, b in zip(newick1.to_pylist(), newick2.to_pylist(), strict=False):
        if a is None or b is None:
            out.append(None)
            continue
        try:
            out.append(float(fn(_read_tree(str(a)), _read_tree(str(b)))))
        except Exception:
            out.append(None)
    return pa.array(out, type=pa.float64())


class RobinsonFoulds(ScalarFunction):
    """Robinson-Foulds (symmetric-difference) topological distance between two trees."""

    class Meta:
        """VGI metadata for the robinson_foulds scalar."""

        name = "robinson_foulds"
        description = "Robinson-Foulds topological distance between two Newick trees"
        categories = ["tree", "phylogenetics"]
        examples = [
            FunctionExample(
                sql=("SELECT skbio.tree.robinson_foulds('((a,b),(c,d));', '((a,c),(b,d));') AS rf_distance"),
                description=(
                    "Ask whether two trees group the same taxa together, ignoring branch lengths "
                    "entirely: these two disagree on whether a pairs with b or with c, so the "
                    "distance is non-zero. This is the standard way to check whether two "
                    "reconstruction methods, or two bootstrap replicates, found the same shape."
                ),
            )
        ]
        tags = {
            "vgi.category": "comparison",
            "vgi.doc_llm": (
                "Scalar function returning the Robinson-Foulds distance between two Newick trees as a "
                "`DOUBLE`: the number of bipartitions (splits) present in one tree but not the other — a "
                "purely topological distance ignoring branch lengths. Pass two `VARCHAR` Newick columns "
                "over the same taxa; 0 means identical topologies, larger means more different. NULL or "
                "unparseable input returns NULL."
            ),
            "vgi.doc_md": (
                "**robinson_foulds** — topological distance between two Newick trees.\n\n"
                "- Inputs: two `VARCHAR` Newick columns over the same taxa\n"
                "- Returns: `DOUBLE` split-difference count (0 = identical topology); ignores branch lengths\n"
                "- NULL for NULL/unparseable input"
            ),
        }

    @classmethod
    def compute(
        cls,
        newick1: Annotated[pa.StringArray, Param(doc="First Newick tree")],
        newick2: Annotated[pa.StringArray, Param(doc="Second Newick tree")],
    ) -> Annotated[pa.DoubleArray, Returns(pa.float64())]:
        """Return the Robinson-Foulds distance per row (NULL on NULL/invalid input)."""
        return _compare(newick1, newick2, lambda a, b: a.compare_rfd(b))


class WeightedRobinsonFoulds(ScalarFunction):
    """Weighted Robinson-Foulds distance (uses branch lengths) between two trees."""

    class Meta:
        """VGI metadata for the weighted_robinson_foulds scalar."""

        name = "weighted_robinson_foulds"
        description = "Weighted Robinson-Foulds distance (branch-length aware) between two Newick trees"
        categories = ["tree", "phylogenetics"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT skbio.tree.weighted_robinson_foulds("
                    "'((a:1,b:1):1,(c:1,d:1):1);', '((a:1,c:1):1,(b:1,d:1):1);') AS wrf_distance"
                ),
                description=(
                    "Compare two trees on topology *and* branch lengths, so a pair that groups "
                    "taxa identically but stretches a branch still scores non-zero. Use this "
                    "rather than robinson_foulds when the lengths carry meaning — divergence "
                    "times, substitution rates — and not just the grouping."
                ),
            )
        ]
        tags = {
            "vgi.category": "comparison",
            "vgi.doc_llm": (
                "Scalar function returning the weighted Robinson-Foulds distance between two Newick trees "
                "as a `DOUBLE`: like the Robinson-Foulds distance but summing the branch-length differences "
                "of matched and unmatched splits, so it reflects both topology and branch lengths. Pass two "
                "`VARCHAR` Newick columns over the same taxa; 0 means identical. NULL or unparseable input "
                "returns NULL."
            ),
            "vgi.doc_md": (
                "**weighted_robinson_foulds** — branch-length-aware tree distance.\n\n"
                "- Inputs: two `VARCHAR` Newick columns over the same taxa\n"
                "- Returns: `DOUBLE` distance combining topology and branch lengths (0 = identical)\n"
                "- NULL for NULL/unparseable input"
            ),
        }

    @classmethod
    def compute(
        cls,
        newick1: Annotated[pa.StringArray, Param(doc="First Newick tree")],
        newick2: Annotated[pa.StringArray, Param(doc="Second Newick tree")],
    ) -> Annotated[pa.DoubleArray, Returns(pa.float64())]:
        """Return the weighted Robinson-Foulds distance per row (NULL on NULL/invalid input)."""
        return _compare(newick1, newick2, lambda a, b: a.compare_wrfd(b))


class CopheneticDistance(ScalarFunction):
    """Distance between two trees based on their tip-to-tip (cophenetic) distances."""

    class Meta:
        """VGI metadata for the cophenetic_distance scalar."""

        name = "cophenetic_distance"
        description = "Distance between two Newick trees based on tip-to-tip distances"
        categories = ["tree", "phylogenetics"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT skbio.tree.cophenetic_distance("
                    "'((a:1,b:1):1,(c:1,d:1):1);', '((a:1,c:1):5,(b:1,d:1):5);') AS cophenetic_distance"
                ),
                description=(
                    "Compare two trees by the tip-to-tip distances they imply, rather than by "
                    "their splits. It answers the question that actually matters downstream — do "
                    "these trees place the taxa at the same distances from each other? — and so "
                    "reacts to the long branches in the second tree that robinson_foulds ignores."
                ),
            )
        ]
        tags = {
            "vgi.category": "comparison",
            "vgi.doc_llm": (
                "Scalar function returning the cophenetic distance between two Newick trees as a `DOUBLE`: "
                "a distance derived from how much their tip-to-tip path-length distance matrices disagree "
                "over the shared taxa (0 means the trees imply identical pairwise distances, larger means "
                "more different). Unlike Robinson-Foulds it is branch-length sensitive. Pass two `VARCHAR` "
                "Newick columns; NULL or unparseable input returns NULL."
            ),
            "vgi.doc_md": (
                "**cophenetic_distance** — disagreement of two trees' tip-to-tip distances.\n\n"
                "- Inputs: two `VARCHAR` Newick columns (shared taxa)\n"
                "- Returns: `DOUBLE` distance (0 = identical pairwise distances); branch-length sensitive\n"
                "- NULL for NULL/unparseable input"
            ),
        }

    @classmethod
    def compute(
        cls,
        newick1: Annotated[pa.StringArray, Param(doc="First Newick tree")],
        newick2: Annotated[pa.StringArray, Param(doc="Second Newick tree")],
    ) -> Annotated[pa.DoubleArray, Returns(pa.float64())]:
        """Return the cophenetic distance per row (NULL on NULL/invalid input)."""
        return _compare(newick1, newick2, lambda a, b: a.compare_cophenet(b))


TREE_FUNCTIONS: list[type] = [
    *_BUILDER_FUNCTIONS,
    TipCount,
    TotalBranchLength,
    TreeHeight,
    RobinsonFoulds,
    WeightedRobinsonFoulds,
    CopheneticDistance,
]
