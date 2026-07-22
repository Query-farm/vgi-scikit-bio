"""Phylogenetic diversity and rarefaction over a long feature table + a tree.

These functions need a phylogenetic tree alongside the feature counts. Because a
table function gets only one subquery slot (the feature table), the tree is
passed as a Newick string argument (``tree := '...'``) whose tip names must match
the feature ids:

* ``faith_pd`` -- Faith's phylogenetic diversity per sample (alpha).
* ``unifrac`` -- weighted/unweighted UniFrac distance matrix between samples
  (beta), emitted long as ``(id_1, id_2, distance)``.
* ``subsample_counts`` -- rarefy each sample's counts to a fixed depth.

Counts are non-negative integers (rounded); a feature absent from the tree
raises a clear error at run time.
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

_EXAMPLE_TREE = "((f1:0.1,f2:0.2):0.3,(f3:0.15,f4:0.25):0.35);"
_EXAMPLE_TABLE = (
    "(VALUES ('s1','f1',1),('s1','f2',1),('s2','f3',1),('s2','f4',1),('s3','f1',1),('s3','f3',1)) "
    "AS t(sample_id, feature_id, count)"
)


def _resolve_triple(schema: pa.Schema, sample: str, feature: str, count: str) -> tuple[str, str, str]:
    """Resolve the (sample, feature, count) column names, defaulting to positional 0/1/2."""
    names = list(schema.names)
    s = sample or (names[0] if len(names) > 0 else "")
    f = feature or (names[1] if len(names) > 1 else "")
    c = count or (names[2] if len(names) > 2 else "")
    for label, col in (("sample", s), ("feature", f), ("count", c)):
        if not col or col not in names:
            raise ValueError(f"{label} column {col!r} not found in input; columns: {', '.join(names)}")
    return s, f, c


def _pivot_int(table: pa.Table, s_col: str, f_col: str, c_col: str) -> tuple[list[str], list[str], Any]:
    """Pivot a long feature table to (sample_ids, feature_ids, integer matrix)."""
    samples = [str(x) for x in table.column(s_col).to_pylist()]
    features = [str(x) for x in table.column(f_col).to_pylist()]
    counts = table.column(c_col).to_pylist()
    sample_ids = sorted(set(samples))
    feature_ids = sorted(set(features))
    s_index = {s: i for i, s in enumerate(sample_ids)}
    f_index = {f: j for j, f in enumerate(feature_ids)}
    mat = np.zeros((len(sample_ids), len(feature_ids)), dtype=np.float64)
    for s, f, c in zip(samples, features, counts, strict=True):
        if c is not None:
            mat[s_index[s], f_index[f]] += float(c)
    return sample_ids, feature_ids, np.rint(mat).astype(np.int64)


def _read_tree(newick: str) -> Any:
    """Parse a Newick tree string, raising a clear error if empty/invalid."""
    import warnings

    from skbio import TreeNode

    if not newick or not newick.strip():
        raise ValueError("tree := '<newick>' is required (a Newick string whose tips match the feature ids)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TreeNode.read([newick.strip()], format="newick")


# ===========================================================================
# faith_pd (phylogenetic alpha diversity)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class _FaithArgs:
    data: Annotated[TableInput, Arg(0, doc="One row per (sample_id, feature_id, count).")]
    tree: Annotated[str, Arg("tree", default="", doc="Newick tree whose tips match the feature ids.")]
    sample: Annotated[str, Arg("sample", default="", doc="Sample-id column (defaults to the first column).")]
    feature: Annotated[str, Arg("feature", default="", doc="Feature-id column (defaults to the second column).")]
    count: Annotated[str, Arg("count", default="", doc="Abundance-count column (defaults to the third column).")]


class FaithPd(SinkBuffer[_FaithArgs, DrainState]):
    """Faith's phylogenetic diversity per sample, given a tree."""

    FunctionArguments: ClassVar[type] = _FaithArgs

    class Meta:
        """VGI metadata for the faith_pd function."""

        name = "faith_pd"
        description = "Faith's phylogenetic diversity per sample (needs a tree)"
        categories = ["diversity", "phylogenetic", "alpha"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT sample_id, round(faith_pd, 4) AS faith_pd FROM skbio.diversity.faith_pd("
                    f"(SELECT * FROM {_EXAMPLE_TABLE}), tree := '{_EXAMPLE_TREE}') "
                    "ORDER BY faith_pd DESC"
                ),
                description=(
                    "Rank samples by how much evolutionary history they contain, not just how "
                    "many features they have. s3 holds one feature from each clade and so spans "
                    "more of the tree than s1 or s2, whose two features are siblings -- the "
                    "distinction plain richness cannot make and the reason to pass a tree at all."
                ),
            )
        ]
        tags = {
            "vgi.category": "phylogenetic",
            "vgi.result_columns_schema": result_columns_schema(
                [
                    ("sample_id", "VARCHAR", "Sample id."),
                    ("faith_pd", "DOUBLE", "Total branch length of the features present in the sample."),
                ]
            ),
            "vgi.doc_llm": (
                "Table function computing Faith's phylogenetic diversity (PD) of each sample: the total "
                "branch length of the sub-tree spanning the features present in that sample. The table arg "
                "is `(SELECT sample_id, feature_id, count FROM ...)` (columns default to positional 1/2/3) "
                "and `tree :=` is a Newick tree whose tip names are the feature ids. Unlike the non-"
                "phylogenetic richness metrics, PD credits samples for containing evolutionarily distinct "
                "features. Returns one `(sample_id, faith_pd)` row per sample; a feature missing from the "
                "tree raises an error."
            ),
            "vgi.doc_md": (
                "**faith_pd** — Faith's phylogenetic diversity per sample.\n\n"
                "- Table arg: `(SELECT sample_id, feature_id, count FROM ...)` (positional 1/2/3)\n"
                "- `tree :=` — a Newick tree whose tips are the feature ids\n"
                "- Returns `(sample_id, faith_pd)`: branch length spanned by each sample's features\n"
                "- Rewards evolutionarily distinct features (unlike plain richness)"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[_FaithArgs]) -> BindResponse:
        """Validate columns and fix the (sample_id, faith_pd) output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        _resolve_triple(input_schema, a.sample, a.feature, a.count)
        fields = [
            sfield("sample_id", pa.string(), "Sample id.", nullable=False),
            sfield("faith_pd", pa.float64(), "Faith's phylogenetic diversity.", nullable=False),
        ]
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_FaithArgs]) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_FaithArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Compute Faith's PD per sample and emit, once."""
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
    def encode(cls, table: pa.Table, args: _FaithArgs) -> dict[str, list[Any]]:
        """Compute Faith's PD per sample and return the result columns."""
        from skbio.diversity import alpha_diversity

        s_col, f_col, c_col = _resolve_triple(table.schema, args.sample, args.feature, args.count)
        sample_ids, feature_ids, matrix = _pivot_int(table, s_col, f_col, c_col)
        tree = _read_tree(args.tree)
        result = alpha_diversity("faith_pd", matrix, ids=sample_ids, taxa=feature_ids, tree=tree)
        return {"sample_id": sample_ids, "faith_pd": [float(x) for x in result.to_numpy()]}


# ===========================================================================
# unifrac (phylogenetic beta diversity)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class _UnifracArgs:
    data: Annotated[TableInput, Arg(0, doc="One row per (sample_id, feature_id, count).")]
    tree: Annotated[str, Arg("tree", default="", doc="Newick tree whose tips match the feature ids.")]
    weighted: Annotated[bool, Arg("weighted", default=False, doc="Weight by abundance (weighted UniFrac).")]
    sample: Annotated[str, Arg("sample", default="", doc="Sample-id column (defaults to the first column).")]
    feature: Annotated[str, Arg("feature", default="", doc="Feature-id column (defaults to the second column).")]
    count: Annotated[str, Arg("count", default="", doc="Abundance-count column (defaults to the third column).")]


class Unifrac(SinkBuffer[_UnifracArgs, DrainState]):
    """UniFrac phylogenetic distance matrix between samples, given a tree."""

    FunctionArguments: ClassVar[type] = _UnifracArgs

    class Meta:
        """VGI metadata for the unifrac function."""

        name = "unifrac"
        description = "UniFrac phylogenetic distance matrix between samples (needs a tree)"
        categories = ["diversity", "phylogenetic", "beta"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT id_1, id_2, round(distance, 4) AS distance FROM skbio.diversity.unifrac("
                    f"(SELECT * FROM {_EXAMPLE_TABLE}), tree := '{_EXAMPLE_TREE}') "
                    "WHERE id_1 < id_2 ORDER BY distance"
                ),
                description=(
                    "Order sample pairs by how much evolutionary history they fail to share. "
                    "Unlike Bray-Curtis, two samples holding different-but-closely-related "
                    "features come out close together here, which is why UniFrac is the default "
                    "beta metric for 16S data. Keep the full matrix (drop the filter) to feed "
                    "pcoa or permanova."
                ),
            )
        ]
        tags = {
            "vgi.category": "phylogenetic",
            "vgi.result_columns_schema": result_columns_schema(
                [
                    ("id_1", "VARCHAR", "First sample id."),
                    ("id_2", "VARCHAR", "Second sample id."),
                    ("distance", "DOUBLE", "UniFrac distance between the two samples."),
                ]
            ),
            "vgi.doc_llm": (
                "Table function computing the UniFrac phylogenetic distance matrix between samples and "
                "emitting it long: one row per ordered sample pair. The table arg is "
                "`(SELECT sample_id, feature_id, count FROM ...)` (columns default to positional 1/2/3) and "
                "`tree :=` is a Newick tree whose tips are the feature ids. UniFrac measures how much "
                "evolutionary history two communities do NOT share; `weighted := true` weights branches by "
                "abundance (weighted UniFrac), otherwise presence/absence (unweighted, the default). "
                "Returns `(id_1, id_2, distance)` for the full matrix — feed it into `skbio.stats.pcoa` or "
                "`skbio.stats.permanova`."
            ),
            "vgi.doc_md": (
                "**UniFrac** — phylogenetic between-sample distance matrix.\n\n"
                "- Table arg: `(SELECT sample_id, feature_id, count FROM ...)` (positional 1/2/3)\n"
                "- `tree :=` — a Newick tree whose tips are the feature ids; `weighted :=` for weighted "
                "UniFrac (default unweighted)\n"
                "- Returns the full matrix long: `id_1`, `id_2`, `distance` — feed to `pcoa`/`permanova`\n"
                "- Measures unshared evolutionary history between communities"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[_UnifracArgs]) -> BindResponse:
        """Validate columns and fix the long distance-matrix output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        _resolve_triple(input_schema, a.sample, a.feature, a.count)
        fields = [
            sfield("id_1", pa.string(), "First sample id.", nullable=False),
            sfield("id_2", pa.string(), "Second sample id.", nullable=False),
            sfield("distance", pa.float64(), "UniFrac distance.", nullable=False),
        ]
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_UnifracArgs]) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_UnifracArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Compute the UniFrac matrix and emit it long, once."""
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
    def encode(cls, table: pa.Table, args: _UnifracArgs) -> dict[str, list[Any]]:
        """Compute the UniFrac distance matrix and return the long-format columns."""
        from skbio.diversity import beta_diversity

        s_col, f_col, c_col = _resolve_triple(table.schema, args.sample, args.feature, args.count)
        sample_ids, feature_ids, matrix = _pivot_int(table, s_col, f_col, c_col)
        tree = _read_tree(args.tree)
        metric = "weighted_unifrac" if args.weighted else "unweighted_unifrac"
        dm = beta_diversity(metric, matrix, ids=sample_ids, taxa=feature_ids, tree=tree)
        data = dm.data
        id1: list[str] = []
        id2: list[str] = []
        dist: list[float] = []
        for i, si in enumerate(sample_ids):
            for j, sj in enumerate(sample_ids):
                id1.append(si)
                id2.append(sj)
                dist.append(float(data[i, j]))
        return {"id_1": id1, "id_2": id2, "distance": dist}


# ===========================================================================
# subsample_counts (rarefaction)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class _SubsampleArgs:
    data: Annotated[TableInput, Arg(0, doc="One row per (sample_id, feature_id, count).")]
    depth: Annotated[int, Arg("depth", default=0, doc="Number of counts to draw per sample (rarefaction depth).")]
    with_replacement: Annotated[bool, Arg("with_replacement", default=False, doc="Sample with replacement.")]
    seed: Annotated[int, Arg("seed", default=0, doc="Random seed (fixed for reproducibility).")]
    sample: Annotated[str, Arg("sample", default="", doc="Sample-id column (defaults to the first column).")]
    feature: Annotated[str, Arg("feature", default="", doc="Feature-id column (defaults to the second column).")]
    count: Annotated[str, Arg("count", default="", doc="Abundance-count column (defaults to the third column).")]


class SubsampleCounts(SinkBuffer[_SubsampleArgs, DrainState]):
    """Rarefy each sample's counts to a fixed depth, emitted long."""

    FunctionArguments: ClassVar[type] = _SubsampleArgs

    class Meta:
        """VGI metadata for the subsample_counts function."""

        name = "subsample_counts"
        description = "Rarefy each sample's feature counts to a fixed depth (long output)"
        categories = ["diversity", "preprocessing"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT sample_id, feature_id, count FROM skbio.diversity.subsample_counts("
                    "(SELECT * FROM "
                    "(VALUES ('s1','a',4),('s1','b',2),('s1','c',6),('s2','a',10),('s2','b',5),('s2','c',5)) "
                    "AS t(sample_id, feature_id, count)), depth := 8) "
                    "ORDER BY sample_id, feature_id"
                ),
                description=(
                    "Level two samples sequenced to different depths (12 and 20 counts) down to a "
                    "common 8, so a later richness or diversity comparison measures biology "
                    "rather than sequencing effort. Every sample's counts now sum to 8; run this "
                    "before the alpha metrics, which are all depth-sensitive."
                ),
            )
        ]
        tags = {
            "vgi.category": "preprocessing",
            "vgi.result_columns_schema": result_columns_schema(
                [
                    ("sample_id", "VARCHAR", "Sample id."),
                    ("feature_id", "VARCHAR", "Feature id."),
                    ("count", "BIGINT", "Rarefied count (sums to the requested depth per sample)."),
                ]
            ),
            "vgi.doc_llm": (
                "Table function performing rarefaction: it subsamples each sample's feature counts down to "
                "a fixed depth (`depth :=`) so that every sample has the same total count — the standard "
                "way to remove sequencing-depth bias before comparing diversity. The table arg is "
                "`(SELECT sample_id, feature_id, count FROM ...)` (columns default to positional 1/2/3). "
                "Samples whose total is below `depth` are dropped. Sampling is without replacement by "
                "default (`with_replacement :=` to change) and uses a fixed `seed :=` for reproducibility. "
                "Returns the rarefied `(sample_id, feature_id, count)` table."
            ),
            "vgi.doc_md": (
                "**subsample_counts** — rarefy each sample to a fixed depth.\n\n"
                "- Table arg: `(SELECT sample_id, feature_id, count FROM ...)` (positional 1/2/3)\n"
                "- `depth :=` — counts to draw per sample; samples below it are dropped\n"
                "- `with_replacement :=` (default false), `seed :=` (fixed for reproducibility)\n"
                "- Returns the rarefied `(sample_id, feature_id, count)` table (removes depth bias)"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[_SubsampleArgs]) -> BindResponse:
        """Validate columns/depth and fix the long rarefied output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        _resolve_triple(input_schema, a.sample, a.feature, a.count)
        if a.depth < 1:
            raise ValueError(f"depth must be >= 1 (got {a.depth})")
        fields = [
            sfield("sample_id", pa.string(), "Sample id.", nullable=False),
            sfield("feature_id", pa.string(), "Feature id.", nullable=False),
            sfield("count", pa.int64(), "Rarefied count.", nullable=False),
        ]
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[_SubsampleArgs]
    ) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_SubsampleArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Rarefy each sample and emit the long table, once."""
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
    def encode(cls, table: pa.Table, args: _SubsampleArgs) -> dict[str, list[Any]]:
        """Rarefy each sample to the depth and return the long-format columns."""
        from skbio.stats import subsample_counts

        s_col, f_col, c_col = _resolve_triple(table.schema, args.sample, args.feature, args.count)
        sample_ids, feature_ids, matrix = _pivot_int(table, s_col, f_col, c_col)
        s_out: list[str] = []
        f_out: list[str] = []
        c_out: list[int] = []
        for i, sid in enumerate(sample_ids):
            if int(matrix[i].sum()) < args.depth:
                continue  # too few counts to rarefy to this depth
            rarefied = subsample_counts(matrix[i], args.depth, replace=args.with_replacement, seed=args.seed)
            for j, fid in enumerate(feature_ids):
                s_out.append(sid)
                f_out.append(fid)
                c_out.append(int(rarefied[j]))
        return {"sample_id": s_out, "feature_id": f_out, "count": c_out}


PHYLO_FUNCTIONS: list[type] = [FaithPd, Unifrac, SubsampleCounts]
