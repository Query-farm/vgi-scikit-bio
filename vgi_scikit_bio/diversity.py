"""Community-ecology diversity: alpha aggregates and beta-diversity matrices.

Two shapes, both operating on a **feature table** in long form -- one row per
(sample, feature) with an abundance ``count``:

* **Alpha diversity** -- per-sample scalars (``shannon``, ``simpson``, ``chao1``,
  ...) as SQL **aggregates** over the ``count`` column, so ``GROUP BY sample``
  gives one diversity value per sample:

      SELECT sample_id, skbio.diversity.shannon(count) AS H
      FROM feature_table GROUP BY sample_id;

* **Beta diversity** -- the between-sample distance matrix as a **table
  function** over ``(sample_id, feature_id, count)``, emitting the full matrix
  long as ``(id_1, id_2, distance)`` ready to feed ``pcoa`` / ``permanova``.

Counts are non-negative abundances; NULL rows are skipped and values are rounded
to the nearest integer (diversity metrics operate on integer abundances).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from skbio.diversity import alpha as skalpha
from vgi.aggregate_function import AggregateFunction
from vgi.arguments import Arg, Param, Returns, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams, ProcessParams
from vgi_rpc import ArrowSerializableDataclass
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .schema_utils import columns_md_rows
from .schema_utils import field as sfield

# ===========================================================================
# Alpha diversity (aggregates over a single count column)
# ===========================================================================


@dataclass(kw_only=True)
class CountState(ArrowSerializableDataclass):
    """Buffered abundance counts for one sample (group)."""

    counts: list[float] = field(default_factory=list)


class _AlphaMetric(AggregateFunction[CountState]):
    """Buffer a sample's feature counts, then score one alpha-diversity metric.

    Subclasses set ``METRIC`` (a ``skbio.diversity.alpha`` callable) and a
    ``Meta``.
    """

    METRIC: ClassVar[Callable[[npt.NDArray[np.int64]], Any]]

    @classmethod
    def initial_state(cls, params: ProcessParams[None]) -> CountState:
        """Return the empty per-sample count buffer."""
        return CountState()

    @classmethod
    def update(
        cls,
        states: dict[int, CountState],
        group_ids: pa.Int64Array,
        count: Annotated[pa.DoubleArray, Param(doc="Feature abundance count")],
    ) -> None:
        """Accumulate this batch's non-NULL counts into each sample's state."""
        batch: dict[int, list[float]] = {}
        for g, c in zip(group_ids.to_pylist(), count.to_pylist(), strict=False):
            if c is None:
                continue
            batch.setdefault(g, []).append(c)
        for g, cs in batch.items():
            s = states[g]
            states[g] = CountState(counts=s.counts + cs)

    @classmethod
    def combine(cls, source: CountState, target: CountState, params: ProcessParams[None]) -> CountState:
        """Merge two partial count buffers for the same sample."""
        return CountState(counts=source.counts + target.counts)

    @classmethod
    def finalize(
        cls,
        group_ids: pa.Int64Array,
        states: dict[int, CountState],
        params: ProcessParams[None],
    ) -> Annotated[pa.RecordBatch, Returns(pa.float64())]:
        """Score each sample's buffered counts, emitting NULL for empty/failing samples."""
        results: list[float | None] = []
        for gid in group_ids:
            s = states.get(gid.as_py())
            if s is None or not s.counts:
                results.append(None)
                continue
            counts = np.rint(np.asarray(s.counts, dtype=np.float64)).astype(np.int64)
            try:
                results.append(float(cls.METRIC(counts)))
            except Exception:
                results.append(None)
        return pa.record_batch({"result": pa.array(results, type=pa.float64())})


def _alpha_example(name: str) -> list[FunctionExample]:
    """Build a one-entry example list for an alpha metric named ``name``."""
    return [
        FunctionExample(
            sql=(
                f"SELECT sample_id, skbio.diversity.{name}(count) FROM "
                "(VALUES (1,'a',4),(1,'b',2),(1,'c',1),(2,'a',1),(2,'b',9)) AS t(sample_id, feature_id, count) "
                "GROUP BY sample_id"
            ),
            description=f"{name} per sample over a long feature table",
        )
    ]


def _alpha_doc(name: str, blurb: str) -> dict[str, str]:
    """Build the doc tags for an alpha metric named ``name``."""
    return {
        "vgi.doc_llm": (
            f"Aggregate computing the **{name}** alpha-diversity metric over a sample's feature abundance "
            f"counts. {blurb} Pass the `count` column of a long feature table and `GROUP BY` the sample "
            "id, so each group yields one diversity value; NULL counts are skipped and counts are rounded "
            "to integers. Returns a single `DOUBLE` per sample."
        ),
        "vgi.doc_md": (
            f"**{name}** — {blurb}\n\n"
            "- Input: the `count` column of a long `(sample, feature, count)` table\n"
            "- Use `GROUP BY sample_id` to get one value per sample\n"
            "- Returns a `DOUBLE`; NULL counts skipped, counts rounded to integers"
        ),
    }


class Shannon(_AlphaMetric):
    """Shannon diversity index (natural log)."""

    METRIC = staticmethod(skalpha.shannon)

    class Meta:
        """VGI metadata for the shannon aggregate."""

        name = "shannon"
        description = "Shannon diversity index of a sample's feature counts"
        categories = ["diversity", "alpha"]
        examples = _alpha_example("shannon")
        tags = _alpha_doc(
            "shannon",
            "It combines richness and evenness as the entropy of the abundance distribution (natural log); "
            "higher means more diverse.",
        )


class Simpson(_AlphaMetric):
    """Simpson diversity index (1 - dominance)."""

    METRIC = staticmethod(skalpha.simpson)

    class Meta:
        """VGI metadata for the simpson aggregate."""

        name = "simpson"
        description = "Simpson diversity index (1 - dominance) of a sample's feature counts"
        categories = ["diversity", "alpha"]
        examples = _alpha_example("simpson")
        tags = _alpha_doc(
            "simpson",
            "It is the probability that two individuals drawn at random belong to different features "
            "(1 - dominance), in [0, 1]; higher means more diverse.",
        )


class InvSimpson(_AlphaMetric):
    """Inverse Simpson index (1 / dominance)."""

    METRIC = staticmethod(skalpha.inv_simpson)

    class Meta:
        """VGI metadata for the inv_simpson aggregate."""

        name = "inv_simpson"
        description = "Inverse Simpson index (1 / dominance) of a sample's feature counts"
        categories = ["diversity", "alpha"]
        examples = _alpha_example("inv_simpson")
        tags = _alpha_doc(
            "inv_simpson",
            "It is the reciprocal of Simpson's dominance, interpretable as the effective number of equally "
            "abundant features; higher means more diverse.",
        )


class ObservedFeatures(_AlphaMetric):
    """Observed feature richness (number of non-zero features)."""

    METRIC = staticmethod(skalpha.observed_features)

    class Meta:
        """VGI metadata for the observed_features aggregate."""

        name = "observed_features"
        description = "Observed richness: number of features present in a sample"
        categories = ["diversity", "alpha", "richness"]
        examples = _alpha_example("observed_features")
        tags = _alpha_doc(
            "observed_features",
            "It is plain richness: the count of features with a non-zero abundance in the sample.",
        )


class Chao1(_AlphaMetric):
    """Chao1 estimated richness (accounts for unobserved features)."""

    METRIC = staticmethod(skalpha.chao1)

    class Meta:
        """VGI metadata for the chao1 aggregate."""

        name = "chao1"
        description = "Chao1 estimated richness of a sample (corrects for unseen features)"
        categories = ["diversity", "alpha", "richness"]
        examples = _alpha_example("chao1")
        tags = _alpha_doc(
            "chao1",
            "It estimates total richness including unobserved features by correcting observed richness "
            "with the number of singletons and doubletons; requires integer counts.",
        )


class PielouEvenness(_AlphaMetric):
    """Pielou's evenness (Shannon / log richness), in [0, 1]."""

    METRIC = staticmethod(skalpha.pielou_e)

    class Meta:
        """VGI metadata for the pielou_evenness aggregate."""

        name = "pielou_evenness"
        description = "Pielou's evenness of a sample's feature counts (in [0, 1])"
        categories = ["diversity", "alpha", "evenness"]
        examples = _alpha_example("pielou_evenness")
        tags = _alpha_doc(
            "pielou_evenness",
            "It is Shannon diversity normalised by its maximum (log of richness), so it measures how "
            "evenly abundance is spread across features, in [0, 1]; 1 is perfectly even.",
        )


class Dominance(_AlphaMetric):
    """Simpson's dominance (probability two individuals share a feature)."""

    METRIC = staticmethod(skalpha.dominance)

    class Meta:
        """VGI metadata for the dominance aggregate."""

        name = "dominance"
        description = "Simpson's dominance of a sample's feature counts (in [0, 1])"
        categories = ["diversity", "alpha"]
        examples = _alpha_example("dominance")
        tags = _alpha_doc(
            "dominance",
            "It is the probability that two individuals drawn at random belong to the same feature, in "
            "[0, 1]; higher means one/few features dominate (the complement of Simpson diversity).",
        )


ALPHA_FUNCTIONS: list[type] = [
    Shannon,
    Simpson,
    InvSimpson,
    ObservedFeatures,
    Chao1,
    PielouEvenness,
    Dominance,
]


# ===========================================================================
# Beta diversity (distance matrix as a table function)
# ===========================================================================

_BETA_METRICS = {
    "braycurtis",
    "jaccard",
    "euclidean",
    "cityblock",
    "canberra",
    "chebyshev",
    "correlation",
    "cosine",
    "hamming",
    "sqeuclidean",
}


@dataclass(slots=True, frozen=True)
class _BetaArgs:
    data: Annotated[TableInput, Arg(0, doc="A long feature table: (sample_id, feature_id, count).")]
    sample: Annotated[str, Arg("sample", default="", doc="Sample-id column (defaults to the first column).")]
    feature: Annotated[str, Arg("feature", default="", doc="Feature-id column (defaults to the second column).")]
    count: Annotated[str, Arg("count", default="", doc="Abundance-count column (defaults to the third column).")]
    metric: Annotated[str, Arg("metric", default="braycurtis", doc="Distance metric (e.g. braycurtis, jaccard).")]


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


class BetaDiversity(SinkBuffer[_BetaArgs, DrainState]):
    """Between-sample distance matrix from a long feature table, emitted long."""

    FunctionArguments: ClassVar[type] = _BetaArgs

    class Meta:
        """VGI metadata for the beta_diversity function."""

        name = "beta_diversity"
        description = "Between-sample distance matrix from a long feature table (long output)"
        categories = ["diversity", "beta", "distance"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM skbio.diversity.beta_diversity((SELECT * FROM "
                    "(VALUES (1,'a',4),(1,'b',2),(2,'a',1),(2,'b',9),(3,'a',0),(3,'b',5)) "
                    "AS t(sample_id, feature_id, count)), metric := 'braycurtis')"
                ),
                description="Bray-Curtis distances between three samples",
            )
        ]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    ("id_1", "VARCHAR", "First sample id."),
                    ("id_2", "VARCHAR", "Second sample id."),
                    ("distance", "DOUBLE", "Distance between the two samples under the chosen metric."),
                ]
            ),
            "vgi.doc_llm": (
                "Table function computing the between-sample beta-diversity distance matrix from a long "
                "feature table and emitting it long: one row per ordered sample pair. The table arg is "
                "`(SELECT sample_id, feature_id, count FROM ...)` (columns default to positional 1/2/3; "
                "override with `sample :=`, `feature :=`, `count :=`), and `metric :=` chooses the distance "
                "(default `braycurtis`; also `jaccard`, `euclidean`, `cityblock`, `canberra`, `cosine`, "
                "`hamming`, ...). The counts are pivoted to a samples x features abundance matrix (missing "
                "cells are 0) and every pairwise distance is computed. Returns `(id_1, id_2, distance)` for "
                "the full matrix (including the zero diagonal) — feed it straight into `skbio.stats.pcoa` "
                "or `skbio.stats.permanova`."
            ),
            "vgi.doc_md": (
                "**Beta diversity** — between-sample distance matrix, long output.\n\n"
                "- Table arg: `(SELECT sample_id, feature_id, count FROM ...)` (positional 1/2/3 by "
                "default; override with `sample :=`/`feature :=`/`count :=`)\n"
                "- `metric :=` — `braycurtis` (default), `jaccard`, `euclidean`, `cityblock`, `canberra`, "
                "`cosine`, `hamming`, ...\n"
                "- Returns the full matrix long: `id_1`, `id_2`, `distance` (`DOUBLE`)\n"
                "- Missing (sample, feature) cells count as 0; feed the output to `pcoa` / `permanova`"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[_BetaArgs]) -> BindResponse:
        """Validate the metric and columns, and fix the long distance-matrix output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.metric not in _BETA_METRICS:
            raise ValueError(f"unknown metric {a.metric!r}; supported: {', '.join(sorted(_BETA_METRICS))}")
        _resolve_triple(input_schema, a.sample, a.feature, a.count)
        fields = [
            sfield("id_1", pa.string(), "First sample id.", nullable=False),
            sfield("id_2", pa.string(), "Second sample id.", nullable=False),
            sfield("distance", pa.float64(), "Distance between the two samples.", nullable=False),
        ]
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_BetaArgs]) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_BetaArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Pivot the buffered counts, compute the distance matrix, and emit it long, once."""
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
    def encode(cls, table: pa.Table, args: _BetaArgs) -> dict[str, list[Any]]:
        """Pivot the long feature table and return the long distance matrix columns."""
        from skbio.diversity import beta_diversity

        s_col, f_col, c_col = _resolve_triple(table.schema, args.sample, args.feature, args.count)
        samples = [str(v) for v in table.column(s_col).to_pylist()]
        features = [str(v) for v in table.column(f_col).to_pylist()]
        counts = table.column(c_col).to_pylist()

        sample_ids = sorted(set(samples))
        feature_ids = sorted(set(features))
        s_index = {s: i for i, s in enumerate(sample_ids)}
        f_index = {f: j for j, f in enumerate(feature_ids)}
        matrix = np.zeros((len(sample_ids), len(feature_ids)), dtype=np.float64)
        for s, f, c in zip(samples, features, counts, strict=True):
            if c is None:
                continue
            matrix[s_index[s], f_index[f]] += float(c)

        dm = beta_diversity(args.metric, matrix, ids=sample_ids)
        id1: list[str] = []
        id2: list[str] = []
        dist: list[float] = []
        data = dm.data
        for i, si in enumerate(sample_ids):
            for j, sj in enumerate(sample_ids):
                id1.append(si)
                id2.append(sj)
                dist.append(float(data[i, j]))
        return {"id_1": id1, "id_2": id2, "distance": dist}


BETA_FUNCTIONS: list[type] = [BetaDiversity]
DIVERSITY_FUNCTIONS: list[type] = [*ALPHA_FUNCTIONS, *BETA_FUNCTIONS]
