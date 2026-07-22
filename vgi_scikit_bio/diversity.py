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
from vgi.arguments import Arg, ConstParam, Param, Returns, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams, ProcessParams
from vgi_rpc import ArrowSerializableDataclass
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .schema_utils import field as sfield
from .schema_utils import result_columns_schema

# ===========================================================================
# Alpha diversity (aggregates over a single count column)
# ===========================================================================


@dataclass(kw_only=True)
class CountState(ArrowSerializableDataclass):
    """Buffered abundance counts for one sample (group), plus the diversity order q.

    Attributes:
        counts: The buffered abundance values for the sample.
        order: The diversity order q, for the parameterized metrics (else unused).
    """

    counts: list[float] = field(default_factory=list)
    order: float = 1.0


class _AlphaScalar(AggregateFunction[CountState]):
    """Buffer a sample's feature counts, then score one scalar alpha-diversity metric.

    Subclasses set ``METRIC`` (a ``skbio.diversity.alpha`` callable) and a
    ``Meta``; most are generated from ``_SCALAR_SPECS`` by ``_make_alpha``.
    """

    METRIC: ClassVar[Callable[..., Any]]

    @classmethod
    def initial_state(cls, params: ProcessParams[None]) -> CountState:
        """Return the empty per-sample count buffer."""
        return CountState()

    @classmethod
    def update(
        cls,
        states: dict[int, CountState],
        group_ids: pa.Int64Array,
        abundance: Annotated[pa.DoubleArray, Param(doc="Abundance value for one (sample, feature) cell")],
    ) -> None:
        """Accumulate this batch's non-NULL counts into each sample's state."""
        batch: dict[int, list[float]] = {}
        for g, c in zip(group_ids.to_pylist(), abundance.to_pylist(), strict=False):
            if c is None:
                continue
            batch.setdefault(g, []).append(c)
        for g, cs in batch.items():
            s = states[g]
            states[g] = CountState(counts=s.counts + cs, order=s.order)

    @classmethod
    def combine(cls, source: CountState, target: CountState, params: ProcessParams[None]) -> CountState:
        """Merge two partial count buffers for the same sample."""
        return CountState(counts=source.counts + target.counts, order=source.order or target.order)

    @classmethod
    def _score(cls, counts: npt.NDArray[np.int64], state: CountState) -> Any:
        """Score one sample's integer counts (overridden by parameterized/list metrics)."""
        return float(cls.METRIC(counts))

    @classmethod
    def finalize(
        cls,
        group_ids: pa.Int64Array,
        states: dict[int, CountState],
        params: ProcessParams[None],
    ) -> Annotated[pa.RecordBatch, Returns(pa.float64())]:
        """Score each sample's buffered counts, emitting NULL for empty/failing samples."""
        results: list[Any] = []
        for gid in group_ids:
            s = states.get(gid.as_py())
            if s is None or not s.counts:
                results.append(None)
                continue
            counts = np.rint(np.asarray(s.counts, dtype=np.float64)).astype(np.int64)
            try:
                results.append(cls._score(counts, s))
            except Exception:
                results.append(None)
        return pa.record_batch({"result": pa.array(results, type=cls._result_type())})

    @classmethod
    def _result_type(cls) -> pa.DataType:
        """Arrow type of the aggregate result (float64 for scalars)."""
        return pa.float64()


class _AlphaOrder(_AlphaScalar):
    """Alpha metric parameterized by a diversity order ``q`` (hill/renyi/tsallis)."""

    @classmethod
    def update(  # type: ignore[override]  # framework reads each aggregate's own update signature
        cls,
        states: dict[int, CountState],
        group_ids: pa.Int64Array,
        abundance: Annotated[pa.DoubleArray, Param(doc="Abundance value for one (sample, feature) cell")],
        q: Annotated[float, ConstParam(doc="Diversity order (0 = richness, 1 = Shannon-like, 2 = Simpson-like)")],
    ) -> None:
        """Accumulate counts and record the requested diversity order q per sample."""
        batch: dict[int, list[float]] = {}
        for g, c in zip(group_ids.to_pylist(), abundance.to_pylist(), strict=False):
            if c is None:
                continue
            batch.setdefault(g, []).append(c)
        for g, cs in batch.items():
            s = states[g]
            states[g] = CountState(counts=s.counts + cs, order=q)

    @classmethod
    def _score(cls, counts: npt.NDArray[np.int64], state: CountState) -> Any:
        """Score at the sample's recorded order q."""
        return float(cls.METRIC(counts, order=state.order))


class _AlphaList(_AlphaScalar):
    """Alpha metric that returns a fixed-length ``DOUBLE[]`` (confidence intervals, osd)."""

    @classmethod
    def _score(cls, counts: npt.NDArray[np.int64], state: CountState) -> Any:
        """Return the metric's tuple/array result as a list of floats."""
        return [float(v) for v in cls.METRIC(counts)]

    @classmethod
    def _result_type(cls) -> pa.DataType:
        """The result is a list of doubles."""
        return pa.list_(pa.float64())

    @classmethod
    def finalize(
        cls,
        group_ids: pa.Int64Array,
        states: dict[int, CountState],
        params: ProcessParams[None],
    ) -> Annotated[pa.RecordBatch, Returns(pa.list_(pa.float64()))]:
        """List-typed finalize (shares scoring with the scalar base)."""
        return super().finalize(group_ids, states, params)


def _alpha_example(name: str, blurb: str, *, order: bool = False) -> list[FunctionExample]:
    """Build a one-entry example list for an alpha metric named ``name``.

    Args:
        name: The metric's machine name.
        blurb: One sentence saying what the metric measures (reused in the docs).
        order: True for the metrics parameterized by a diversity order ``q``.

    Returns:
        A one-entry example list for the metric's ``Meta.examples``.
    """
    order_arg = ", q := 1" if order else ""
    order_note = (
        " The diversity order `q :=` is what tunes the metric: q = 0 counts features equally, "
        "higher q weights the abundant ones more."
        if order
        else ""
    )
    return [
        FunctionExample(
            sql=(
                f"SELECT sample_id, skbio.diversity.{name}(count{order_arg}) AS {name} FROM "
                "(VALUES (1,'a',4),(1,'b',2),(1,'c',1),(2,'a',1),(2,'b',9)) AS t(sample_id, feature_id, count) "
                "GROUP BY sample_id ORDER BY sample_id"
            ),
            description=(
                f"Score two communities with {name} in a single GROUP BY: sample 1 spreads 7 counts "
                f"over three features, sample 2 puts 9 of its 10 into one, and the metric is what "
                f"separates them. {blurb} Being an aggregate rather than a table function is the "
                f"point — one query scores every sample in the table."
                f"{order_note}"
            ),
        )
    ]


def _alpha_doc(name: str, blurb: str, *, returns: str = "a single `DOUBLE` per sample") -> dict[str, str]:
    """Build the doc tags for an alpha metric named ``name``."""
    return {
        "vgi.doc_llm": (
            f"Aggregate computing the **{name}** alpha-diversity metric over one sample's feature abundance "
            f"values. {blurb} Pass the `count` column of a long feature table and `GROUP BY` the sample "
            f"id, so each group yields one result; NULL rows are skipped and values are rounded to "
            f"integers. Returns {returns}."
        ),
        "vgi.doc_md": (
            f"**{name}** — {blurb}\n\n"
            "- Input: the `count` column of a long `(sample, feature, count)` table\n"
            "- Use `GROUP BY sample_id` to get one result per sample\n"
            f"- Returns {returns}; NULL rows skipped, values rounded to integers"
        ),
    }


def _camel(name: str) -> str:
    """Turn a snake_case metric name into a CamelCase class name."""
    return "".join(part.title() for part in name.split("_")) or "Alpha"


def _make_alpha(
    name: str,
    fn: Callable[..., Any],
    blurb: str,
    *,
    base: type = _AlphaScalar,
    categories: list[str] | None = None,
    order: bool = False,
    returns: str = "a single `DOUBLE` per sample",
) -> type:
    """Generate an alpha-diversity aggregate class from a metric spec."""
    meta = type(
        "Meta",
        (),
        {
            "__doc__": f"VGI metadata for the {name} aggregate.",
            "name": name,
            "description": f"{name} alpha-diversity metric over one sample's feature counts",
            "categories": categories or ["diversity", "alpha"],
            "examples": _alpha_example(name, blurb, order=order),
            "tags": {**_alpha_doc(name, blurb, returns=returns), "vgi.category": "alpha"},
        },
    )
    return type(
        _camel(name), (base,), {"__doc__": f"{name} alpha-diversity metric.", "METRIC": staticmethod(fn), "Meta": meta}
    )


# Scalar metrics (one DOUBLE per sample), each `fn(counts)` with scikit-bio defaults.
_SCALAR_SPECS: list[tuple[str, str, str]] = [
    (
        "shannon",
        "shannon",
        "It is the entropy of the abundance distribution (natural log); higher means richer and more even.",
    ),
    (
        "simpson",
        "simpson",
        "It is the chance two random individuals belong to different features (1 minus dominance), in [0, 1].",
    ),
    (
        "inv_simpson",
        "inv_simpson",
        "It is the reciprocal of Simpson's dominance — the effective number of equally-abundant features.",
    ),
    (
        "observed_features",
        "observed_features",
        "It is plain richness: the number of features with a non-zero abundance.",
    ),
    (
        "chao1",
        "chao1",
        "It corrects observed richness with the singleton and doubleton counts to estimate unseen features.",
    ),
    (
        "pielou_evenness",
        "pielou_e",
        "It is Shannon diversity over its maximum, measuring how evenly abundance spreads, in [0, 1].",
    ),
    (
        "dominance",
        "dominance",
        "It is the chance two random individuals share a feature (Simpson's D); higher means a few dominate.",
    ),
    (
        "ace",
        "ace",
        "It is the abundance-based coverage estimator of total richness, weighting the contribution of rare features.",
    ),
    (
        "berger_parker_d",
        "berger_parker_d",
        "It is the proportional abundance of the single most abundant feature (Berger-Parker dominance).",
    ),
    (
        "brillouin_d",
        "brillouin_d",
        "It is Brillouin's index, an entropy diversity for a fully-censused (not sampled) community.",
    ),
    ("doubles", "doubles", "It is the number of features observed exactly twice (doubletons)."),
    ("singles", "singles", "It is the number of features observed exactly once (singletons)."),
    (
        "enspie",
        "enspie",
        "It is the effective species count from the probability of interspecific encounter (= inverse Simpson).",
    ),
    (
        "fisher_alpha",
        "fisher_alpha",
        "It is Fisher's alpha, the parameter of the log-series model fitted to the abundance distribution.",
    ),
    (
        "gini_index",
        "gini_index",
        "It is the Gini coefficient of the abundance distribution — the inequality of abundances, in [0, 1].",
    ),
    (
        "goods_coverage",
        "goods_coverage",
        "It is Good's coverage estimate: the estimated proportion of the community that has actually been observed.",
    ),
    ("heip_evenness", "heip_e", "It is Heip's evenness index, a richness-corrected evenness in [0, 1]."),
    (
        "kempton_taylor_q",
        "kempton_taylor_q",
        "It is the Kempton-Taylor Q index, the slope of the ranked log-abundance curve between its quartiles.",
    ),
    ("margalef", "margalef", "It is Margalef's richness index: richness scaled by the natural log of the total count."),
    (
        "mcintosh_d",
        "mcintosh_d",
        "It is McIntosh's dominance index, based on the Euclidean norm of the abundance vector.",
    ),
    ("mcintosh_e", "mcintosh_e", "It is McIntosh's evenness index, in [0, 1]."),
    (
        "menhinick",
        "menhinick",
        "It is Menhinick's richness index: richness divided by the square root of the total count.",
    ),
    (
        "robbins",
        "robbins",
        "It is the Robbins estimator of the probability that the next individual sampled belongs to an unseen feature.",
    ),
    (
        "simpson_d",
        "simpson_d",
        "It is Simpson's dominance index D, the probability that two individuals share a feature.",
    ),
    (
        "simpson_e",
        "simpson_e",
        "It is Simpson's evenness, the inverse-Simpson diversity divided by richness, in [0, 1].",
    ),
    (
        "strong",
        "strong",
        "It is Strong's dominance index (DW): the peak departure of the cumulative-abundance curve from evenness.",
    ),
]

# Parameterized metrics taking a diversity order q (required `order :=`).
_ORDER_SPECS: list[tuple[str, str, str]] = [
    (
        "hill",
        "hill",
        "It is the Hill number of order q (effective feature count): q=0 richness, q~1 exp(Shannon), q=2 inv-Simpson.",
    ),
    (
        "renyi",
        "renyi",
        "It is the Renyi entropy of order q, a family of diversity indices generalising richness/Shannon/Simpson.",
    ),
    ("tsallis", "tsallis", "It is the Tsallis entropy of order q, a non-extensive generalisation of Shannon entropy."),
]

# Metrics returning a fixed-length DOUBLE[] (confidence intervals, osd triple).
_LIST_SPECS: list[tuple[str, str, str, str]] = [
    (
        "chao1_ci",
        "chao1_ci",
        "It is the confidence interval around the Chao1 richness estimate, as [lower, upper].",
        "a 2-element `DOUBLE[]` `[lower, upper]` per sample",
    ),
    (
        "esty_ci",
        "esty_ci",
        "It is Esty's confidence interval for the community coverage, as [lower, upper].",
        "a 2-element `DOUBLE[]` `[lower, upper]` per sample",
    ),
    (
        "osd",
        "osd",
        "It is the (observed richness, singletons, doubletons) triple used by richness estimators.",
        "a 3-element `DOUBLE[]` `[observed, singles, doubles]` per sample",
    ),
]

ALPHA_FUNCTIONS: list[type] = [
    *[_make_alpha(name, getattr(skalpha, attr), blurb) for name, attr, blurb in _SCALAR_SPECS],
    *[
        _make_alpha(name, getattr(skalpha, attr), blurb, base=_AlphaOrder, order=True)
        for name, attr, blurb in _ORDER_SPECS
    ],
    *[
        _make_alpha(name, getattr(skalpha, attr), blurb, base=_AlphaList, returns=returns)
        for name, attr, blurb, returns in _LIST_SPECS
    ],
]


# ===========================================================================
# Beta diversity (distance matrix as a table function)
# ===========================================================================


def _nonphylo_beta_metrics() -> set[str]:
    """Every scikit-bio beta metric that works on a plain counts matrix (excludes UniFrac).

    The two UniFrac metrics are phylogenetic — they need a tree and are exposed
    separately in ``vgi_scikit_bio.phylo`` — so they are filtered out here.
    """
    from skbio.diversity import get_beta_diversity_metrics

    return {m for m in get_beta_diversity_metrics() if "unifrac" not in m}


_BETA_METRICS = _nonphylo_beta_metrics()


@dataclass(slots=True, frozen=True)
class _BetaArgs:
    data: Annotated[TableInput, Arg(0, doc="One row per (sample_id, feature_id, count).")]
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
                    "SELECT id_1, id_2, round(distance, 4) AS distance FROM "
                    "skbio.diversity.beta_diversity((SELECT * FROM "
                    "(VALUES (1,'a',4),(1,'b',2),(2,'a',1),(2,'b',9),(3,'a',0),(3,'b',5)) "
                    "AS t(sample_id, feature_id, count)), metric := 'braycurtis') "
                    "WHERE id_1 < id_2 ORDER BY distance"
                ),
                description=(
                    "Rank the sample pairs from most to least similar under Bray-Curtis. The "
                    "id_1 < id_2 filter keeps one row per unordered pair (the function emits the "
                    "full square matrix, diagonal included) — drop the filter when feeding pcoa, "
                    "permanova, or a tree builder, all of which want the whole matrix."
                ),
            )
        ]
        tags = {
            "vgi.result_columns_schema": result_columns_schema(
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
