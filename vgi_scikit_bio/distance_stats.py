"""Distance-matrix hypothesis tests: PERMANOVA, ANOSIM, and the Mantel test.

All three are single-row table functions over a long distance matrix:

* ``permanova`` / ``anosim`` test whether a grouping explains between-sample
  distances. Their input carries the grouping as a fourth column giving the
  group label of ``id_1``: ``(id_1, id_2, distance, group)``.
* ``mantel`` tests the correlation between two distance matrices supplied as two
  distance columns over the same id pairs: ``(id_1, id_2, distance_x, distance_y)``.

Each buffers its input, runs the test once, and returns a single row of the test
statistic, p-value, and permutation count.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pandas as pd
import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .distance_utils import distance_matrix_from_long
from .schema_utils import columns_md_rows
from .schema_utils import field as sfield


def _require(schema: pa.Schema, col: str, label: str) -> str:
    """Return ``col`` if present in the schema, else raise a clear error."""
    if not col or col not in schema.names:
        raise ValueError(f"{label} column {col!r} not found in input; columns: {', '.join(schema.names)}")
    return col


# ===========================================================================
# PERMANOVA / ANOSIM (grouping in a fourth column)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class _GroupedTestArgs:
    data: Annotated[
        TableInput, Arg(0, doc="A long distance matrix with a group column: (id_1, id_2, distance, group).")
    ]
    id_1: Annotated[str, Arg("id_1", default="", doc="First-id column (defaults to the first column).")]
    id_2: Annotated[str, Arg("id_2", default="", doc="Second-id column (defaults to the second column).")]
    distance: Annotated[str, Arg("distance", default="", doc="Distance column (defaults to the third column).")]
    group: Annotated[str, Arg("group", default="", doc="Group label of id_1 (defaults to the fourth column).")]
    permutations: Annotated[int, Arg("permutations", default=999, doc="Number of permutations for the p-value.")]


def _resolve_grouped(schema: pa.Schema, args: _GroupedTestArgs) -> tuple[str, str, str, str]:
    """Resolve (id_1, id_2, distance, group) column names, defaulting to positional 0-3."""
    names = list(schema.names)
    a = args.id_1 or (names[0] if len(names) > 0 else "")
    b = args.id_2 or (names[1] if len(names) > 1 else "")
    d = args.distance or (names[2] if len(names) > 2 else "")
    g = args.group or (names[3] if len(names) > 3 else "")
    return (
        _require(schema, a, "id_1"),
        _require(schema, b, "id_2"),
        _require(schema, d, "distance"),
        _require(schema, g, "group"),
    )


def _grouping_series(table: pa.Table, id1_col: str, group_col: str, ids: list[str]) -> pd.Series:
    """Build a per-sample grouping Series aligned to ``ids`` from the (id_1, group) rows."""
    mapping: dict[str, Any] = {}
    for sample, grp in zip(table.column(id1_col).to_pylist(), table.column(group_col).to_pylist(), strict=True):
        if sample is None:
            continue
        mapping.setdefault(str(sample), grp)
    missing = [i for i in ids if mapping.get(i) is None]
    if missing:
        raise ValueError(f"no group label for sample(s): {', '.join(missing[:5])}")
    return pd.Series([mapping[i] for i in ids], index=ids, name="group")


_GROUPED_RESULT_FIELDS = [
    sfield("method", pa.string(), "Test name (PERMANOVA or ANOSIM).", nullable=False),
    sfield("test_statistic", pa.float64(), "The test statistic (pseudo-F for PERMANOVA, R for ANOSIM).", nullable=True),
    sfield("p_value", pa.float64(), "Permutation p-value.", nullable=True),
    sfield("sample_size", pa.int64(), "Number of samples.", nullable=False),
    sfield("number_of_groups", pa.int64(), "Number of distinct groups.", nullable=False),
    sfield("permutations", pa.int64(), "Number of permutations used.", nullable=False),
]


class _GroupedTest(SinkBuffer[_GroupedTestArgs, DrainState]):
    """Shared buffering + one-row-result plumbing for PERMANOVA/ANOSIM."""

    FunctionArguments: ClassVar[type] = _GroupedTestArgs
    TEST: ClassVar[Callable[..., Any]]

    @classmethod
    def on_bind(cls, params: BindParams[_GroupedTestArgs]) -> BindResponse:
        """Validate columns and fix the single-row result schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        _resolve_grouped(input_schema, a)
        if a.permutations < 1:
            raise ValueError(f"permutations must be >= 1 (got {a.permutations})")
        return BindResponse(output_schema=pa.schema(_GROUPED_RESULT_FIELDS))

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[_GroupedTestArgs]
    ) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_GroupedTestArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Reconstruct the matrix + grouping, run the test, and emit one result row, once."""
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
    def encode(cls, table: pa.Table, args: _GroupedTestArgs) -> dict[str, list[Any]]:
        """Run the grouped distance test and return its single-row result columns."""
        id1, id2, dist, grp = _resolve_grouped(table.schema, args)
        dm = distance_matrix_from_long(table, id1, id2, dist)
        grouping = _grouping_series(table, id1, grp, list(dm.ids))
        result = cls.TEST(dm, grouping, permutations=args.permutations)
        return {
            "method": [str(result["method name"])],
            "test_statistic": [float(result["test statistic"])],
            "p_value": [None if result["p-value"] is None else float(result["p-value"])],
            "sample_size": [int(result["sample size"])],
            "number_of_groups": [int(result["number of groups"])],
            "permutations": [int(result["number of permutations"])],
        }


def _grouped_doc(name: str, stat: str, blurb: str) -> dict[str, str]:
    """Build the doc tags for a grouped distance test."""
    return {
        "vgi.result_columns_md": columns_md_rows(
            [
                ("method", "VARCHAR", "Test name."),
                ("test_statistic", "DOUBLE", f"The {stat} statistic."),
                ("p_value", "DOUBLE", "Permutation p-value."),
                ("sample_size", "BIGINT", "Number of samples."),
                ("number_of_groups", "BIGINT", "Number of distinct groups."),
                ("permutations", "BIGINT", "Number of permutations used."),
            ]
        ),
        "vgi.doc_llm": (
            f"Single-row table function running the **{name}** test: {blurb} The table arg is "
            "`(SELECT id_1, id_2, distance, group FROM ...)` where `group` is the group label of `id_1` "
            "(columns default to positional 1-4; override with `id_1 :=`, `id_2 :=`, `distance :=`, "
            "`group :=`), typically built by joining a group label onto a `beta_diversity` matrix. "
            "`permutations :=` sets the number of permutations for the p-value (default 999). Returns one "
            f"row with the {stat} `test_statistic`, the `p_value`, and the sample/group/permutation counts."
        ),
        "vgi.doc_md": (
            f"**{name}** — {blurb}\n\n"
            "- Table arg: `(SELECT id_1, id_2, distance, group FROM ...)` — `group` is the label of `id_1` "
            "(positional 1-4 by default)\n"
            "- `permutations :=` — permutation count for the p-value (default 999)\n"
            f"- Returns one row: `method`, `test_statistic` ({stat}), `p_value`, `sample_size`, "
            "`number_of_groups`, `permutations`\n"
            "- Build the input by joining a grouping onto a `beta_diversity` distance matrix"
        ),
    }


class Permanova(_GroupedTest):
    """PERMANOVA: test whether a grouping explains between-sample distances."""

    @staticmethod
    def TEST(dm: Any, grouping: Any, permutations: int) -> Any:
        """Run scikit-bio's permanova."""
        from skbio.stats.distance import permanova

        return permanova(dm, grouping, permutations=permutations)

    class Meta:
        """VGI metadata for the permanova function."""

        name = "permanova"
        description = "PERMANOVA test: does a grouping explain between-sample distances?"
        categories = ["stats", "distance", "hypothesis-test"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM skbio.stats.permanova((SELECT b.id_1, b.id_2, b.distance, g.grp FROM "
                    "skbio.diversity.beta_diversity((SELECT * FROM (VALUES ('s1','a',4),('s1','b',1),"
                    "('s2','a',3),('s2','b',2),('s3','a',1),('s3','b',8),('s4','a',0),('s4','b',9)) "
                    "AS t(sample_id, feature_id, count))) AS b JOIN (VALUES ('s1','x'),('s2','x'),"
                    "('s3','y'),('s4','y')) AS g(sample, grp) ON b.id_1 = g.sample))"
                ),
                description="PERMANOVA over a beta-diversity matrix with a two-group split",
            )
        ]
        tags = _grouped_doc(
            "PERMANOVA",
            "pseudo-F",
            "a permutational multivariate ANOVA that tests whether samples group by a categorical variable "
            "in distance space, comparing within- vs between-group distances.",
        )


class Anosim(_GroupedTest):
    """ANOSIM: test whether within-group distances are smaller than between-group."""

    @staticmethod
    def TEST(dm: Any, grouping: Any, permutations: int) -> Any:
        """Run scikit-bio's anosim."""
        from skbio.stats.distance import anosim

        return anosim(dm, grouping, permutations=permutations)

    class Meta:
        """VGI metadata for the anosim function."""

        name = "anosim"
        description = "ANOSIM test: are within-group distances smaller than between-group?"
        categories = ["stats", "distance", "hypothesis-test"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM skbio.stats.anosim((SELECT b.id_1, b.id_2, b.distance, g.grp FROM "
                    "skbio.diversity.beta_diversity((SELECT * FROM (VALUES ('s1','a',4),('s1','b',1),"
                    "('s2','a',3),('s2','b',2),('s3','a',1),('s3','b',8),('s4','a',0),('s4','b',9)) "
                    "AS t(sample_id, feature_id, count))) AS b JOIN (VALUES ('s1','x'),('s2','x'),"
                    "('s3','y'),('s4','y')) AS g(sample, grp) ON b.id_1 = g.sample))"
                ),
                description="ANOSIM over a beta-diversity matrix with a two-group split",
            )
        ]
        tags = _grouped_doc(
            "ANOSIM",
            "R",
            "an analysis of similarities that tests whether distances within groups are smaller than "
            "distances between groups; the R statistic ranges from -1 to 1 (near 1 = strong separation).",
        )


# ===========================================================================
# Mantel test (two distance columns)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class _MantelArgs:
    data: Annotated[TableInput, Arg(0, doc="Two aligned distance matrices: (id_1, id_2, distance_x, distance_y).")]
    id_1: Annotated[str, Arg("id_1", default="", doc="First-id column (defaults to the first column).")]
    id_2: Annotated[str, Arg("id_2", default="", doc="Second-id column (defaults to the second column).")]
    distance_x: Annotated[str, Arg("distance_x", default="", doc="First distance column (defaults to the third).")]
    distance_y: Annotated[str, Arg("distance_y", default="", doc="Second distance column (defaults to the fourth).")]
    method: Annotated[str, Arg("method", default="pearson", doc="Correlation method: pearson or spearman.")]
    permutations: Annotated[int, Arg("permutations", default=999, doc="Number of permutations for the p-value.")]


class Mantel(SinkBuffer[_MantelArgs, DrainState]):
    """Mantel test: correlation between two distance matrices over the same ids."""

    FunctionArguments: ClassVar[type] = _MantelArgs

    class Meta:
        """VGI metadata for the mantel function."""

        name = "mantel"
        description = "Mantel test: correlation between two distance matrices"
        categories = ["stats", "distance", "hypothesis-test"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM skbio.stats.mantel((SELECT * FROM "
                    "(VALUES ('a','b',0.5,0.4),('a','c',0.7,0.9),('b','c',0.6,0.5)) "
                    "AS d(id_1, id_2, distance_x, distance_y)))"
                ),
                description="Mantel correlation between two distance matrices",
            )
        ]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    ("method", "VARCHAR", "Correlation method (pearson or spearman)."),
                    ("correlation", "DOUBLE", "Mantel correlation coefficient (in [-1, 1])."),
                    ("p_value", "DOUBLE", "Permutation p-value."),
                    ("n", "BIGINT", "Number of samples compared."),
                    ("permutations", "BIGINT", "Number of permutations used."),
                ]
            ),
            "vgi.doc_llm": (
                "Single-row table function running the **Mantel test**: the correlation between two "
                "distance matrices measured over the same set of samples, with a permutation p-value. The "
                "table arg is `(SELECT id_1, id_2, distance_x, distance_y FROM ...)` — the two `distance_*` "
                "columns are the corresponding entries of the two matrices for each id pair (columns "
                "default to positional 1-4; override with `id_1 :=`, `id_2 :=`, `distance_x :=`, "
                "`distance_y :=`). `method :=` picks `pearson` (default) or `spearman`, and `permutations "
                ":=` the permutation count (default 999). Returns one row: the `correlation`, `p_value`, "
                "sample count `n`, and permutation count. Use it to test whether two dissimilarities (e.g. "
                "genetic vs geographic distance) covary."
            ),
            "vgi.doc_md": (
                "**Mantel test** — correlation between two distance matrices.\n\n"
                "- Table arg: `(SELECT id_1, id_2, distance_x, distance_y FROM ...)` (positional 1-4 by "
                "default); the two `distance_*` columns are aligned entries of the two matrices\n"
                "- `method :=` — `pearson` (default) or `spearman`; `permutations :=` — p-value permutations "
                "(default 999)\n"
                "- Returns one row: `method`, `correlation`, `p_value`, `n`, `permutations`\n"
                "- Tests whether two dissimilarities (e.g. genetic vs geographic) covary"
            ),
        }

    @classmethod
    def _resolve(cls, schema: pa.Schema, args: _MantelArgs) -> tuple[str, str, str, str]:
        """Resolve (id_1, id_2, distance_x, distance_y) column names, defaulting to positional 0-3."""
        names = list(schema.names)
        a = args.id_1 or (names[0] if len(names) > 0 else "")
        b = args.id_2 or (names[1] if len(names) > 1 else "")
        dx = args.distance_x or (names[2] if len(names) > 2 else "")
        dy = args.distance_y or (names[3] if len(names) > 3 else "")
        return (
            _require(schema, a, "id_1"),
            _require(schema, b, "id_2"),
            _require(schema, dx, "distance_x"),
            _require(schema, dy, "distance_y"),
        )

    @classmethod
    def on_bind(cls, params: BindParams[_MantelArgs]) -> BindResponse:
        """Validate columns/method and fix the single-row result schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        cls._resolve(input_schema, a)
        if a.method not in {"pearson", "spearman"}:
            raise ValueError(f"method must be 'pearson' or 'spearman' (got {a.method!r})")
        if a.permutations < 1:
            raise ValueError(f"permutations must be >= 1 (got {a.permutations})")
        fields = [
            sfield("method", pa.string(), "Correlation method.", nullable=False),
            sfield("correlation", pa.float64(), "Mantel correlation coefficient.", nullable=True),
            sfield("p_value", pa.float64(), "Permutation p-value.", nullable=True),
            sfield("n", pa.int64(), "Number of samples compared.", nullable=False),
            sfield("permutations", pa.int64(), "Number of permutations used.", nullable=False),
        ]
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_MantelArgs]) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_MantelArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Reconstruct both matrices, run the Mantel test, and emit one row, once."""
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
    def encode(cls, table: pa.Table, args: _MantelArgs) -> dict[str, list[Any]]:
        """Run the Mantel test and return its single-row result columns."""
        from skbio.stats.distance import mantel

        id1, id2, dx, dy = cls._resolve(table.schema, args)
        dm_x = distance_matrix_from_long(table, id1, id2, dx)
        dm_y = distance_matrix_from_long(table, id1, id2, dy)
        corr, p_value, n = mantel(dm_x, dm_y, method=args.method, permutations=args.permutations)
        return {
            "method": [args.method],
            "correlation": [None if corr is None else float(corr)],
            "p_value": [None if p_value is None else float(p_value)],
            "n": [int(n)],
            "permutations": [int(args.permutations)],
        }


DISTANCE_STATS_FUNCTIONS: list[type] = [Permanova, Anosim, Mantel]
