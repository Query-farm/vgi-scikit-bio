"""Pairwise sequence alignment as scalar and table functions.

Two shapes:

* **score scalars** -- ``align_score_nucleotide`` / ``align_score_protein`` take
  two sequence columns and return the optimal global-alignment score per row.
* **alignment table functions** -- ``pairwise_align_nucleotide`` /
  ``pairwise_align_protein`` align a pair of sequence columns and emit the
  aligned strings, the score, and the aligned length, one row per input pair
  (``mode := 'global'`` or ``'local'``):

      SELECT id, aligned_1, aligned_2, score
      FROM skbio.alignment.pairwise_align_nucleotide((SELECT id, ref, read FROM pairs),
                                                     seq1 := 'ref', seq2 := 'read');

Sequences are upper-cased before parsing; a NULL or unparseable pair yields NULL
outputs rather than failing the query.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pyarrow as pa
from skbio import DNA, Protein
from vgi.arguments import Arg, Param, Returns, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.scalar_function import ScalarFunction
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .schema_utils import field as sfield
from .schema_utils import result_dynamic_columns_md


def _aligners(kind: str) -> tuple[Callable[..., Any], Callable[..., Any], type]:
    """Return the (global, local, sequence-type) triple for a sequence kind."""
    from skbio.alignment import (
        global_pairwise_align_nucleotide,
        global_pairwise_align_protein,
        local_pairwise_align_nucleotide,
        local_pairwise_align_protein,
    )

    if kind == "nucl":
        return global_pairwise_align_nucleotide, local_pairwise_align_nucleotide, DNA
    return global_pairwise_align_protein, local_pairwise_align_protein, Protein


def _clean(value: Any) -> str | None:
    """Normalise one input cell to an upper-cased sequence string, or None."""
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _align(kind: str, mode: str, a: str, b: str) -> tuple[str, str, float, int]:
    """Align two sequences and return (aligned_1, aligned_2, score, aligned_length)."""
    import warnings

    global_fn, local_fn, seq_type = _aligners(kind)
    fn = local_fn if mode == "local" else global_fn
    with warnings.catch_warnings():
        # These DP aligners emit a "slow pure-Python implementation" notice; we
        # keep them because they return the full aligned strings (pair_align
        # returns only a path), and callers here align modest sequences.
        warnings.simplefilter("ignore")
        msa, score, _start_end = fn(seq_type(a), seq_type(b))
    aligned = [str(s) for s in msa]
    return aligned[0], aligned[1], float(score), len(aligned[0])


# ===========================================================================
# Score scalars
# ===========================================================================


def _score_array(kind: str, seq1: pa.Array, seq2: pa.Array) -> pa.Array:
    """Global-alignment score per row for a sequence kind (NULL on NULL/invalid)."""
    out: list[float | None] = []
    for a, b in zip(seq1.to_pylist(), seq2.to_pylist(), strict=False):
        ca, cb = _clean(a), _clean(b)
        if ca is None or cb is None:
            out.append(None)
            continue
        try:
            out.append(_align(kind, "global", ca, cb)[2])
        except Exception:
            out.append(None)
    return pa.array(out, type=pa.float64())


class AlignScoreNucleotide(ScalarFunction):
    """Optimal global nucleotide alignment score between two DNA sequences."""

    class Meta:
        """VGI metadata for the align_score_nucleotide scalar."""

        name = "align_score_nucleotide"
        description = "Optimal global alignment score between two DNA sequences"
        categories = ["alignment", "nucleotide"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.alignment.align_score_nucleotide('ACTGGT', 'ACTGT') AS score",
                description=(
                    "Score how similar two DNA sequences are with a full Needleman-Wunsch "
                    "alignment, which unlike hamming_distance tolerates the differing lengths "
                    "here by opening a gap. Reach for this when you only need the number -- "
                    "pairwise_align_nucleotide when you need to see where the gap fell."
                ),
            )
        ]
        tags = {
            "vgi.category": "score",
            "vgi.doc_llm": (
                "Scalar function returning the optimal global (Needleman-Wunsch) alignment score between two "
                "DNA sequences as a `DOUBLE`. Pass two `VARCHAR` DNA columns (case-insensitive); higher "
                "scores mean more similar sequences under scikit-bio's default nucleotide scoring (match/"
                "mismatch and affine gap penalties). A NULL or non-DNA pair returns NULL. Use "
                "`pairwise_align_nucleotide` to also get the aligned strings."
            ),
            "vgi.doc_md": (
                "**align_score_nucleotide** — global alignment score of two DNA sequences.\n\n"
                "- Inputs: two `VARCHAR` DNA columns (case-insensitive)\n"
                "- Returns: a `DOUBLE` score (higher = more similar); NULL for NULL/non-DNA input\n"
                "- Use `pairwise_align_nucleotide` for the aligned sequences too"
            ),
        }

    @classmethod
    def compute(
        cls,
        seq1: Annotated[pa.StringArray, Param(doc="First DNA sequence")],
        seq2: Annotated[pa.StringArray, Param(doc="Second DNA sequence")],
    ) -> Annotated[pa.DoubleArray, Returns(pa.float64())]:
        """Return the global alignment score per row (NULL on NULL/invalid input)."""
        return _score_array("nucl", seq1, seq2)


class AlignScoreProtein(ScalarFunction):
    """Optimal global protein alignment score between two protein sequences."""

    class Meta:
        """VGI metadata for the align_score_protein scalar."""

        name = "align_score_protein"
        description = "Optimal global alignment score between two protein sequences"
        categories = ["alignment", "protein"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.alignment.align_score_protein('MRITMK', 'MRIMK') AS score",
                description=(
                    "Score two protein sequences under a substitution matrix, so chemically "
                    "similar amino acids cost less than unrelated ones -- the reason protein "
                    "similarity cannot be measured by counting identities. Use "
                    "pairwise_align_protein when the alignment itself matters."
                ),
            )
        ]
        tags = {
            "vgi.category": "score",
            "vgi.doc_llm": (
                "Scalar function returning the optimal global (Needleman-Wunsch) alignment score between two "
                "protein sequences as a `DOUBLE`, using scikit-bio's default protein substitution matrix and "
                "gap penalties. Pass two `VARCHAR` protein columns (case-insensitive); higher scores mean "
                "more similar sequences. A NULL or non-protein pair returns NULL. Use "
                "`pairwise_align_protein` to also get the aligned strings."
            ),
            "vgi.doc_md": (
                "**align_score_protein** — global alignment score of two protein sequences.\n\n"
                "- Inputs: two `VARCHAR` protein columns (case-insensitive)\n"
                "- Returns: a `DOUBLE` score (higher = more similar); NULL for NULL/non-protein input\n"
                "- Uses the default substitution matrix; `pairwise_align_protein` gives the aligned sequences"
            ),
        }

    @classmethod
    def compute(
        cls,
        seq1: Annotated[pa.StringArray, Param(doc="First protein sequence")],
        seq2: Annotated[pa.StringArray, Param(doc="Second protein sequence")],
    ) -> Annotated[pa.DoubleArray, Returns(pa.float64())]:
        """Return the global alignment score per row (NULL on NULL/invalid input)."""
        return _score_array("prot", seq1, seq2)


# ===========================================================================
# Alignment table functions (aligned strings + score)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class _AlignArgs:
    data: Annotated[TableInput, Arg(0, doc="An optional id column and two sequence columns.")]
    seq1: Annotated[str, Arg("seq1", default="", doc="First sequence column (defaults to the first non-id column).")]
    seq2: Annotated[str, Arg("seq2", default="", doc="Second sequence column (defaults to the second non-id column).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column carried onto each row.")]
    mode: Annotated[str, Arg("mode", default="global", doc="Alignment mode: 'global' or 'local'.")]


def _resolve_pair(schema: pa.Schema, id_col: str, seq1: str, seq2: str) -> tuple[str, str]:
    """Resolve the two sequence column names, defaulting to the non-id columns in order."""
    non_id = [n for n in schema.names if n != id_col]
    s1 = seq1 or (non_id[0] if len(non_id) > 0 else "")
    s2 = seq2 or (non_id[1] if len(non_id) > 1 else "")
    for label, col in (("seq1", s1), ("seq2", s2)):
        if not col or col not in schema.names:
            raise ValueError(f"{label} column {col!r} not found in input; columns: {', '.join(schema.names)}")
    return s1, s2


class _PairwiseAlign(SinkBuffer[_AlignArgs, DrainState]):
    """Align two sequence columns row by row, emitting aligned strings + score."""

    FunctionArguments: ClassVar[type] = _AlignArgs
    KIND: ClassVar[str]

    @classmethod
    def on_bind(cls, params: BindParams[_AlignArgs]) -> BindResponse:
        """Validate columns/mode and fix the aligned output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        _resolve_pair(input_schema, a.id, a.seq1, a.seq2)
        if a.mode not in ("global", "local"):
            raise ValueError(f"mode must be 'global' or 'local' (got {a.mode!r})")
        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        fields.append(sfield("aligned_1", pa.string(), "First sequence with alignment gaps.", nullable=True))
        fields.append(sfield("aligned_2", pa.string(), "Second sequence with alignment gaps.", nullable=True))
        fields.append(sfield("score", pa.float64(), "Optimal alignment score.", nullable=True))
        fields.append(sfield("length", pa.int64(), "Aligned length (columns).", nullable=True))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_AlignArgs]) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_AlignArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Align each buffered pair and emit one row, once."""
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
    def encode(cls, table: pa.Table, args: _AlignArgs) -> dict[str, list[Any]]:
        """Align each row and return the aligned-output columns."""
        s1_col, s2_col = _resolve_pair(table.schema, args.id, args.seq1, args.seq2)
        s1_vals = table.column(s1_col).to_pylist()
        s2_vals = table.column(s2_col).to_pylist()
        id_vals = table.column(args.id).to_pylist() if args.id else None

        ids: list[Any] = []
        a1: list[str | None] = []
        a2: list[str | None] = []
        score: list[float | None] = []
        length: list[int | None] = []
        for row, (a, b) in enumerate(zip(s1_vals, s2_vals, strict=True)):
            ca, cb = _clean(a), _clean(b)
            if id_vals is not None:
                ids.append(id_vals[row])
            try:
                al1, al2, sc, ln = (
                    (None, None, None, None) if ca is None or cb is None else _align(cls.KIND, args.mode, ca, cb)
                )
            except Exception:
                al1, al2, sc, ln = None, None, None, None
            a1.append(al1)
            a2.append(al2)
            score.append(sc)
            length.append(ln)

        columns: dict[str, list[Any]] = {}
        if args.id:
            columns[args.id] = ids
        columns["aligned_1"] = a1
        columns["aligned_2"] = a2
        columns["score"] = score
        columns["length"] = length
        return columns


_ALIGN_COLUMNS: list[tuple[str, str, str]] = [
    ("aligned_1", "VARCHAR", "First sequence with alignment gaps ('-')."),
    ("aligned_2", "VARCHAR", "Second sequence with alignment gaps ('-')."),
    ("score", "DOUBLE", "Optimal alignment score."),
    ("length", "BIGINT", "Aligned length (number of columns)."),
]


def _align_result_cols() -> str:
    """Result-schema variants for an alignment table function.

    The shape depends on ``id :=``: naming an id column carries it through as the
    first output column, under the input column's own name and type, so there are
    two variants rather than one static schema.
    """
    return result_dynamic_columns_md(
        [
            ("Default -- no `id :=`", _ALIGN_COLUMNS),
            (
                "With `id := 'read_id'` (a `VARCHAR` id column)",
                [("read_id", "VARCHAR", "The named id column, carried through unchanged.")] + _ALIGN_COLUMNS,
            ),
        ],
        note=(
            "The carried column takes the *name and type* of the input column named by `id :=`; "
            "the second variant shows the common case of a `VARCHAR` read id."
        ),
    )


class PairwiseAlignNucleotide(_PairwiseAlign):
    """Global/local pairwise alignment of two DNA columns (aligned strings + score)."""

    KIND: ClassVar[str] = "nucl"

    class Meta:
        """VGI metadata for the pairwise_align_nucleotide function."""

        name = "pairwise_align_nucleotide"
        description = "Pairwise align two DNA columns; emit aligned strings and score"
        categories = ["alignment", "nucleotide"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT read_id, aligned_1, aligned_2, score, length FROM "
                    "skbio.alignment.pairwise_align_nucleotide((SELECT * FROM "
                    "(VALUES ('r1', 'ACTGGT', 'ACTGT'), ('r2', 'GATTACA', 'GATACA')) "
                    "AS p(read_id, ref, read)), id := 'read_id', seq1 := 'ref', seq2 := 'read') "
                    "ORDER BY score DESC"
                ),
                description=(
                    "Align each read against its reference and rank the reads by how well they "
                    "match: the aligned strings show where the gaps ('-') fell, so you can see "
                    "the indel that the score alone only hints at. Carrying read_id through "
                    "(id := 'read_id') is what lets the alignment be joined back to the source rows."
                ),
            )
        ]
        tags = {
            "vgi.category": "pairwise",
            "vgi.result_dynamic_columns_md": _align_result_cols(),
            "vgi.doc_llm": (
                "Table function performing pairwise alignment of two DNA sequence columns and emitting the "
                "aligned strings, the score, and the aligned length, one row per input pair. The table arg is "
                "`(SELECT id?, seq1_col, seq2_col FROM ...)`; `seq1 :=`/`seq2 :=` name the two DNA columns "
                "(default the first two non-id columns), `id :=` an optional carried id, and `mode :=` picks "
                "`global` (Needleman-Wunsch, default) or `local` (Smith-Waterman). Aligned sequences use "
                "`-` for gaps; a NULL or non-DNA pair yields NULL outputs. Use `align_score_nucleotide` for "
                "just the score."
            ),
            "vgi.doc_md": (
                "**pairwise_align_nucleotide** — align two DNA columns, aligned strings + score.\n\n"
                "- Table arg: `(SELECT id?, seq1, seq2 FROM ...)`; `seq1 :=`/`seq2 :=` the DNA columns, "
                "`id :=` optional, `mode :=` `global` (default) or `local`\n"
                "- Returns `(id?, aligned_1, aligned_2, score, length)`; gaps shown as `-`\n"
                "- NULL/non-DNA pairs yield NULL; `align_score_nucleotide` returns just the score"
            ),
        }


class PairwiseAlignProtein(_PairwiseAlign):
    """Global/local pairwise alignment of two protein columns (aligned strings + score)."""

    KIND: ClassVar[str] = "prot"

    class Meta:
        """VGI metadata for the pairwise_align_protein function."""

        name = "pairwise_align_protein"
        description = "Pairwise align two protein columns; emit aligned strings and score"
        categories = ["alignment", "protein"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT read_id, aligned_1, aligned_2, score, length FROM "
                    "skbio.alignment.pairwise_align_protein((SELECT * FROM "
                    "(VALUES ('p1', 'MRITMK', 'MRIMK'), ('p2', 'MKVLAA', 'MKVLAA')) "
                    "AS p(read_id, ref, read)), id := 'read_id', seq1 := 'ref', seq2 := 'read') "
                    "ORDER BY read_id"
                ),
                description=(
                    "Align candidate protein sequences against their references and inspect the "
                    "gapped alignment beside the score. Comparing an imperfect pair with an "
                    "identical one shows what a 'good' score looks like for this substitution "
                    "matrix, which a bare score column cannot tell you."
                ),
            )
        ]
        tags = {
            "vgi.category": "pairwise",
            "vgi.result_dynamic_columns_md": _align_result_cols(),
            "vgi.doc_llm": (
                "Table function performing pairwise alignment of two protein sequence columns and emitting "
                "the aligned strings, the score, and the aligned length, one row per input pair. The table "
                "arg is `(SELECT id?, seq1_col, seq2_col FROM ...)`; `seq1 :=`/`seq2 :=` name the two protein "
                "columns (default the first two non-id columns), `id :=` an optional carried id, and "
                "`mode :=` picks `global` (default) or `local`. Uses scikit-bio's default substitution "
                "matrix; gaps are `-`, and a NULL or non-protein pair yields NULL. Use "
                "`align_score_protein` for just the score."
            ),
            "vgi.doc_md": (
                "**pairwise_align_protein** — align two protein columns, aligned strings + score.\n\n"
                "- Table arg: `(SELECT id?, seq1, seq2 FROM ...)`; `seq1 :=`/`seq2 :=` the protein columns, "
                "`id :=` optional, `mode :=` `global` (default) or `local`\n"
                "- Returns `(id?, aligned_1, aligned_2, score, length)`; gaps shown as `-`\n"
                "- NULL/non-protein pairs yield NULL; `align_score_protein` returns just the score"
            ),
        }


ALIGNMENT_FUNCTIONS: list[type] = [
    AlignScoreNucleotide,
    AlignScoreProtein,
    PairwiseAlignNucleotide,
    PairwiseAlignProtein,
]
