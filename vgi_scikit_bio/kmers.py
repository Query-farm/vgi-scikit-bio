"""Sequence composition as table functions: k-mer and single-residue counts.

Both turn a sequence column into a **long** ``(id?, token, count)`` matrix -- one
row per distinct token per input sequence -- which is the natural shape for SQL
(pivot back to a wide matrix, join token weights, or aggregate per sequence):

    -- 4-mer profile of each read
    SELECT id, kmer, count
    FROM skbio.sequence.kmer_frequencies((SELECT id, seq FROM reads), id := 'id', k := 4);

The token vocabulary is data-dependent (it is not known until the sequences are
read), so long format sidesteps the fixed-output-width limit. Both functions
buffer the input relation, then emit. Tokens are computed with scikit-bio's
generic ``Sequence``, so DNA, RNA, and protein all work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pyarrow as pa
from skbio import Sequence
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .schema_utils import columns_md_rows
from .schema_utils import field as sfield

_ID_NOTE = "If an `id` column is named, it is carried through as the first column on each row."


def _sequence_column(input_schema: pa.Schema, id_col: str, seq_arg: str) -> str:
    """Resolve which column holds the sequences to tokenize."""
    if seq_arg:
        if seq_arg not in input_schema.names:
            raise ValueError(
                f"sequence column {seq_arg!r} not found in input; columns: {', '.join(input_schema.names)}"
            )
        return seq_arg
    candidates = [n for n in input_schema.names if n != id_col]
    if len(candidates) != 1:
        raise ValueError(
            "could not infer the sequence column; pass sequence := 'column' "
            f"(non-id columns: {', '.join(candidates) or '<none>'})"
        )
    return str(candidates[0])


def _require_string(input_schema: pa.Schema, col: str) -> None:
    """Raise unless the named column is a string/VARCHAR column."""
    t = input_schema.field(col).type
    if not pa.types.is_string(t) and not pa.types.is_large_string(t):
        raise ValueError(f"sequence column {col!r} must be a string/VARCHAR column")


@dataclass(slots=True, frozen=True)
class _KmerArgs:
    data: Annotated[TableInput, Arg(0, doc="An id column and a sequence column.")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column carried onto each emitted row.")]
    sequence: Annotated[
        str, Arg("sequence", default="", doc="Sequence column to tokenize (defaults to the single non-id column).")
    ]
    k: Annotated[int, Arg("k", default=3, doc="k-mer length (number of residues per token).")]


class KmerFrequencies(SinkBuffer[_KmerArgs, DrainState]):
    """Count overlapping k-mers per sequence, emitted long as ``(id?, kmer, count)``."""

    FunctionArguments: ClassVar[type] = _KmerArgs

    class Meta:
        """VGI metadata for the kmer_frequencies function."""

        name = "kmer_frequencies"
        description = "Count overlapping k-mers per sequence (long format)"
        categories = ["sequence", "composition", "kmer"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM skbio.sequence.kmer_frequencies("
                    "(SELECT * FROM (VALUES (1, 'ATGCGGATTACAGG'), (2, 'TTGCACGT')) AS reads(id, seq)), "
                    "id := 'id', k := 3)"
                ),
                description="3-mer counts per sequence",
            )
        ]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    ("kmer", "VARCHAR", "A k-length subsequence (token)."),
                    ("count", "BIGINT", "Number of overlapping occurrences in the sequence."),
                ],
                note=_ID_NOTE,
            ),
            "vgi.doc_llm": (
                "Table function that counts overlapping k-mers (length-`k` subsequences) in each input "
                "sequence and emits them long: one row per distinct k-mer per sequence. The table arg is "
                "`(SELECT id_col, seq_col FROM ...)`; `sequence :=` names the string column (defaults to the "
                "single non-id column), `id :=` an optional id carried onto each row, and `k :=` the k-mer "
                "length (default 3). Uses scikit-bio's generic sequence, so DNA/RNA/protein all work. "
                "Returns `(id?, kmer, count)` where `count` is the overlapping occurrence count — pivot back "
                "to a wide k-mer matrix (a feature table for clustering/classification), or aggregate per "
                "k-mer across sequences."
            ),
            "vgi.doc_md": (
                "**k-mer frequencies** — overlapping length-`k` token counts per sequence, long format.\n\n"
                "- Table arg: `(SELECT id, seq FROM ...)`; `sequence :=` the string column, `id :=` an "
                "optional carried id, `k :=` the k-mer length (default 3)\n"
                "- Returns one row per distinct k-mer per sequence (plus the carried `id` if given):\n"
                "  - `kmer` — a length-`k` subsequence\n"
                "  - `count` (`BIGINT`) — overlapping occurrences in that sequence\n"
                "- Works for DNA/RNA/protein; `PIVOT` for a wide k-mer feature matrix"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[_KmerArgs]) -> BindResponse:
        """Validate the sequence column, resolve k, and fix the long output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        seq_col = _sequence_column(input_schema, a.id, a.sequence)
        _require_string(input_schema, seq_col)
        if a.k < 1:
            raise ValueError(f"k must be >= 1 (got {a.k})")
        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        fields.append(sfield("kmer", pa.string(), "A k-length subsequence (token).", nullable=False))
        fields.append(sfield("count", pa.int64(), "Overlapping occurrences in the sequence.", nullable=False))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_KmerArgs]) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_KmerArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Emit each buffered sequence's k-mer counts as one long batch, once."""
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
    def encode(cls, table: pa.Table, args: _KmerArgs) -> dict[str, list[Any]]:
        """Tokenize each sequence into overlapping k-mers, returning long-format columns."""
        seq_col = _sequence_column(table.schema, args.id, args.sequence)
        seqs = table.column(seq_col).to_pylist()
        id_vals = table.column(args.id).to_pylist() if args.id else None

        ids: list[Any] = []
        kmer_col: list[str] = []
        count_col: list[int] = []
        for row, raw in enumerate(seqs):
            if raw is None:
                continue
            text = str(raw).strip().upper()
            if len(text) < args.k:
                continue
            for token, n in Sequence(text).kmer_frequencies(args.k, overlap=True).items():
                if id_vals is not None:
                    ids.append(id_vals[row])
                kmer_col.append(str(token))
                count_col.append(int(n))

        columns: dict[str, list[Any]] = {}
        if args.id:
            columns[args.id] = ids
        columns["kmer"] = kmer_col
        columns["count"] = count_col
        return columns


@dataclass(slots=True, frozen=True)
class _ResidueArgs:
    data: Annotated[TableInput, Arg(0, doc="An id column and a sequence column.")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column carried onto each emitted row.")]
    sequence: Annotated[
        str, Arg("sequence", default="", doc="Sequence column to tokenize (defaults to the single non-id column).")
    ]


class ResidueFrequencies(SinkBuffer[_ResidueArgs, DrainState]):
    """Count single residues (bases/amino acids) per sequence, long ``(id?, residue, count)``."""

    FunctionArguments: ClassVar[type] = _ResidueArgs

    class Meta:
        """VGI metadata for the residue_frequencies function."""

        name = "residue_frequencies"
        description = "Count single residues (bases or amino acids) per sequence (long format)"
        categories = ["sequence", "composition"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM skbio.sequence.residue_frequencies("
                    "(SELECT * FROM (VALUES (1, 'ATGCGGATTACAGG'), (2, 'TTGCACGT')) AS reads(id, seq)), "
                    "id := 'id')"
                ),
                description="Base composition per sequence",
            )
        ]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    ("residue", "VARCHAR", "A single residue (base or amino acid)."),
                    ("count", "BIGINT", "Number of occurrences in the sequence."),
                ],
                note=_ID_NOTE,
            ),
            "vgi.doc_llm": (
                "Table function that counts single residues (nucleotide bases or amino acids) in each input "
                "sequence and emits them long: one row per distinct residue per sequence. The table arg is "
                "`(SELECT id_col, seq_col FROM ...)`; `sequence :=` names the string column (defaults to the "
                "single non-id column) and `id :=` an optional id carried onto each row. Uses scikit-bio's "
                "generic sequence, so DNA/RNA/protein all work. Returns `(id?, residue, count)` — the "
                "composition of each sequence; sum, or pivot to a wide base-count matrix. Equivalent to a "
                "1-mer `kmer_frequencies`."
            ),
            "vgi.doc_md": (
                "**Residue frequencies** — single base/amino-acid counts per sequence, long format.\n\n"
                "- Table arg: `(SELECT id, seq FROM ...)`; `sequence :=` the string column, `id :=` an "
                "optional carried id\n"
                "- Returns one row per distinct residue per sequence (plus the carried `id` if given):\n"
                "  - `residue` — a single base or amino acid\n"
                "  - `count` (`BIGINT`) — occurrences in that sequence\n"
                "- Works for DNA/RNA/protein; the 1-mer case of `kmer_frequencies`"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[_ResidueArgs]) -> BindResponse:
        """Validate the sequence column and fix the long output schema."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        seq_col = _sequence_column(input_schema, a.id, a.sequence)
        _require_string(input_schema, seq_col)
        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        fields.append(sfield("residue", pa.string(), "A single residue (base or amino acid).", nullable=False))
        fields.append(sfield("count", pa.int64(), "Occurrences in the sequence.", nullable=False))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_ResidueArgs]) -> DrainState:
        """Start a fresh single-shot finalize cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_ResidueArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Emit each buffered sequence's residue counts as one long batch, once."""
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
    def encode(cls, table: pa.Table, args: _ResidueArgs) -> dict[str, list[Any]]:
        """Count single residues per sequence, returning long-format columns."""
        seq_col = _sequence_column(table.schema, args.id, args.sequence)
        seqs = table.column(seq_col).to_pylist()
        id_vals = table.column(args.id).to_pylist() if args.id else None

        ids: list[Any] = []
        residue_col: list[str] = []
        count_col: list[int] = []
        for row, raw in enumerate(seqs):
            if raw is None:
                continue
            text = str(raw).strip().upper()
            if not text:
                continue
            for residue, n in Sequence(text).frequencies().items():
                if id_vals is not None:
                    ids.append(id_vals[row])
                residue_col.append(str(residue))
                count_col.append(int(n))

        columns: dict[str, list[Any]] = {}
        if args.id:
            columns[args.id] = ids
        columns["residue"] = residue_col
        columns["count"] = count_col
        return columns


KMER_FUNCTIONS: list[type] = [KmerFrequencies, ResidueFrequencies]
