"""Nucleotide and protein sequence operations as DuckDB scalar functions.

Each function maps a VARCHAR sequence column (or a pair of them) to one value
per row, wrapping scikit-bio's grammared sequence types (``DNA``, ``RNA``,
``Protein``):

    SELECT id, skbio.sequence.gc_content(seq) AS gc,
               skbio.sequence.reverse_complement(seq) AS rc
    FROM reads;

Inputs are upper-cased and stripped before parsing. A row that is NULL or is not
a valid sequence for the requested operation yields NULL rather than raising, so
a single malformed read never fails the whole query.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

import pyarrow as pa
from skbio import DNA, Protein
from vgi.arguments import Param, Returns
from vgi.metadata import FunctionExample
from vgi.scalar_function import ScalarFunction


def _clean(value: Any) -> str | None:
    """Normalise one input cell to an upper-cased sequence string, or None."""
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _map_str(seqs: pa.Array, fn: Callable[[str], str | None]) -> pa.Array:
    """Apply ``fn`` to each cleaned sequence, returning a string result array."""
    out: list[str | None] = []
    for raw in seqs.to_pylist():
        text = _clean(raw)
        if text is None:
            out.append(None)
            continue
        try:
            out.append(fn(text))
        except Exception:
            out.append(None)
    return pa.array(out, type=pa.string())


def _map_float(seqs: pa.Array, fn: Callable[[str], float | None]) -> pa.Array:
    """Apply ``fn`` to each cleaned sequence, returning a float result array."""
    out: list[float | None] = []
    for raw in seqs.to_pylist():
        text = _clean(raw)
        if text is None:
            out.append(None)
            continue
        try:
            out.append(fn(text))
        except Exception:
            out.append(None)
    return pa.array(out, type=pa.float64())


def _map_bool(seqs: pa.Array, fn: Callable[[str], bool]) -> pa.Array:
    """Apply a validity predicate to each cleaned sequence (NULL stays NULL)."""
    out: list[bool | None] = []
    for raw in seqs.to_pylist():
        text = _clean(raw)
        if text is None:
            out.append(None)
            continue
        out.append(fn(text))
    return pa.array(out, type=pa.bool_())


# ===========================================================================
# Nucleotide transforms
# ===========================================================================


class GcContent(ScalarFunction):
    """Fraction of G/C bases in a DNA sequence (0.0-1.0)."""

    class Meta:
        """VGI metadata for the gc_content scalar."""

        name = "gc_content"
        description = "GC content of a DNA sequence as a fraction in [0, 1]"
        categories = ["sequence", "nucleotide"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.gc_content('ATGCGGATTACAGG')",
                description="GC fraction of an inline DNA sequence",
            )
        ]
        tags = {
            "vgi.doc_llm": (
                "Scalar function returning the GC content of a DNA sequence as a `DOUBLE` fraction in "
                "`[0, 1]` — the number of G and C bases divided by the sequence length (gaps excluded). "
                "Pass one `VARCHAR` column of DNA (case-insensitive; whitespace ignored). Rows that are "
                "NULL or not valid DNA (e.g. protein or free text) return NULL. Use it to profile "
                "read/contig composition, screen for GC-bias, or bin sequences by GC — multiply by 100 "
                "for a percentage."
            ),
            "vgi.doc_md": (
                "**GC content** — fraction of G/C bases in a DNA sequence.\n\n"
                "- Input: one `VARCHAR` DNA column (case-insensitive)\n"
                "- Returns: a `DOUBLE` in `[0, 1]` (`* 100` for a percentage)\n"
                "- NULL for NULL or non-DNA input; gaps are excluded from the denominator"
            ),
        }

    @classmethod
    def compute(
        cls,
        seq: Annotated[pa.StringArray, Param(doc="A DNA sequence")],
    ) -> Annotated[pa.DoubleArray, Returns(pa.float64())]:
        """Return each sequence's GC fraction (NULL for invalid DNA)."""
        return _map_float(seq, lambda s: float(DNA(s).gc_content()))


class ReverseComplement(ScalarFunction):
    """Reverse complement of a DNA sequence."""

    class Meta:
        """VGI metadata for the reverse_complement scalar."""

        name = "reverse_complement"
        description = "Reverse complement of a DNA sequence"
        categories = ["sequence", "nucleotide"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.reverse_complement('ATGCGGATTACAGG')",
                description="Reverse complement of an inline DNA sequence",
            )
        ]
        tags = {
            "vgi.doc_llm": (
                "Scalar function returning the reverse complement of a DNA sequence as a `VARCHAR`: the "
                "complement of each base (A<->T, G<->C) read 3'->5', i.e. the sequence of the opposite "
                "strand. Pass one DNA column (case-insensitive; whitespace ignored). Handles IUPAC "
                "degenerate codes and gaps; NULL or non-DNA input returns NULL. Use it to get the "
                "opposite-strand sequence, orient reads, or design/compare primers."
            ),
            "vgi.doc_md": (
                "**Reverse complement** — the opposite-strand sequence of DNA.\n\n"
                "- Input: one `VARCHAR` DNA column (case-insensitive)\n"
                "- Returns: a `VARCHAR` — complement of each base, reversed\n"
                "- IUPAC degenerate codes and gaps supported; NULL for NULL/non-DNA input"
            ),
        }

    @classmethod
    def compute(
        cls,
        seq: Annotated[pa.StringArray, Param(doc="A DNA sequence")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Return each sequence's reverse complement (NULL for invalid DNA)."""
        return _map_str(seq, lambda s: str(DNA(s).reverse_complement()))


class Complement(ScalarFunction):
    """Complement of a DNA sequence (not reversed)."""

    class Meta:
        """VGI metadata for the complement scalar."""

        name = "complement"
        description = "Complement of a DNA sequence (same 5'->3' order)"
        categories = ["sequence", "nucleotide"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.complement('ATGCGGATTACAGG')",
                description="Base complement of an inline DNA sequence",
            )
        ]
        tags = {
            "vgi.doc_llm": (
                "Scalar function returning the base complement of a DNA sequence as a `VARCHAR` (A<->T, "
                "G<->C) in the same 5'->3' order — unlike `reverse_complement`, the sequence is not "
                "reversed. Pass one DNA column (case-insensitive). IUPAC degenerate codes and gaps are "
                "supported; NULL or non-DNA input returns NULL."
            ),
            "vgi.doc_md": (
                "**Complement** — base-by-base complement of DNA, order preserved.\n\n"
                "- Input: one `VARCHAR` DNA column (case-insensitive)\n"
                "- Returns: a `VARCHAR`; use `reverse_complement` for the opposite strand\n"
                "- NULL for NULL/non-DNA input"
            ),
        }

    @classmethod
    def compute(
        cls,
        seq: Annotated[pa.StringArray, Param(doc="A DNA sequence")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Return each sequence's base complement (NULL for invalid DNA)."""
        return _map_str(seq, lambda s: str(DNA(s).complement()))


class Transcribe(ScalarFunction):
    """Transcribe a DNA sequence to RNA (T -> U)."""

    class Meta:
        """VGI metadata for the transcribe scalar."""

        name = "transcribe"
        description = "Transcribe a DNA sequence to RNA (T -> U)"
        categories = ["sequence", "nucleotide"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.transcribe('ATGCGGATTACAGG')",
                description="RNA transcript of an inline DNA sequence",
            )
        ]
        tags = {
            "vgi.doc_llm": (
                "Scalar function transcribing a DNA sequence into its RNA transcript as a `VARCHAR`, "
                "replacing every thymine (T) with uracil (U). Pass one DNA column (case-insensitive). "
                "NULL or non-DNA input returns NULL. Chain with `translate` to go DNA -> protein, or use "
                "it to prepare RNA sequences for downstream analysis."
            ),
            "vgi.doc_md": (
                "**Transcribe** — DNA to RNA (T becomes U).\n\n"
                "- Input: one `VARCHAR` DNA column (case-insensitive)\n"
                "- Returns: the RNA transcript as `VARCHAR`\n"
                "- NULL for NULL/non-DNA input"
            ),
        }

    @classmethod
    def compute(
        cls,
        seq: Annotated[pa.StringArray, Param(doc="A DNA sequence")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Return each sequence's RNA transcript (NULL for invalid DNA)."""
        return _map_str(seq, lambda s: str(DNA(s).transcribe()))


class Translate(ScalarFunction):
    """Translate a DNA sequence to protein using the standard genetic code."""

    class Meta:
        """VGI metadata for the translate scalar."""

        name = "translate"
        description = "Translate a DNA sequence to protein (standard genetic code)"
        categories = ["sequence", "nucleotide", "protein"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.translate('ATGCGGATTACAGGT')",
                description="Protein translation of an inline DNA sequence",
            )
        ]
        tags = {
            "vgi.doc_llm": (
                "Scalar function translating a DNA sequence to its amino-acid sequence as a `VARCHAR`, "
                "reading successive codons from the first base with the standard genetic code (NCBI table "
                "1); a stop codon is emitted as `*`. Pass one DNA column (case-insensitive). Trailing bases "
                "that do not complete a codon are ignored. NULL or non-DNA input returns NULL. Pair with "
                "`transcribe` for the RNA intermediate."
            ),
            "vgi.doc_md": (
                "**Translate** — DNA to protein via the standard genetic code.\n\n"
                "- Input: one `VARCHAR` DNA column (case-insensitive)\n"
                "- Returns: the amino-acid sequence as `VARCHAR` (stop codons as `*`)\n"
                "- Reads codons from base 0; trailing partial codon ignored; NULL for NULL/non-DNA input"
            ),
        }

    @classmethod
    def compute(
        cls,
        seq: Annotated[pa.StringArray, Param(doc="A DNA sequence")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Return each sequence's protein translation (NULL for invalid DNA)."""
        return _map_str(seq, lambda s: str(DNA(s).translate()))


# ===========================================================================
# Validation
# ===========================================================================


def _is_valid(kind: type, text: str) -> bool:
    """True if ``text`` parses as the given grammared sequence type."""
    try:
        kind(text)
        return True
    except Exception:
        return False


class IsValidDna(ScalarFunction):
    """Whether a string is a valid IUPAC DNA sequence."""

    class Meta:
        """VGI metadata for the is_valid_dna scalar."""

        name = "is_valid_dna"
        description = "True if the string is a valid IUPAC DNA sequence"
        categories = ["sequence", "nucleotide", "validation"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.is_valid_dna('ATGCN'), skbio.sequence.is_valid_dna('HELLO')",
                description="Validate DNA strings",
            )
        ]
        tags = {
            "vgi.doc_llm": (
                "Scalar predicate returning `BOOLEAN` true when a string is a valid IUPAC DNA sequence — "
                "only the recognised nucleotide characters (A, C, G, T, the degenerate codes, and gap "
                "characters), case-insensitively. Pass one `VARCHAR` column. NULL input returns NULL; any "
                "other invalid content returns false. Use it to filter or flag non-DNA rows before calling "
                "the DNA transforms."
            ),
            "vgi.doc_md": (
                "**is_valid_dna** — validate a DNA string.\n\n"
                "- Input: one `VARCHAR` column (case-insensitive)\n"
                "- Returns: `BOOLEAN` (NULL for NULL input)\n"
                "- Accepts IUPAC nucleotide + degenerate + gap characters only"
            ),
        }

    @classmethod
    def compute(
        cls,
        seq: Annotated[pa.StringArray, Param(doc="A candidate DNA sequence")],
    ) -> Annotated[pa.BooleanArray, Returns(pa.bool_())]:
        """Return whether each string parses as DNA (NULL stays NULL)."""
        return _map_bool(seq, lambda s: _is_valid(DNA, s))


class IsValidProtein(ScalarFunction):
    """Whether a string is a valid IUPAC protein sequence."""

    class Meta:
        """VGI metadata for the is_valid_protein scalar."""

        name = "is_valid_protein"
        description = "True if the string is a valid IUPAC protein sequence"
        categories = ["sequence", "protein", "validation"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.is_valid_protein('MRIT'), skbio.sequence.is_valid_protein('ATGC1')",
                description="Validate protein strings",
            )
        ]
        tags = {
            "vgi.doc_llm": (
                "Scalar predicate returning `BOOLEAN` true when a string is a valid IUPAC protein "
                "sequence — the 20 amino-acid codes plus the recognised ambiguity and gap characters, "
                "case-insensitively. Pass one `VARCHAR` column. NULL input returns NULL; other invalid "
                "content returns false. Note that pure-nucleotide strings can also be valid protein "
                "(A/C/G/T are amino-acid codes), so use `is_valid_dna` to distinguish DNA specifically."
            ),
            "vgi.doc_md": (
                "**is_valid_protein** — validate a protein string.\n\n"
                "- Input: one `VARCHAR` column (case-insensitive)\n"
                "- Returns: `BOOLEAN` (NULL for NULL input)\n"
                "- Accepts the 20 amino acids + ambiguity/gap characters"
            ),
        }

    @classmethod
    def compute(
        cls,
        seq: Annotated[pa.StringArray, Param(doc="A candidate protein sequence")],
    ) -> Annotated[pa.BooleanArray, Returns(pa.bool_())]:
        """Return whether each string parses as protein (NULL stays NULL)."""
        return _map_bool(seq, lambda s: _is_valid(Protein, s))


# ===========================================================================
# Pairwise sequence distance
# ===========================================================================


class HammingDistance(ScalarFunction):
    """Hamming distance (mismatch fraction) between two equal-length sequences."""

    class Meta:
        """VGI metadata for the hamming_distance scalar."""

        name = "hamming_distance"
        description = "Hamming distance (fraction of differing positions) between two sequences"
        categories = ["sequence", "distance"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.hamming_distance('ACGTACGT', 'ACGAACGT')",
                description="Hamming distance between two equal-length sequences",
            )
        ]
        tags = {
            "vgi.doc_llm": (
                "Scalar function returning the Hamming distance between two equal-length sequences as a "
                "`DOUBLE` in `[0, 1]`: the fraction of positions at which the characters differ. Pass two "
                "`VARCHAR` columns (case-insensitive; compared as generic sequences, so DNA/RNA/protein "
                "all work). Returns NULL if either value is NULL or the two sequences differ in length. "
                "Multiply by the length for a raw mismatch count; use it for read-vs-reference comparison "
                "or simple sequence similarity."
            ),
            "vgi.doc_md": (
                "**Hamming distance** — mismatch fraction between two equal-length sequences.\n\n"
                "- Inputs: two `VARCHAR` sequence columns (case-insensitive)\n"
                "- Returns: a `DOUBLE` in `[0, 1]`; NULL if either is NULL or lengths differ\n"
                "- `* length` for a raw mismatch count"
            ),
        }

    @classmethod
    def compute(
        cls,
        seq1: Annotated[pa.StringArray, Param(doc="First sequence")],
        seq2: Annotated[pa.StringArray, Param(doc="Second sequence (same length as the first)")],
    ) -> Annotated[pa.DoubleArray, Returns(pa.float64())]:
        """Return the per-row Hamming fraction (NULL on NULL or length mismatch)."""
        from skbio import Sequence

        out: list[float | None] = []
        for a, b in zip(seq1.to_pylist(), seq2.to_pylist(), strict=False):
            ca, cb = _clean(a), _clean(b)
            if ca is None or cb is None or len(ca) != len(cb):
                out.append(None)
                continue
            try:
                out.append(float(Sequence(ca).distance(Sequence(cb))))
            except Exception:
                out.append(None)
        return pa.array(out, type=pa.float64())


SEQUENCE_FUNCTIONS: list[type] = [
    GcContent,
    ReverseComplement,
    Complement,
    Transcribe,
    Translate,
    IsValidDna,
    IsValidProtein,
    HammingDistance,
]
