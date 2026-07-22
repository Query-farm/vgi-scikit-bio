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
from skbio import DNA, RNA, Protein, Sequence
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
        try:
            out.append(fn(text))
        except Exception:
            out.append(None)
    return pa.array(out, type=pa.bool_())


def _map_int(seqs: pa.Array, fn: Callable[[str], int | None]) -> pa.Array:
    """Apply ``fn`` to each cleaned sequence, returning an int64 result array."""
    out: list[int | None] = []
    for raw in seqs.to_pylist():
        text = _clean(raw)
        if text is None:
            out.append(None)
            continue
        try:
            out.append(fn(text))
        except Exception:
            out.append(None)
    return pa.array(out, type=pa.int64())


def _map_pair_int(seq1: pa.Array, seq2: pa.Array, fn: Callable[[str, str], int | None]) -> pa.Array:
    """Apply a two-sequence integer function per row (NULL on NULL/invalid)."""
    out: list[int | None] = []
    for a, b in zip(seq1.to_pylist(), seq2.to_pylist(), strict=False):
        ca, cb = _clean(a), _clean(b)
        if ca is None or cb is None:
            out.append(None)
            continue
        try:
            out.append(fn(ca, cb))
        except Exception:
            out.append(None)
    return pa.array(out, type=pa.int64())


def _map_pair_bool(seq1: pa.Array, seq2: pa.Array, fn: Callable[[str, str], bool]) -> pa.Array:
    """Apply a two-sequence predicate per row (NULL on NULL/invalid)."""
    out: list[bool | None] = []
    for a, b in zip(seq1.to_pylist(), seq2.to_pylist(), strict=False):
        ca, cb = _clean(a), _clean(b)
        if ca is None or cb is None:
            out.append(None)
            continue
        try:
            out.append(fn(ca, cb))
        except Exception:
            out.append(None)
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
                sql="SELECT skbio.sequence.gc_content('ATGCGGATTACAGG') AS gc",
                description=(
                    "Profile a read's base composition as a single number in [0, 1]. GC content "
                    "is the first thing to check when reads look odd: an outlier flags "
                    "contamination or a different organism, and it drives PCR and melting "
                    "behaviour. Multiply by 100 for the percentage people usually quote."
                ),
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
                sql="SELECT skbio.sequence.reverse_complement('ATGCGGATTACAGG') AS rc",
                description=(
                    "Read a sequence off the opposite strand. Sequencing gives reads in either "
                    "orientation, so normalising with this is what lets two reads of the same "
                    "locus be compared or deduplicated at all; it is also how a reverse primer "
                    "is written down."
                ),
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
                sql="SELECT skbio.sequence.complement('ATGCGGATTACAGG') AS comp",
                description=(
                    "Complement each base while keeping 5'->3' order — deliberately *not* the "
                    "opposite strand. Compare it with reverse_complement on the same input to "
                    "see the difference that trips people up: only the reversed form is what "
                    "actually pairs with the original."
                ),
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
                sql="SELECT skbio.sequence.transcribe('ATGCGGATTACAGG') AS rna",
                description=(
                    "Produce the RNA transcript of a DNA template (every T becomes U). Use it "
                    "when a downstream tool or reference is written in RNA alphabet, or as the "
                    "explicit middle step of the DNA -> RNA -> protein chain that translate "
                    "otherwise does in one hop."
                ),
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
                sql="SELECT skbio.sequence.translate('ATGCGGATTACAGGT') AS protein",
                description=(
                    "Turn a coding sequence into its amino-acid sequence, reading codons from "
                    "base 0 with the standard genetic code. Note the trailing base that does not "
                    "complete a codon is simply dropped -- if the frame is unknown, use "
                    "translate_six_frames instead of guessing this one."
                ),
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
                sql=(
                    "SELECT skbio.sequence.is_valid_dna('ATGCN') AS looks_like_dna, "
                    "skbio.sequence.is_valid_dna('HELLO') AS looks_like_dna_too"
                ),
                description=(
                    "Screen a text column before running the DNA transforms over it. Contrasting "
                    "a degenerate-but-valid read (N is a legal IUPAC code) with free text shows "
                    "where the line is: use it in a WHERE clause so junk rows are filtered rather "
                    "than silently turning into NULLs downstream."
                ),
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
                sql=(
                    "SELECT skbio.sequence.is_valid_protein('MRIT') AS looks_like_protein, "
                    "skbio.sequence.is_valid_protein('ATGC1') AS looks_like_protein_too"
                ),
                description=(
                    "Check a column really holds amino-acid sequences before feeding the protein "
                    "aligners. Beware the asymmetry this example hints at: a pure A/C/G/T string "
                    "is *also* valid protein, so pair this with is_valid_dna when you need to "
                    "tell the two alphabets apart."
                ),
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
                sql="SELECT skbio.sequence.hamming_distance('ACGTACGT', 'ACGAACGT') AS distance",
                description=(
                    "Score how far a read has drifted from its reference as a length-normalised "
                    "fraction, so reads of different lengths stay comparable. One substitution in "
                    "eight bases gives 0.125; multiply by the length to recover the raw mismatch "
                    "count. Requires equal lengths -- align first if there are indels."
                ),
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


# ===========================================================================
# Additional single-sequence scalars
# ===========================================================================


def _doc(name: str, kind: str, llm: str, md: str) -> dict[str, str]:
    """Build the doc/category tags for a sequence scalar."""
    return {"vgi.category": kind, "vgi.doc_llm": llm, "vgi.doc_md": md}


def _grammared(text: str) -> Any:
    """Parse a (possibly gapped) sequence as DNA if valid, else protein.

    ``degap`` / ``has_gaps`` live on the grammared sequence types (DNA/RNA/
    Protein), not the base ``Sequence``, so a concrete alphabet is required.
    """
    try:
        return DNA(text)
    except Exception:
        return Protein(text)


class GcFrequency(ScalarFunction):
    """Number of G/C bases in a DNA sequence (a raw count, not a fraction)."""

    class Meta:
        """VGI metadata for the gc_frequency scalar."""

        name = "gc_frequency"
        description = "Number of G/C bases in a DNA sequence"
        categories = ["sequence", "nucleotide"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.gc_frequency('ATGCGGATTACAGG') AS gc_bases",
                description=(
                    "Count G and C bases outright rather than as a fraction. The raw count is "
                    "what you want when summing across reads or weighting by length; "
                    "gc_content gives the same information already divided by the sequence length."
                ),
            )
        ]
        tags = _doc(
            "gc_frequency",
            "transforms",
            "Scalar function returning the number of G and C bases in a DNA sequence as a `BIGINT` (the raw "
            "count behind `gc_content`). Pass one `VARCHAR` DNA column (case-insensitive); NULL or non-DNA "
            "input returns NULL. Divide by `length(seq)` for the fraction, or use `gc_content` directly.",
            "**gc_frequency** — count of G/C bases in a DNA sequence.\n\n"
            "- Input: one `VARCHAR` DNA column (case-insensitive)\n"
            "- Returns: a `BIGINT` count (NULL for NULL/non-DNA input)\n"
            "- The raw count behind `gc_content`",
        )

    @classmethod
    def compute(
        cls,
        seq: Annotated[pa.StringArray, Param(doc="A DNA sequence")],
    ) -> Annotated[pa.Int64Array, Returns(pa.int64())]:
        """Return each sequence's G/C base count (NULL for invalid DNA)."""
        return _map_int(seq, lambda s: int(DNA(s).gc_frequency()))


class Degap(ScalarFunction):
    """Remove gap characters ('-', '.') from a sequence."""

    class Meta:
        """VGI metadata for the degap scalar."""

        name = "degap"
        description = "Remove gap characters from a sequence"
        categories = ["sequence"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.degap('AC-GT--A') AS ungapped",
                description=(
                    "Recover the original sequence from an aligned one by dropping the gap "
                    "characters an alignment inserted. Run it before length, composition, or "
                    "k-mer calculations, all of which would otherwise count '-' as if it were a "
                    "residue."
                ),
            )
        ]
        tags = _doc(
            "degap",
            "transforms",
            "Scalar function removing gap characters (`-` and `.`) from a sequence, returning the ungapped "
            "sequence as a `VARCHAR`. Works for any alphabet (DNA/RNA/protein). Pass one `VARCHAR` column; "
            "NULL input returns NULL. Use it to recover the original sequence from an aligned one.",
            "**degap** — strip gap characters from a sequence.\n\n"
            "- Input: one `VARCHAR` sequence column (any alphabet)\n"
            "- Returns: the ungapped sequence as `VARCHAR` (NULL for NULL input)\n"
            "- Removes `-` and `.`; recovers the sequence behind an alignment",
        )

    @classmethod
    def compute(
        cls,
        seq: Annotated[pa.StringArray, Param(doc="A sequence, possibly with gaps")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Return each sequence with gap characters removed (NULL stays NULL)."""
        return _map_str(seq, lambda s: str(_grammared(s).degap()))


class ReverseTranscribe(ScalarFunction):
    """Reverse transcribe an RNA sequence to DNA (U -> T)."""

    class Meta:
        """VGI metadata for the reverse_transcribe scalar."""

        name = "reverse_transcribe"
        description = "Reverse transcribe an RNA sequence to DNA (U -> T)"
        categories = ["sequence", "nucleotide"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.reverse_transcribe('AUGCGGAUUACAGG') AS dna",
                description=(
                    "Bring RNA-alphabet sequences (U instead of T) back into DNA space so they "
                    "can be compared against a DNA reference or run through the DNA-only "
                    "functions. It is the exact inverse of transcribe."
                ),
            )
        ]
        tags = _doc(
            "reverse_transcribe",
            "transforms",
            "Scalar function reverse-transcribing an RNA sequence to DNA as a `VARCHAR`, replacing every "
            "uracil (U) with thymine (T) — the inverse of `transcribe`. Pass one `VARCHAR` RNA column "
            "(case-insensitive); NULL or non-RNA input returns NULL.",
            "**reverse_transcribe** — RNA to DNA (U becomes T).\n\n"
            "- Input: one `VARCHAR` RNA column (case-insensitive)\n"
            "- Returns: the DNA sequence as `VARCHAR` (NULL for NULL/non-RNA input)\n"
            "- The inverse of `transcribe`",
        )

    @classmethod
    def compute(
        cls,
        seq: Annotated[pa.StringArray, Param(doc="An RNA sequence")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Return each RNA sequence reverse-transcribed to DNA (NULL for invalid RNA)."""
        return _map_str(seq, lambda s: str(RNA(s).reverse_transcribe()))


class HasGaps(ScalarFunction):
    """Whether a sequence contains any gap characters."""

    class Meta:
        """VGI metadata for the has_gaps scalar."""

        name = "has_gaps"
        description = "True if the sequence contains gap characters"
        categories = ["sequence", "validation"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT skbio.sequence.has_gaps('AC-GT') AS aligned_row, skbio.sequence.has_gaps('ACGT') AS raw_row"
                ),
                description=(
                    "Tell aligned sequences apart from raw ones in a mixed column — the gapped "
                    "row is the one that came out of an alignment. Use it to decide which rows "
                    "need degap before a length or composition calculation."
                ),
            )
        ]
        tags = _doc(
            "has_gaps",
            "validation",
            "Scalar predicate returning `BOOLEAN` true when a sequence contains at least one gap character "
            "(`-` or `.`). Works for any alphabet. Pass one `VARCHAR` column; NULL input returns NULL. Use "
            "it to find aligned (gapped) rows or to filter them before `degap`.",
            "**has_gaps** — does a sequence contain gaps?\n\n"
            "- Input: one `VARCHAR` sequence column (any alphabet)\n"
            "- Returns: `BOOLEAN` (NULL for NULL input)\n"
            "- True if any `-` or `.` is present",
        )

    @classmethod
    def compute(
        cls,
        seq: Annotated[pa.StringArray, Param(doc="A sequence")],
    ) -> Annotated[pa.BooleanArray, Returns(pa.bool_())]:
        """Return whether each sequence contains gaps (NULL stays NULL)."""
        return _map_bool(seq, lambda s: bool(_grammared(s).has_gaps()))


class HasDegenerates(ScalarFunction):
    """Whether a DNA sequence contains degenerate (ambiguity) codes."""

    class Meta:
        """VGI metadata for the has_degenerates scalar."""

        name = "has_degenerates"
        description = "True if the DNA sequence contains degenerate (ambiguity) codes"
        categories = ["sequence", "nucleotide", "validation"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT skbio.sequence.has_degenerates('ACGTN') AS ambiguous_read, "
                    "skbio.sequence.has_degenerates('ACGT') AS clean_read"
                ),
                description=(
                    "Flag reads containing ambiguity codes (N, R, Y, ...), i.e. positions the "
                    "basecaller could not resolve. Filtering these out is a standard quality "
                    "step before variant calling or exact-match lookups, where an N would "
                    "otherwise silently fail to match."
                ),
            )
        ]
        tags = _doc(
            "has_degenerates",
            "validation",
            "Scalar predicate returning `BOOLEAN` true when a DNA sequence contains at least one degenerate "
            "(ambiguity) code such as N, R, or Y — a base that stands for several possibilities. Pass one "
            "`VARCHAR` DNA column; NULL or non-DNA input returns NULL. Use it to flag reads with ambiguous "
            "bases.",
            "**has_degenerates** — does DNA contain ambiguity codes?\n\n"
            "- Input: one `VARCHAR` DNA column (case-insensitive)\n"
            "- Returns: `BOOLEAN` (NULL for NULL/non-DNA input)\n"
            "- True if any IUPAC degenerate code (N, R, Y, ...) is present",
        )

    @classmethod
    def compute(
        cls,
        seq: Annotated[pa.StringArray, Param(doc="A DNA sequence")],
    ) -> Annotated[pa.BooleanArray, Returns(pa.bool_())]:
        """Return whether each DNA sequence has degenerate codes (NULL for invalid DNA)."""
        return _map_bool(seq, lambda s: bool(DNA(s).has_degenerates()))


class CountSubsequence(ScalarFunction):
    """Number of (non-overlapping) occurrences of a subsequence within a sequence."""

    class Meta:
        """VGI metadata for the count_subsequence scalar."""

        name = "count_subsequence"
        description = "Count occurrences of a subsequence within a sequence"
        categories = ["sequence"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.count_subsequence('ATGCGATGCATG', 'ATG') AS start_codons",
                description=(
                    "Count how often a motif occurs in a sequence — here the ATG start codon, "
                    "three times. This is the cheap way to screen a table of reads for a "
                    "restriction site, primer binding site, or repeat before spending an "
                    "alignment on them."
                ),
            )
        ]
        tags = _doc(
            "count_subsequence",
            "composition",
            "Scalar function returning how many times a subsequence occurs within a sequence, as a "
            "`BIGINT`. Pass the sequence column first and the subsequence to search for second (both "
            "`VARCHAR`, case-insensitive); NULL input returns NULL. Handy for counting a motif or codon "
            "across reads.",
            "**count_subsequence** — occurrences of a subsequence.\n\n"
            "- Inputs: the sequence, then the subsequence to count (both `VARCHAR`)\n"
            "- Returns: a `BIGINT` occurrence count (NULL for NULL input)\n"
            "- Counts a motif/codon within each sequence",
        )

    @classmethod
    def compute(
        cls,
        seq: Annotated[pa.StringArray, Param(doc="The sequence to search")],
        subsequence: Annotated[pa.StringArray, Param(doc="The subsequence")],
    ) -> Annotated[pa.Int64Array, Returns(pa.int64())]:
        """Return the occurrence count of the subsequence per row (NULL on NULL input)."""
        return _map_pair_int(seq, subsequence, lambda s, sub: int(Sequence(s).count(sub)))


class IsReverseComplement(ScalarFunction):
    """Whether one DNA sequence is the reverse complement of another."""

    class Meta:
        """VGI metadata for the is_reverse_complement scalar."""

        name = "is_reverse_complement"
        description = "True if the second DNA sequence is the reverse complement of the first"
        categories = ["sequence", "nucleotide", "validation"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.is_reverse_complement('ATGC', 'GCAT') AS same_locus",
                description=(
                    "Decide whether two reads are the same sequence seen from opposite strands, "
                    "in one predicate rather than reverse-complementing one side and comparing. "
                    "Use it to collapse strand-duplicate reads or to check a primer pair points "
                    "the right way."
                ),
            )
        ]
        tags = _doc(
            "is_reverse_complement",
            "validation",
            "Scalar predicate returning `BOOLEAN` true when the second DNA sequence is exactly the reverse "
            "complement of the first. Pass two `VARCHAR` DNA columns (case-insensitive); a NULL or non-DNA "
            "pair returns NULL. Use it to test whether two reads come from opposite strands.",
            "**is_reverse_complement** — are two DNA sequences reverse complements?\n\n"
            "- Inputs: two `VARCHAR` DNA columns (case-insensitive)\n"
            "- Returns: `BOOLEAN` (NULL for NULL/non-DNA input)\n"
            "- True when the second is the reverse complement of the first",
        )

    @classmethod
    def compute(
        cls,
        seq1: Annotated[pa.StringArray, Param(doc="First DNA sequence")],
        seq2: Annotated[pa.StringArray, Param(doc="Second DNA sequence")],
    ) -> Annotated[pa.BooleanArray, Returns(pa.bool_())]:
        """Return whether seq2 is the reverse complement of seq1 (NULL on NULL/invalid)."""
        return _map_pair_bool(seq1, seq2, lambda a, b: bool(DNA(a).is_reverse_complement(DNA(b))))


class MismatchCount(ScalarFunction):
    """Number of positions at which two equal-length sequences differ."""

    class Meta:
        """VGI metadata for the mismatch_count scalar."""

        name = "mismatch_count"
        description = "Number of differing positions between two equal-length sequences"
        categories = ["sequence", "distance"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.mismatch_count('ACGTACGT', 'ACGAACGT') AS mismatches",
                description=(
                    "Count differing positions between a read and its reference as a whole "
                    "number, which is what edit-distance thresholds are usually expressed in "
                    "('accept up to 2 mismatches'). hamming_distance gives the same comparison "
                    "as a length-normalised fraction."
                ),
            )
        ]
        tags = _doc(
            "mismatch_count",
            "distance",
            "Scalar function returning the number of positions at which two equal-length sequences differ, "
            "as a `BIGINT` (the raw count behind the Hamming distance). Pass two `VARCHAR` sequence columns "
            "(case-insensitive); NULL input or a length mismatch returns NULL. Use `match_count` for the "
            "agreeing positions or `hamming_distance` for the fraction.",
            "**mismatch_count** — differing positions between two equal-length sequences.\n\n"
            "- Inputs: two `VARCHAR` sequence columns of equal length\n"
            "- Returns: a `BIGINT` mismatch count (NULL if NULL or lengths differ)\n"
            "- The count behind `hamming_distance`; see also `match_count`",
        )

    @classmethod
    def compute(
        cls,
        seq1: Annotated[pa.StringArray, Param(doc="First sequence")],
        seq2: Annotated[pa.StringArray, Param(doc="Second sequence (same length)")],
    ) -> Annotated[pa.Int64Array, Returns(pa.int64())]:
        """Return the mismatch count per row (NULL on NULL or length mismatch)."""
        return _map_pair_int(seq1, seq2, lambda a, b: int(Sequence(a).mismatch_frequency(Sequence(b))))


class MatchCount(ScalarFunction):
    """Number of positions at which two equal-length sequences agree."""

    class Meta:
        """VGI metadata for the match_count scalar."""

        name = "match_count"
        description = "Number of agreeing positions between two equal-length sequences"
        categories = ["sequence", "distance"]
        examples = [
            FunctionExample(
                sql="SELECT skbio.sequence.match_count('ACGTACGT', 'ACGAACGT') AS matches",
                description=(
                    "Count agreeing positions between two equal-length sequences — the "
                    "complement of mismatch_count (the two always sum to the length). Divide by "
                    "the length for percent identity, the number most similarity thresholds are "
                    "quoted in."
                ),
            )
        ]
        tags = _doc(
            "match_count",
            "distance",
            "Scalar function returning the number of positions at which two equal-length sequences agree, "
            "as a `BIGINT`. Pass two `VARCHAR` sequence columns (case-insensitive); NULL input or a length "
            "mismatch returns NULL. It is the complement of `mismatch_count` (the two sum to the length).",
            "**match_count** — agreeing positions between two equal-length sequences.\n\n"
            "- Inputs: two `VARCHAR` sequence columns of equal length\n"
            "- Returns: a `BIGINT` match count (NULL if NULL or lengths differ)\n"
            "- The complement of `mismatch_count`",
        )

    @classmethod
    def compute(
        cls,
        seq1: Annotated[pa.StringArray, Param(doc="First sequence")],
        seq2: Annotated[pa.StringArray, Param(doc="Second sequence (same length)")],
    ) -> Annotated[pa.Int64Array, Returns(pa.int64())]:
        """Return the match count per row (NULL on NULL or length mismatch)."""
        return _map_pair_int(seq1, seq2, lambda a, b: int(Sequence(a).match_frequency(Sequence(b))))


SEQUENCE_FUNCTIONS: list[type] = [
    GcContent,
    GcFrequency,
    ReverseComplement,
    Complement,
    Transcribe,
    ReverseTranscribe,
    Translate,
    Degap,
    IsValidDna,
    IsValidProtein,
    HasGaps,
    HasDegenerates,
    IsReverseComplement,
    CountSubsequence,
    HammingDistance,
    MismatchCount,
    MatchCount,
]
