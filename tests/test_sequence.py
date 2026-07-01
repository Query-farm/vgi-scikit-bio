"""Unit tests for the sequence scalar functions."""

from __future__ import annotations

import math

import pyarrow as pa

from vgi_scikit_bio.sequence import (
    Complement,
    CountSubsequence,
    Degap,
    GcContent,
    GcFrequency,
    HammingDistance,
    HasDegenerates,
    HasGaps,
    IsReverseComplement,
    IsValidDna,
    IsValidProtein,
    MatchCount,
    MismatchCount,
    ReverseComplement,
    ReverseTranscribe,
    Transcribe,
    Translate,
)


def _seqs() -> pa.Array:
    return pa.array(["ATGCGGATTACAGG", "atgc", None, "HELLO WORLD"])


class TestGcContent:
    def test_fraction(self) -> None:
        out = GcContent.compute(pa.array(["GGCC", "ATAT"])).to_pylist()
        assert out == [1.0, 0.0]

    def test_null_and_invalid(self) -> None:
        out = GcContent.compute(_seqs()).to_pylist()
        assert out[2] is None  # NULL stays NULL
        assert out[3] is None  # non-DNA -> NULL
        assert math.isclose(out[0], 0.5)

    def test_case_insensitive(self) -> None:
        out = GcContent.compute(pa.array(["gc", "GC"])).to_pylist()
        assert out[0] == out[1] == 1.0


class TestReverseComplement:
    def test_basic(self) -> None:
        assert ReverseComplement.compute(pa.array(["ATGC"])).to_pylist() == ["GCAT"]

    def test_complement_not_reversed(self) -> None:
        assert Complement.compute(pa.array(["ATGC"])).to_pylist() == ["TACG"]

    def test_invalid_is_null(self) -> None:
        assert ReverseComplement.compute(pa.array(["XYZ"])).to_pylist() == [None]


class TestTranscribeTranslate:
    def test_transcribe(self) -> None:
        assert Transcribe.compute(pa.array(["ATGC"])).to_pylist() == ["AUGC"]

    def test_translate(self) -> None:
        # ATG CGG ATT ACA GGT -> M R I T G
        assert Translate.compute(pa.array(["ATGCGGATTACAGGT"])).to_pylist() == ["MRITG"]

    def test_translate_trailing_partial_codon_ignored(self) -> None:
        # ATG CG (partial) -> M
        assert Translate.compute(pa.array(["ATGCG"])).to_pylist() == ["M"]


class TestValidation:
    def test_valid_dna(self) -> None:
        out = IsValidDna.compute(pa.array(["ATGCN", "HELLO", None])).to_pylist()
        assert out == [True, False, None]

    def test_valid_protein(self) -> None:
        out = IsValidProtein.compute(pa.array(["MRIT", "ATGC1", None])).to_pylist()
        assert out == [True, False, None]


class TestHammingDistance:
    def test_fraction(self) -> None:
        out = HammingDistance.compute(pa.array(["ACGTACGT"]), pa.array(["ACGAACGT"])).to_pylist()
        assert math.isclose(out[0], 0.125)

    def test_length_mismatch_is_null(self) -> None:
        out = HammingDistance.compute(pa.array(["AAAA"]), pa.array(["AAA"])).to_pylist()
        assert out == [None]

    def test_null_is_null(self) -> None:
        out = HammingDistance.compute(pa.array(["AAAA", None]), pa.array([None, "AAAA"])).to_pylist()
        assert out == [None, None]


class TestAdditionalScalars:
    def test_gc_frequency(self) -> None:
        assert GcFrequency.compute(pa.array(["ATGCGGATTACAGG"])).to_pylist() == [7]

    def test_degap(self) -> None:
        assert Degap.compute(pa.array(["AC-GT--A", "MR-IT"])).to_pylist() == ["ACGTA", "MRIT"]

    def test_reverse_transcribe(self) -> None:
        assert ReverseTranscribe.compute(pa.array(["AUGC"])).to_pylist() == ["ATGC"]

    def test_has_gaps(self) -> None:
        assert HasGaps.compute(pa.array(["AC-GT", "ACGT", None])).to_pylist() == [True, False, None]

    def test_has_degenerates(self) -> None:
        assert HasDegenerates.compute(pa.array(["ACGTN", "ACGT"])).to_pylist() == [True, False]

    def test_count_subsequence(self) -> None:
        assert CountSubsequence.compute(pa.array(["ATGCGATGCATG"]), pa.array(["ATG"])).to_pylist() == [3]

    def test_is_reverse_complement(self) -> None:
        out = IsReverseComplement.compute(pa.array(["ATGC", "ATGC"]), pa.array(["GCAT", "ATGC"])).to_pylist()
        assert out == [True, False]

    def test_mismatch_and_match_count(self) -> None:
        mm = MismatchCount.compute(pa.array(["ACGTACGT"]), pa.array(["ACGAACGT"])).to_pylist()
        mt = MatchCount.compute(pa.array(["ACGTACGT"]), pa.array(["ACGAACGT"])).to_pylist()
        assert mm == [1] and mt == [7]

    def test_mismatch_length_mismatch_is_null(self) -> None:
        assert MismatchCount.compute(pa.array(["AAAA"]), pa.array(["AAA"])).to_pylist() == [None]
