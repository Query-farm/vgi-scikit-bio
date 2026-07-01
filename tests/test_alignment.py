"""Unit tests for the pairwise alignment functions."""

from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa

from vgi_scikit_bio.alignment import (
    AlignScoreNucleotide,
    AlignScoreProtein,
    PairwiseAlignNucleotide,
    PairwiseAlignProtein,
)


class TestScores:
    def test_nucleotide_score(self) -> None:
        out = AlignScoreNucleotide.compute(pa.array(["ACTGGT"]), pa.array(["ACTGT"])).to_pylist()
        assert out == [2.0]

    def test_identical_scores_higher(self) -> None:
        same = AlignScoreNucleotide.compute(pa.array(["ACGTACGT"]), pa.array(["ACGTACGT"])).to_pylist()[0]
        diff = AlignScoreNucleotide.compute(pa.array(["ACGTACGT"]), pa.array(["TTTTTTTT"])).to_pylist()[0]
        assert same > diff

    def test_protein_score(self) -> None:
        out = AlignScoreProtein.compute(pa.array(["MRITMK"]), pa.array(["MRIMK"])).to_pylist()
        assert isinstance(out[0], float)

    def test_null_and_invalid(self) -> None:
        out = AlignScoreNucleotide.compute(pa.array(["ACGT", None]), pa.array([None, "ACGT"])).to_pylist()
        assert out == [None, None]


class TestPairwiseAlign:
    def _args(self, **kw: object) -> SimpleNamespace:
        base = {"id": "id", "seq1": "ref", "seq2": "read", "mode": "global"}
        base.update(kw)
        return SimpleNamespace(**base)

    def _pairs(self) -> pa.Table:
        return pa.table({"id": [1, 2], "ref": ["ACTGGT", "GGGG"], "read": ["ACTGT", "GGG"]})

    def test_global_aligns_with_gaps(self) -> None:
        out = PairwiseAlignNucleotide.encode(self._pairs(), self._args())
        assert out["aligned_1"][0] == "ACTGGT"
        assert out["aligned_2"][0] == "ACTGT-"
        assert out["score"][0] == 2.0
        assert out["length"][0] == 6

    def test_local_mode(self) -> None:
        out = PairwiseAlignNucleotide.encode(self._pairs(), self._args(mode="local"))
        # aligned strings have equal length and no leading/trailing overhang past the match
        assert len(out["aligned_1"][0]) == len(out["aligned_2"][0])

    def test_null_pair_is_null(self) -> None:
        t = pa.table({"id": [1], "ref": [None], "read": ["ACGT"]})
        out = PairwiseAlignNucleotide.encode(t, self._args())
        assert out["aligned_1"] == [None] and out["score"] == [None]

    def test_protein_alignment(self) -> None:
        t = pa.table({"id": [1], "ref": ["MRITMK"], "read": ["MRIMK"]})
        out = PairwiseAlignProtein.encode(t, self._args())
        assert len(out["aligned_1"][0]) == len(out["aligned_2"][0])
        assert isinstance(out["score"][0], float)
