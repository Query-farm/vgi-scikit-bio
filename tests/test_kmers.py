"""Unit tests for the k-mer / residue composition table functions."""

from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest

from vgi_scikit_bio.kmers import KmerFrequencies, ResidueFrequencies, _sequence_column


def _reads() -> pa.Table:
    return pa.table({"id": [1, 2], "seq": ["ATGCGGATTACAGG", "TTGC"]})


def _kmer_args(**kw: object) -> SimpleNamespace:
    base = {"id": "id", "sequence": "seq", "k": 3}
    base.update(kw)
    return SimpleNamespace(**base)


class TestSequenceColumn:
    def test_explicit(self) -> None:
        assert _sequence_column(_reads().schema, "id", "seq") == "seq"

    def test_inferred(self) -> None:
        assert _sequence_column(_reads().schema, "id", "") == "seq"

    def test_ambiguous_errors(self) -> None:
        schema = pa.schema([pa.field("id", pa.int64()), pa.field("a", pa.string()), pa.field("b", pa.string())])
        with pytest.raises(ValueError, match="could not infer"):
            _sequence_column(schema, "id", "")


class TestKmerFrequencies:
    def test_counts(self) -> None:
        out = KmerFrequencies.encode(_reads(), _kmer_args())
        # ATG appears once in the first read
        atg = [(i, v) for i, km, v in zip(out["id"], out["kmer"], out["count"], strict=True) if km == "ATG"]
        assert atg == [(1, 1)]
        assert all(isinstance(v, int) for v in out["count"])

    def test_short_sequence_skipped(self) -> None:
        t = pa.table({"id": [1], "seq": ["AT"]})  # shorter than k=3
        out = KmerFrequencies.encode(t, _kmer_args())
        assert out["kmer"] == []

    def test_k_length(self) -> None:
        out = KmerFrequencies.encode(_reads(), _kmer_args(k=2))
        assert all(len(km) == 2 for km in out["kmer"])


class TestResidueFrequencies:
    def test_base_counts(self) -> None:
        out = ResidueFrequencies.encode(_reads(), SimpleNamespace(id="id", sequence="seq"))
        first = {r: c for i, r, c in zip(out["id"], out["residue"], out["count"], strict=True) if i == 1}
        # ATGCGGATTACAGG has 4 A, 2 C, 5 G, 3 T
        assert first == {"A": 4, "C": 2, "G": 5, "T": 3}
