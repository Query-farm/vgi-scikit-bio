"""Unit tests for tree construction and Newick inspection."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pyarrow as pa

from vgi_scikit_bio.tree import NeighborJoining, TipCount, TotalBranchLength


def _dm() -> pa.Table:
    return pa.table(
        {
            "id_1": ["a", "a", "a", "b", "b", "c"],
            "id_2": ["b", "c", "d", "c", "d", "d"],
            "distance": [5.0, 9.0, 9.0, 10.0, 10.0, 8.0],
        }
    )


class TestNeighborJoining:
    def test_newick_string(self) -> None:
        out = NeighborJoining.encode(_dm(), SimpleNamespace(id_1="id_1", id_2="id_2", distance="distance"))
        newick = out["newick"][0]
        assert newick.endswith(";")
        # all four taxa appear
        for taxon in ("a", "b", "c", "d"):
            assert taxon in newick


class TestNewickScalars:
    NEWICK = "((a:2.0,b:3.0):3.0,d:4.0,c:4.0);"

    def test_tip_count(self) -> None:
        out = TipCount.compute(pa.array([self.NEWICK, None, "not a tree"])).to_pylist()
        assert out == [4, None, None]

    def test_total_branch_length(self) -> None:
        out = TotalBranchLength.compute(pa.array([self.NEWICK])).to_pylist()
        assert math.isclose(out[0], 16.0)

    def test_null_is_null(self) -> None:
        assert TotalBranchLength.compute(pa.array([None])).to_pylist() == [None]
