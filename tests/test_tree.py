"""Unit tests for tree construction, inspection, and comparison."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pyarrow as pa

from vgi_scikit_bio.tree import TREE_FUNCTIONS

_TREE = {c.Meta.name: c for c in TREE_FUNCTIONS}


def _dm() -> pa.Table:
    return pa.table(
        {
            "id_1": ["a", "a", "a", "b", "b", "c"],
            "id_2": ["b", "c", "d", "c", "d", "d"],
            "distance": [5.0, 9.0, 9.0, 10.0, 10.0, 8.0],
        }
    )


class TestBuilders:
    def test_full_builder_coverage(self) -> None:
        assert {"neighbor_joining", "upgma", "gme", "bme"} <= set(_TREE)

    def test_each_builder_makes_a_tree(self) -> None:
        args = SimpleNamespace(id_1="id_1", id_2="id_2", distance="distance")
        for name in ("neighbor_joining", "upgma", "gme", "bme"):
            newick = _TREE[name].encode(_dm(), args)["newick"][0]
            assert newick.endswith(";")
            for taxon in ("a", "b", "c", "d"):
                assert taxon in newick

    def test_upgma_is_ultrametric(self) -> None:
        args = SimpleNamespace(id_1="id_1", id_2="id_2", distance="distance")
        newick = _TREE["upgma"].encode(_dm(), args)["newick"][0]
        # all tips equidistant from the root -> tree_height well-defined and > 0
        h = _TREE["tree_height"].compute(pa.array([newick])).to_pylist()[0]
        assert h > 0


class TestInspection:
    NEWICK = "((a:2.0,b:3.0):3.0,d:4.0,c:4.0);"

    def test_tip_count(self) -> None:
        out = _TREE["tip_count"].compute(pa.array([self.NEWICK, None, "not a tree"])).to_pylist()
        assert out == [4, None, None]

    def test_total_branch_length(self) -> None:
        out = _TREE["total_branch_length"].compute(pa.array([self.NEWICK])).to_pylist()
        assert math.isclose(out[0], 16.0)

    def test_tree_height(self) -> None:
        out = _TREE["tree_height"].compute(pa.array([self.NEWICK])).to_pylist()
        assert math.isclose(out[0], 6.0)  # deepest tip: a at 2 + 3 + ... -> 6

    def test_null_is_null(self) -> None:
        assert _TREE["total_branch_length"].compute(pa.array([None])).to_pylist() == [None]


class TestComparison:
    def test_robinson_foulds(self) -> None:
        rf = _TREE["robinson_foulds"]
        same = rf.compute(pa.array(["((a,b),(c,d));"]), pa.array(["((a,b),(c,d));"])).to_pylist()[0]
        diff = rf.compute(pa.array(["((a,b),(c,d));"]), pa.array(["((a,c),(b,d));"])).to_pylist()[0]
        assert same == 0.0
        assert diff > 0.0

    def test_weighted_robinson_foulds(self) -> None:
        wrf = _TREE["weighted_robinson_foulds"]
        out = wrf.compute(
            pa.array(["((a:1,b:1):1,(c:1,d:1):1);"]), pa.array(["((a:1,c:1):1,(b:1,d:1):1);"])
        ).to_pylist()
        assert out[0] > 0.0

    def test_cophenetic_distance(self) -> None:
        cd = _TREE["cophenetic_distance"]
        same = cd.compute(
            pa.array(["((a:1,b:1):1,(c:1,d:1):1);"]), pa.array(["((a:1,b:1):1,(c:1,d:1):1);"])
        ).to_pylist()[0]
        assert math.isclose(same, 0.0)

    def test_null_is_null(self) -> None:
        out = _TREE["robinson_foulds"].compute(pa.array([None]), pa.array(["((a,b),(c,d));"])).to_pylist()
        assert out == [None]
