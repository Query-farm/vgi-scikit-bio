"""Unit tests for phylogenetic diversity and rarefaction."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pyarrow as pa

from vgi_scikit_bio.phylo import FaithPd, SubsampleCounts, Unifrac

_TREE = "((f1:0.1,f2:0.2):0.3,(f3:0.15,f4:0.25):0.35);"


def _feature() -> pa.Table:
    return pa.table(
        {
            "sample_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "feature_id": ["f1", "f2", "f3", "f4", "f1", "f3"],
            "count": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def _args(**kw: object) -> SimpleNamespace:
    base = {"sample": "sample_id", "feature": "feature_id", "count": "count", "tree": _TREE}
    base.update(kw)
    return SimpleNamespace(**base)


class TestFaithPd:
    def test_per_sample(self) -> None:
        out = FaithPd.encode(_feature(), _args())
        assert out["sample_id"] == ["s1", "s2", "s3"]
        # s1 spans f1+f2 subtree = 0.1+0.2+0.3 = 0.6
        assert math.isclose(out["faith_pd"][0], 0.6)

    def test_missing_tree_errors(self) -> None:
        try:
            FaithPd.encode(_feature(), _args(tree=""))
        except ValueError as e:
            assert "tree" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected a ValueError for a missing tree")


class TestUnifrac:
    def test_full_matrix(self) -> None:
        out = Unifrac.encode(_feature(), _args(weighted=False))
        assert len(out["distance"]) == 9  # 3 samples -> full matrix
        cells = {(a, b): d for a, b, d in zip(out["id_1"], out["id_2"], out["distance"], strict=True)}
        assert cells[("s1", "s1")] == 0.0
        # s1 (f1,f2) and s2 (f3,f4) share no branches -> unweighted UniFrac 1.0
        assert math.isclose(cells[("s1", "s2")], 1.0)

    def test_weighted(self) -> None:
        out = Unifrac.encode(_feature(), _args(weighted=True))
        assert len(out["distance"]) == 9


class TestSubsampleCounts:
    def _table(self) -> pa.Table:
        return pa.table(
            {
                "sample_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
                "feature_id": ["a", "b", "c", "a", "b", "c"],
                "count": [4.0, 2.0, 6.0, 10.0, 5.0, 5.0],
            }
        )

    def _sargs(self, **kw: object) -> SimpleNamespace:
        base = {
            "sample": "sample_id",
            "feature": "feature_id",
            "count": "count",
            "depth": 8,
            "with_replacement": False,
            "seed": 0,
        }
        base.update(kw)
        return SimpleNamespace(**base)

    def test_rarefies_to_depth(self) -> None:
        out = SubsampleCounts.encode(self._table(), self._sargs())
        totals: dict[str, int] = {}
        for s, c in zip(out["sample_id"], out["count"], strict=True):
            totals[s] = totals.get(s, 0) + c
        assert all(t == 8 for t in totals.values())  # each sample summed to the depth

    def test_deterministic(self) -> None:
        out1 = SubsampleCounts.encode(self._table(), self._sargs())
        out2 = SubsampleCounts.encode(self._table(), self._sargs())
        assert out1["count"] == out2["count"]  # fixed seed

    def test_drops_shallow_samples(self) -> None:
        t = pa.table({"sample_id": ["s1"], "feature_id": ["a"], "count": [3.0]})
        out = SubsampleCounts.encode(t, self._sargs(depth=8))
        assert out["sample_id"] == []  # total 3 < depth 8 -> dropped
