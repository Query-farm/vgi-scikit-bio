"""Unit tests for ordination, distance tests, and compositional transforms."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pyarrow as pa
import pytest

from vgi_scikit_bio.composition import Clr, Ilr
from vgi_scikit_bio.distance_stats import Anosim, Mantel, Permanova
from vgi_scikit_bio.ordination import Pcoa


def _square_dm() -> pa.Table:
    return pa.table(
        {
            "id_1": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "id_2": ["a", "b", "c", "a", "b", "c", "a", "b", "c"],
            "distance": [0.0, 0.5, 0.7, 0.5, 0.0, 0.6, 0.7, 0.6, 0.0],
        }
    )


class TestPcoa:
    def test_shape(self) -> None:
        out = Pcoa.encode(_square_dm(), SimpleNamespace(id_1="id_1", id_2="id_2", distance="distance", n_components=2))
        assert out["sample_id"] == ["a", "b", "c"]
        assert len(out["pc_1"]) == 3 and len(out["pc_2"]) == 3

    def test_more_components_than_available_padded_null(self) -> None:
        # 3 samples -> at most 2 informative axes; a 5th is NULL-padded
        out = Pcoa.encode(_square_dm(), SimpleNamespace(id_1="id_1", id_2="id_2", distance="distance", n_components=5))
        assert set(out) == {"sample_id", "pc_1", "pc_2", "pc_3", "pc_4", "pc_5"}
        assert out["pc_5"] == [None, None, None]


def _grouped_dm() -> pa.Table:
    # a full-square 4-sample matrix (every sample appears as id_1) with a group column
    ids = ["s1", "s2", "s3", "s4"]
    groups = {"s1": "x", "s2": "x", "s3": "y", "s4": "y"}
    dist = {
        ("s1", "s2"): 0.2,
        ("s1", "s3"): 0.8,
        ("s1", "s4"): 0.9,
        ("s2", "s3"): 0.7,
        ("s2", "s4"): 0.85,
        ("s3", "s4"): 0.3,
    }
    id1, id2, d, grp = [], [], [], []
    for a in ids:
        for b in ids:
            id1.append(a)
            id2.append(b)
            d.append(0.0 if a == b else dist.get((a, b), dist.get((b, a))))
            grp.append(groups[a])
    return pa.table({"id_1": id1, "id_2": id2, "distance": d, "grp": grp})


class TestGroupedTests:
    def _args(self, **kw: object) -> SimpleNamespace:
        base = {"id_1": "id_1", "id_2": "id_2", "distance": "distance", "group": "grp", "permutations": 99}
        base.update(kw)
        return SimpleNamespace(**base)

    def test_permanova_result(self) -> None:
        out = Permanova.encode(_grouped_dm(), self._args())
        assert out["method"] == ["PERMANOVA"]
        assert out["sample_size"] == [4]
        assert out["number_of_groups"] == [2]
        assert out["test_statistic"][0] > 0

    def test_anosim_result(self) -> None:
        out = Anosim.encode(_grouped_dm(), self._args())
        assert out["method"] == ["ANOSIM"]
        assert -1.0 <= out["test_statistic"][0] <= 1.0

    def test_missing_group_errors(self) -> None:
        # a condensed (no-diagonal) matrix leaves the last sample without an id_1 row
        t = pa.table(
            {
                "id_1": ["a", "a", "b"],
                "id_2": ["b", "c", "c"],
                "distance": [0.5, 0.7, 0.6],
                "grp": ["x", "x", "x"],
            }
        )
        with pytest.raises(ValueError, match="no group label"):
            Permanova.encode(t, self._args())


class TestMantel:
    def test_correlation(self) -> None:
        t = pa.table(
            {
                "id_1": ["a", "a", "b"],
                "id_2": ["b", "c", "c"],
                "distance_x": [0.5, 0.7, 0.6],
                "distance_y": [0.4, 0.9, 0.5],
            }
        )
        args = SimpleNamespace(
            id_1="id_1",
            id_2="id_2",
            distance_x="distance_x",
            distance_y="distance_y",
            method="pearson",
            permutations=99,
        )
        out = Mantel.encode(t, args)
        assert out["n"] == [3]
        assert -1.0 <= out["correlation"][0] <= 1.0


def _composition() -> pa.Table:
    return pa.table(
        {
            "sample_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "feature_id": ["a", "b", "c", "a", "b", "c"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


class TestComposition:
    def _args(self, **kw: object) -> SimpleNamespace:
        base = {"sample": "sample_id", "feature": "feature_id", "value": "value", "pseudocount": 0.0}
        base.update(kw)
        return SimpleNamespace(**base)

    def test_clr_same_shape(self) -> None:
        out = Clr.encode(_composition(), self._args())
        assert len(out["clr"]) == 6  # 2 samples x 3 features

    def test_clr_row_sums_to_zero(self) -> None:
        out = Clr.encode(_composition(), self._args())
        s1 = [v for s, v in zip(out["sample_id"], out["clr"], strict=True) if s == "s1"]
        assert math.isclose(sum(s1), 0.0, abs_tol=1e-9)

    def test_ilr_reduces_dimension(self) -> None:
        out = Ilr.encode(_composition(), self._args())
        # 3 features -> 2 ILR components per sample
        s1 = [c for s, c in zip(out["sample_id"], out["component"], strict=True) if s == "s1"]
        assert sorted(s1) == [1, 2]

    def test_pseudocount_handles_zeros(self) -> None:
        t = pa.table(
            {
                "sample_id": ["s1", "s1", "s1"],
                "feature_id": ["a", "b", "c"],
                "value": [0.0, 2.0, 3.0],
            }
        )
        out = Clr.encode(t, self._args(pseudocount=1.0))
        assert all(v is not None and not math.isnan(v) for v in out["clr"])
