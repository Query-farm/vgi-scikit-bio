"""Unit tests for ordination, distance tests, and compositional transforms."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pyarrow as pa
import pytest

from vgi_scikit_bio.composition import COMPOSITION_FUNCTIONS, Ancom, Clr, DirmultTtest, Ilr
from vgi_scikit_bio.distance_stats import Anosim, Mantel, Permanova
from vgi_scikit_bio.ordination import Pcoa

_COMP = {c.Meta.name: c for c in COMPOSITION_FUNCTIONS}


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

    def test_full_function_coverage(self) -> None:
        expected = {
            "clr",
            "ilr",
            "closure",
            "centralize",
            "rclr",
            "multi_replace",
            "power",
            "alr",
            "clr_inv",
            "ilr_inv",
            "alr_inv",
            "pairwise_vlr",
            "ancom",
            "dirmult_ttest",
        }
        assert expected <= set(_COMP)

    def test_closure_normalizes_to_proportions(self) -> None:
        out = _COMP["closure"].encode(_composition(), self._args())
        s1 = [v for s, v in zip(out["sample_id"], out["proportion"], strict=True) if s == "s1"]
        assert math.isclose(sum(s1), 1.0)

    def test_alr_reduces_dimension(self) -> None:
        out = _COMP["alr"].encode(_composition(), self._args(ref_idx=0))
        s1 = [c for s, c in zip(out["sample_id"], out["component"], strict=True) if s == "s1"]
        assert sorted(s1) == [1, 2]  # 3 features -> 2 ALR components

    def test_clr_inv_roundtrips(self) -> None:
        clr = Clr.encode(_composition(), self._args())
        coords = pa.table({"sample_id": clr["sample_id"], "feature_id": clr["feature_id"], "value": clr["clr"]})
        back = _COMP["clr_inv"].encode(coords, self._args())
        s1 = [v for s, v in zip(back["sample_id"], back["value"], strict=True) if s == "s1"]
        assert math.isclose(sum(s1), 1.0)  # inverse gives proportions

    def test_power_transform(self) -> None:
        out = _COMP["power"].encode(_composition(), self._args(power=2.0))
        assert len(out["value"]) == 6

    def test_pairwise_vlr_matrix(self) -> None:
        t = pa.table(
            {
                "sample_id": ["s1", "s1", "s1", "s2", "s2", "s2", "s3", "s3", "s3"],
                "feature_id": ["a", "b", "c"] * 3,
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 1.0, 7.0],
            }
        )
        out = _COMP["pairwise_vlr"].encode(t, self._args())
        assert len(out["vlr"]) == 9  # 3x3 feature matrix
        diag = [v for a, b, v in zip(out["feature_1"], out["feature_2"], out["vlr"], strict=True) if a == b]
        assert diag == [0.0, 0.0, 0.0]


def _diff_table() -> pa.Table:
    rows = [
        ("s1", "b1", 12),
        ("s1", "b2", 11),
        ("s1", "b3", 10),
        ("s2", "b1", 9),
        ("s2", "b2", 11),
        ("s2", "b3", 12),
        ("s3", "b1", 1),
        ("s3", "b2", 11),
        ("s3", "b3", 10),
        ("s4", "b1", 22),
        ("s4", "b2", 21),
        ("s4", "b3", 9),
        ("s5", "b1", 20),
        ("s5", "b2", 22),
        ("s5", "b3", 10),
        ("s6", "b1", 23),
        ("s6", "b2", 21),
        ("s6", "b3", 14),
    ]
    grp = {"s1": "x", "s2": "x", "s3": "x", "s4": "y", "s5": "y", "s6": "y"}
    return pa.table(
        {
            "sample_id": [r[0] for r in rows],
            "feature_id": [r[1] for r in rows],
            "count": [float(r[2]) for r in rows],
            "grp": [grp[r[0]] for r in rows],
        }
    )


class TestDifferentialAbundance:
    def _args(self) -> SimpleNamespace:
        return SimpleNamespace(sample="sample_id", feature="feature_id", value="count", group="grp")

    def test_ancom_per_feature(self) -> None:
        out = Ancom.encode(_diff_table(), self._args())
        assert out["feature_id"] == ["b1", "b2", "b3"]
        assert all(isinstance(w, int) for w in out["w"])
        assert all(isinstance(s, bool) for s in out["significant"])

    def test_dirmult_ttest_deterministic(self) -> None:
        a = self._args()
        out1 = DirmultTtest.encode(_diff_table(), a)
        out2 = DirmultTtest.encode(_diff_table(), a)
        assert out1["log2_fold_change"] == out2["log2_fold_change"]  # fixed seed
        assert set(out1) == {"feature_id", "t_statistic", "log2_fold_change", "pvalue", "qvalue", "significant"}
