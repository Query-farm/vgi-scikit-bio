"""Unit tests for alpha aggregates and the beta-diversity table function."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pyarrow as pa

from tests.harness import run_alpha
from vgi_scikit_bio.diversity import ALPHA_FUNCTIONS, BetaDiversity

_ALPHA = {cls.Meta.name: cls for cls in ALPHA_FUNCTIONS}


def _run_order(cls: type, counts: list[float], order: float) -> float:
    states = {0: cls.initial_state(None)}
    cls.update(states, pa.array([0] * len(counts), pa.int64()), pa.array(counts, pa.float64()), order)
    return cls.finalize(pa.array([0], pa.int64()), states, None).column("result").to_pylist()[0]


def _run_list(cls: type, counts: list[float]) -> list[float]:
    states = {0: cls.initial_state(None)}
    cls.update(states, pa.array([0] * len(counts), pa.int64()), pa.array(counts, pa.float64()))
    return cls.finalize(pa.array([0], pa.int64()), states, None).column("result").to_pylist()[0]


class TestAlpha:
    def test_full_metric_coverage(self) -> None:
        # every scikit-bio non-phylogenetic, deterministic metric is exposed
        assert {"shannon", "simpson", "chao1", "ace", "fisher_alpha", "margalef", "gini_index"} <= set(_ALPHA)
        assert len(_ALPHA) >= 30

    def test_shannon_matches_skbio(self) -> None:
        assert math.isclose(run_alpha(_ALPHA["shannon"], [4, 2, 1])[0], 0.9556998911125343)

    def test_observed_features_is_richness(self) -> None:
        assert run_alpha(_ALPHA["observed_features"], [4, 0, 1, 0, 3])[0] == 3.0

    def test_grouped_per_sample(self) -> None:
        out = run_alpha(_ALPHA["shannon"], [4, 2, 1, 1, 9], group_ids=[1, 1, 1, 2, 2])
        assert set(out) == {1, 2}
        assert out[1] > out[2]

    def test_dominance_complements_simpson(self) -> None:
        counts = [4, 2, 1]
        assert math.isclose(run_alpha(_ALPHA["simpson"], counts)[0] + run_alpha(_ALPHA["dominance"], counts)[0], 1.0)

    def test_chao1_ge_observed(self) -> None:
        counts = [4, 2, 1, 1]
        assert run_alpha(_ALPHA["chao1"], counts)[0] >= run_alpha(_ALPHA["observed_features"], counts)[0]

    def test_every_scalar_metric_runs(self) -> None:
        # each generated scalar metric produces a finite-or-NaN float without error
        counts = [4, 2, 1, 0, 3, 1, 1]
        for name in ("ace", "berger_parker_d", "fisher_alpha", "margalef", "menhinick", "strong", "goods_coverage"):
            val = run_alpha(_ALPHA[name], counts)[0]
            assert isinstance(val, float)

    def test_hill_order_parameter(self) -> None:
        counts = [4, 2, 1, 3, 1, 1]
        q0 = _run_order(_ALPHA["hill"], counts, 0.0)
        q2 = _run_order(_ALPHA["hill"], counts, 2.0)
        # Hill q=0 is richness (>= any higher-order effective count)
        assert q0 >= q2

    def test_renyi_tsallis_run(self) -> None:
        counts = [4, 2, 1, 3]
        assert isinstance(_run_order(_ALPHA["renyi"], counts, 1.0), float)
        assert isinstance(_run_order(_ALPHA["tsallis"], counts, 2.0), float)

    def test_list_metrics_return_arrays(self) -> None:
        counts = [4, 2, 1, 0, 3, 1, 1]
        assert len(_run_list(_ALPHA["chao1_ci"], counts)) == 2
        assert len(_run_list(_ALPHA["esty_ci"], counts)) == 2
        osd = _run_list(_ALPHA["osd"], counts)
        assert osd == [6.0, 3.0, 1.0]  # observed, singles, doubles

    def test_empty_group_is_null(self) -> None:
        shannon = _ALPHA["shannon"]
        states = {0: shannon.initial_state(None)}
        shannon.update(states, pa.array([0], type=pa.int64()), pa.array([None], type=pa.float64()))
        batch = shannon.finalize(pa.array([0], type=pa.int64()), states, None)
        assert batch.column("result").to_pylist() == [None]


def _feature_table() -> pa.Table:
    return pa.table(
        {
            "sample_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "feature_id": ["a", "b", "a", "b", "a", "b"],
            "count": [4.0, 2.0, 1.0, 9.0, 0.0, 5.0],
        }
    )


def _beta_args(**kw: object) -> SimpleNamespace:
    base = {"sample": "sample_id", "feature": "feature_id", "count": "count", "metric": "braycurtis"}
    base.update(kw)
    return SimpleNamespace(**base)


class TestBetaDiversity:
    def test_full_square_matrix(self) -> None:
        out = BetaDiversity.encode(_feature_table(), _beta_args())
        # 3 samples -> 9 ordered pairs
        assert len(out["id_1"]) == 9

    def test_diagonal_is_zero(self) -> None:
        out = BetaDiversity.encode(_feature_table(), _beta_args())
        diag = [d for a, b, d in zip(out["id_1"], out["id_2"], out["distance"], strict=True) if a == b]
        assert diag == [0.0, 0.0, 0.0]

    def test_symmetric(self) -> None:
        out = BetaDiversity.encode(_feature_table(), _beta_args())
        cells = {(a, b): d for a, b, d in zip(out["id_1"], out["id_2"], out["distance"], strict=True)}
        assert math.isclose(cells[("s1", "s2")], cells[("s2", "s1")])

    def test_jaccard_metric(self) -> None:
        out = BetaDiversity.encode(_feature_table(), _beta_args(metric="jaccard"))
        assert len(out["distance"]) == 9
