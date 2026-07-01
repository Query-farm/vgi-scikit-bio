"""Unit tests for alpha aggregates and the beta-diversity table function."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pyarrow as pa

from tests.harness import run_alpha
from vgi_scikit_bio.diversity import (
    BetaDiversity,
    Chao1,
    Dominance,
    ObservedFeatures,
    Shannon,
    Simpson,
)


class TestAlpha:
    def test_shannon_matches_skbio(self) -> None:
        # counts [4, 2, 1] -> natural-log Shannon entropy
        result = run_alpha(Shannon, [4, 2, 1])[0]
        assert math.isclose(result, 0.9556998911125343)

    def test_observed_features_is_richness(self) -> None:
        assert run_alpha(ObservedFeatures, [4, 0, 1, 0, 3])[0] == 3.0

    def test_grouped_per_sample(self) -> None:
        out = run_alpha(Shannon, [4, 2, 1, 1, 9], group_ids=[1, 1, 1, 2, 2])
        assert set(out) == {1, 2}
        assert out[1] > out[2]  # sample 1 is more even/diverse

    def test_simpson_in_unit_interval(self) -> None:
        assert 0.0 <= run_alpha(Simpson, [4, 2, 1])[0] <= 1.0

    def test_dominance_complements_simpson(self) -> None:
        counts = [4, 2, 1]
        assert math.isclose(run_alpha(Simpson, counts)[0] + run_alpha(Dominance, counts)[0], 1.0)

    def test_chao1_ge_observed(self) -> None:
        counts = [4, 2, 1, 1]
        assert run_alpha(Chao1, counts)[0] >= run_alpha(ObservedFeatures, counts)[0]

    def test_empty_group_is_null(self) -> None:
        # a group with only NULL counts scores NULL
        states = {0: Shannon.initial_state(None)}
        Shannon.update(states, pa.array([0], type=pa.int64()), pa.array([None], type=pa.float64()))
        batch = Shannon.finalize(pa.array([0], type=pa.int64()), states, None)
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
