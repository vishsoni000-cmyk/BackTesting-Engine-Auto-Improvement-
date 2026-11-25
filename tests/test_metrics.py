"""Tests for equity-curve performance metrics."""

import math

import pandas as pd
import pytest

from autobacktest.metrics import performance_metrics


def test_metrics_calculate_return_and_peak_to_trough_drawdown():
    curve = pd.DataFrame({"equity": [100.0, 110.0, 88.0, 100.0]})

    metrics = performance_metrics(curve, periods_per_year=12)

    assert metrics["total_return_pct"] == pytest.approx(0.0)
    assert metrics["max_drawdown_pct"] == pytest.approx(-20.0)
    assert metrics["annualized_volatility_pct"] > 0
    assert math.isfinite(metrics["sharpe_ratio"])


def test_metrics_do_not_mutate_equity_curve():
    curve = pd.DataFrame({"equity": [100.0, 101.0, 102.0]})
    original = curve.copy()

    performance_metrics(curve)

    pd.testing.assert_frame_equal(curve, original)


@pytest.mark.parametrize("equity", [[100.0], [100.0, 0.0], [100.0, float("nan")]])
def test_invalid_equity_curves_are_rejected(equity):
    with pytest.raises(ValueError):
        performance_metrics(pd.DataFrame({"equity": equity}))


def test_missing_equity_column_is_rejected():
    with pytest.raises(ValueError, match="equity"):
        performance_metrics(pd.DataFrame({"cash": [100, 101]}))
