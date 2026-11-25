"""Tests for baseline strategy behavior and signal timing."""

import numpy as np
import pandas as pd
import pytest

from autobacktest.strategies import EMACrossover, Strategy


def test_ema_crossover_outputs_long_flat_target_positions():
    prices = [10, 11, 12, 13, 14, 13, 12, 11, 10, 9]
    frame = pd.DataFrame({"close": prices})
    strategy = EMACrossover(fast_period=2, slow_period=4)

    signals = strategy.generate_signals(frame)

    assert isinstance(strategy, Strategy)
    assert signals.index.equals(frame.index)
    assert signals.name == "target_position"
    assert set(signals.unique()).issubset({0, 1})
    assert signals.iloc[:3].eq(0).all()
    assert signals.iloc[4] == 1
    assert signals.iloc[-1] == 0


def test_signals_are_prefix_stable_no_future_data_leakage():
    frame = pd.DataFrame({"close": [10, 9, 8, 9, 10, 11, 12, 1000]})
    strategy = EMACrossover(fast_period=2, slow_period=3)

    full = strategy.generate_signals(frame)
    prefix = strategy.generate_signals(frame.iloc[:7])

    pd.testing.assert_series_equal(full.iloc[:7], prefix)


@pytest.mark.parametrize("fast,slow", [(0, 3), (2, 2), (4, 2), (True, 3), (2, 1.5)])
def test_invalid_ema_periods_are_rejected(fast, slow):
    with pytest.raises(ValueError):
        EMACrossover(fast, slow)


@pytest.mark.parametrize("close", [[1, 0, 2], [1, np.inf, 2], ["x", 2, 3]])
def test_invalid_close_prices_are_rejected(close):
    with pytest.raises(ValueError):
        EMACrossover(2, 3).generate_signals(pd.DataFrame({"close": close}))


def test_missing_close_column_is_rejected():
    with pytest.raises(ValueError, match="close"):
        EMACrossover(2, 3).generate_signals(pd.DataFrame({"open": [1, 2]}))
