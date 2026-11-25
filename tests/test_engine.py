"""Accounting and timing tests for the backtest engine."""

import pandas as pd
import pytest

from autobacktest.engine import BacktestEngine


class FixedSignals:
    def __init__(self, targets):
        self.targets = targets

    def generate_signals(self, data):
        return pd.Series(self.targets, index=data.index, dtype="int8", name="target_position")


def market_data():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "open": [10.0, 12.0, 12.5, 13.0],
            "high": [11.0, 12.5, 13.0, 13.5],
            "low": [9.5, 11.5, 12.0, 12.5],
            "close": [10.5, 12.2, 12.8, 13.2],
            "volume": [100.0] * 4,
        }
    )


def test_signals_fill_on_next_candle_with_fees_and_slippage():
    result = BacktestEngine(initial_cash=1000, fee_rate=0.01, slippage_rate=0.01).run(
        market_data(), FixedSignals([1, 1, 0, 0])
    )

    assert result.trades["side"].tolist() == ["buy", "sell"]
    assert result.trades["timestamp"].tolist() == [
        pd.Timestamp("2024-01-01 01:00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 03:00:00", tz="UTC"),
    ]
    assert result.trades.iloc[0]["price"] == pytest.approx(12.12)
    assert result.trades.iloc[1]["price"] == pytest.approx(12.87)
    assert (result.trades["fee"] > 0).all()
    assert result.final_cash == pytest.approx(result.final_equity)
    assert result.equity_curve.iloc[0]["position"] == 0
    assert result.equity_curve.iloc[1]["position"] == 1
    assert result.equity_curve.iloc[3]["position"] == 0


def test_flat_strategy_does_not_trade_and_preserves_cash():
    result = BacktestEngine(initial_cash=500).run(market_data(), FixedSignals([0, 0, 0, 0]))

    assert result.trades.empty
    assert result.final_cash == 500
    assert result.final_equity == 500
    assert result.total_return_pct == 0


def test_invalid_target_positions_are_rejected():
    with pytest.raises(ValueError, match="0/1"):
        BacktestEngine().run(market_data(), FixedSignals([0, 2, 0, 0]))


@pytest.mark.parametrize("kwargs", [{"initial_cash": 0}, {"fee_rate": -0.1}, {"slippage_rate": 1}])
def test_invalid_engine_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        BacktestEngine(**kwargs)
