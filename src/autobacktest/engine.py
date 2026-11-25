"""Deterministic spot backtest execution with conservative fill timing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from autobacktest.ohlcv import validate_ohlcv
from autobacktest.strategies import Strategy


@dataclass(frozen=True)
class BacktestResult:
    """Results from one long/flat spot simulation."""

    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    initial_cash: float
    final_cash: float
    final_equity: float
    total_return_pct: float


class BacktestEngine:
    """Execute target-position signals at the following candle's open.

    The MVP supports spot long/flat positions, no leverage, and all available
    cash allocation. Buy and sell fills include adverse slippage and fees.
    Open positions are marked to the final close, not forcibly liquidated.
    """

    def __init__(
        self,
        initial_cash: float = 10_000.0,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
    ) -> None:
        if not np.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        for name, value in (("fee_rate", fee_rate), ("slippage_rate", slippage_rate)):
            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        self.initial_cash = float(initial_cash)
        self.fee_rate = float(fee_rate)
        self.slippage_rate = float(slippage_rate)

    def run(self, data: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        """Run a strategy over validated candles and return accounting records."""
        frame = validate_ohlcv(data)
        signals = strategy.generate_signals(frame)
        if not isinstance(signals, pd.Series):
            raise TypeError("strategy must return a pandas Series")
        if not signals.index.equals(frame.index):
            raise ValueError("strategy signals must preserve the input index")
        try:
            values = pd.to_numeric(signals, errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("strategy signals must be numeric target positions") from exc
        if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
            raise ValueError("strategy signals must contain only finite 0/1 target positions")

        cash = self.initial_cash
        quantity = 0.0
        equity_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []

        for i, candle in frame.iterrows():
            # A close-derived signal from candle i-1 can only fill at candle i.
            if i > 0:
                target = int(values[i - 1])
                open_price = float(candle["open"])
                if target == 1 and quantity == 0.0:
                    fill_price = open_price * (1.0 + self.slippage_rate)
                    quantity = cash / (fill_price * (1.0 + self.fee_rate))
                    notional = quantity * fill_price
                    fee = notional * self.fee_rate
                    cash -= notional + fee
                    trade_rows.append(
                        {
                            "timestamp": candle["timestamp"],
                            "side": "buy",
                            "price": fill_price,
                            "quantity": quantity,
                            "notional": notional,
                            "fee": fee,
                        }
                    )
                elif target == 0 and quantity > 0.0:
                    fill_price = open_price * (1.0 - self.slippage_rate)
                    notional = quantity * fill_price
                    fee = notional * self.fee_rate
                    cash += notional - fee
                    trade_rows.append(
                        {
                            "timestamp": candle["timestamp"],
                            "side": "sell",
                            "price": fill_price,
                            "quantity": quantity,
                            "notional": notional,
                            "fee": fee,
                        }
                    )
                    quantity = 0.0

            close_price = float(candle["close"])
            equity_rows.append(
                {
                    "timestamp": candle["timestamp"],
                    "close": close_price,
                    "cash": cash,
                    "quantity": quantity,
                    "position": int(quantity > 0.0),
                    "equity": cash + quantity * close_price,
                }
            )

        equity_curve = pd.DataFrame(equity_rows)
        trades = pd.DataFrame(
            trade_rows,
            columns=["timestamp", "side", "price", "quantity", "notional", "fee"],
        )
        final_equity = float(equity_curve["equity"].iloc[-1])
        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            initial_cash=self.initial_cash,
            final_cash=float(cash),
            final_equity=final_equity,
            total_return_pct=(final_equity / self.initial_cash - 1.0) * 100.0,
        )
