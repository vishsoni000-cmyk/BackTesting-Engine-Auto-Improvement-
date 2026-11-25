"""Strategy interface and deterministic baseline strategies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Strategy(Protocol):
    """A strategy maps historical market data to target-position signals."""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return target positions indexed exactly like input data."""
        ...


class EMACrossover:
    """Long/flat EMA crossover strategy with no look-ahead.

    Signals are calculated using the close of each candle. A backtest engine
    must execute a signal no earlier than the following candle.
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26) -> None:
        for name, value in (("fast_period", fast_period), ("slow_period", slow_period)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return 1 when fast EMA exceeds slow EMA, otherwise 0."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if "close" not in data.columns:
            raise ValueError("data must contain a close column")
        if data.empty:
            raise ValueError("data must contain at least one row")

        try:
            close = pd.to_numeric(data["close"], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("close values must be numeric") from exc
        values = close.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("close values must be finite and strictly positive")

        fast = close.ewm(span=self.fast_period, adjust=False, min_periods=self.fast_period).mean()
        slow = close.ewm(span=self.slow_period, adjust=False, min_periods=self.slow_period).mean()
        signals = (fast > slow).fillna(False).astype("int8")
        signals.name = "target_position"
        return signals
