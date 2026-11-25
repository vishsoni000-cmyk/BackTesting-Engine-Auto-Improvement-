"""OHLCV ingestion and strict candle validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


class OHLCVValidationError(ValueError):
    """Raised when candle data is missing or violates market-data invariants."""


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of valid OHLCV candles.

    Timestamp values are converted to UTC. Rows are sorted chronologically.
    Duplicate timestamps, non-finite values, invalid prices, and impossible candle
    ranges are rejected rather than silently repaired.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise OHLCVValidationError(f"Missing required OHLCV columns: {', '.join(missing)}")
    if frame.empty:
        raise OHLCVValidationError("OHLCV data must contain at least one candle")

    result = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    try:
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise OHLCVValidationError("timestamp values must be valid datetimes") from exc

    for column in (*PRICE_COLUMNS, "volume"):
        try:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise OHLCVValidationError(f"{column} values must be numeric") from exc

    numeric = result.loc[:, [*PRICE_COLUMNS, "volume"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise OHLCVValidationError("OHLCV numeric values must all be finite")
    if (result.loc[:, list(PRICE_COLUMNS)] <= 0).any().any():
        raise OHLCVValidationError("OHLC prices must be strictly positive")
    if (result["volume"] < 0).any():
        raise OHLCVValidationError("volume must be non-negative")
    if (result["high"] < result["low"]).any():
        raise OHLCVValidationError("high must be greater than or equal to low")
    if (result["high"] < result.loc[:, ["open", "close"]].max(axis=1)).any():
        raise OHLCVValidationError("high must be at least max(open, close)")
    if (result["low"] > result.loc[:, ["open", "close"]].min(axis=1)).any():
        raise OHLCVValidationError("low must be at most min(open, close)")

    result = result.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if result["timestamp"].duplicated().any():
        raise OHLCVValidationError("duplicate candle timestamps are not allowed")
    return result


def fetch_ohlcv(
    exchange: Any,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since: int | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch one OHLCV page through a CCXT-compatible exchange object."""
    if not symbol.strip():
        raise ValueError("symbol must not be empty")
    if not timeframe.strip():
        raise ValueError("timeframe must not be empty")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    if not rows:
        raise OHLCVValidationError("Exchange returned no OHLCV candles")
    if any(not isinstance(row, (list, tuple)) or len(row) < 6 for row in rows):
        raise OHLCVValidationError("Exchange returned malformed OHLCV rows")
    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    return validate_ohlcv(frame)


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate OHLCV candles from a CSV file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"OHLCV CSV not found: {source}")
    try:
        frame = pd.read_csv(source)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise OHLCVValidationError(f"Could not read OHLCV CSV: {source}") from exc
    return validate_ohlcv(frame)
