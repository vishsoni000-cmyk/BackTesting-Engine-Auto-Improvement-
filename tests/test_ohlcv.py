import pandas as pd
import pytest

from autobacktest.ohlcv import OHLCVValidationError, fetch_ohlcv, validate_ohlcv


def candles():
    return pd.DataFrame(
        {
            "timestamp": ["2024-01-01T01:00:00Z", "2024-01-01T00:00:00Z"],
            "open": [11, 10],
            "high": [12, 11],
            "low": [10, 9],
            "close": [11.5, 10.5],
            "volume": [2, 3],
        }
    )


def test_validation_sorts_and_converts_timestamps_to_utc():
    result = validate_ohlcv(candles())
    assert result["timestamp"].is_monotonic_increasing
    assert str(result["timestamp"].dt.tz) == "UTC"


@pytest.mark.parametrize(
    "column,value",
    [
        ("close", -1),
        ("volume", -1),
        ("high", float("nan")),
        ("open", float("inf")),
    ],
)
def test_invalid_numeric_values_are_rejected(column, value):
    frame = candles()
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    with pytest.raises(OHLCVValidationError):
        validate_ohlcv(frame)


def test_duplicate_timestamps_are_rejected():
    frame = candles()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    with pytest.raises(OHLCVValidationError, match="duplicate"):
        validate_ohlcv(frame)


def test_impossible_high_low_range_is_rejected():
    frame = candles()
    frame["high"] = frame["high"].astype(float)
    frame.loc[0, "high"] = 10.5
    with pytest.raises(OHLCVValidationError):
        validate_ohlcv(frame)


def test_fetch_ohlcv_uses_exchange_adapter_and_validates():
    class FakeExchange:
        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            assert symbol == "BTC/USDT"
            assert timeframe == "1h"
            assert limit == 1
            return [[1704067200000, 10, 12, 9, 11, 5]]

    result = fetch_ohlcv(FakeExchange(), limit=1)
    assert len(result) == 1
    assert result.loc[0, "close"] == 11


def test_empty_exchange_response_rejected():
    class EmptyExchange:
        def fetch_ohlcv(self, *args, **kwargs):
            return []

    with pytest.raises(OHLCVValidationError, match="no OHLCV"):
        fetch_ohlcv(EmptyExchange())
