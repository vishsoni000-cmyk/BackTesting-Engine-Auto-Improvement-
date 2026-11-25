# BackTesting Engine — Auto Improvement

A research-focused cryptocurrency backtesting engine built incrementally around data integrity, realistic execution assumptions, reproducible evaluation, and evidence-based error learning.

## Current implementation

- **OHLCV ingestion:** CSV loading and CCXT-compatible fetch adapter.
- **Candle validation:** UTC timestamps, sorted/unique candles, finite positive prices, non-negative volume, and consistent high/low ranges.
- **Strategy interface:** deterministic target-position signals; EMA crossover baseline (spot long/flat).
- **Execution simulator:** signal from candle close fills at the next candle open; adverse slippage and fees are applied to buys and sells.
- **Accounting:** cash, quantity, marked-to-market equity curve, and fill ledger.
- **Performance metrics:** total return, maximum drawdown, annualized volatility, and Sharpe ratio.
- **Walk-forward splits:** expanding chronological train/test folds with no shuffling.
- **Error memory:** stable fingerprints, repeat counts, persisted diagnostics, and confirmed resolution notes. Backtest errors can be recorded by passing an error-memory instance to the engine.

The core is intentionally deterministic. AI review, strategy optimization, and the dashboard are not enabled yet.

## Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## Quick start

Prepare a CSV with these required columns:

`timestamp, open, high, low, close, volume`

Then run a baseline backtest:

```python
from autobacktest.engine import BacktestEngine
from autobacktest.metrics import performance_metrics
from autobacktest.ohlcv import load_ohlcv_csv
from autobacktest.strategies import EMACrossover

data = load_ohlcv_csv("data/BTC_USDT_1h.csv")
strategy = EMACrossover(fast_period=12, slow_period=26)

result = BacktestEngine(
    initial_cash=10_000,
    fee_rate=0.001,
    slippage_rate=0.0005,
).run(data, strategy)

metrics = performance_metrics(
    result.equity_curve,
    periods_per_year=365 * 24,  # hourly candles
)
print(metrics)
print(result.trades)
```

To persist recurring failures:

```python
from autobacktest.error_memory import ErrorMemory

memory = ErrorMemory("data/error_memory.json")
engine = BacktestEngine(error_memory=memory)
```

## Execution assumptions and limitations

- Spot long/flat only; no shorting, leverage, funding, or partial fills.
- The engine uses all available cash for an entry.
- Signals calculated from candle close are executed no earlier than the next candle open.
- Buy slippage increases the fill price; sell slippage decreases it.
- Open positions are marked to the final close and are not forcibly liquidated.
- Walk-forward utility yields complete test windows only; strategy selection and fold orchestration are not automated yet.
- Metrics depend on the correct bar frequency supplied through `periods_per_year`.

## Error-learning design

Errors are normalized into stable fingerprints and stored with occurrence counts and context. A human-confirmed resolution and regression-test reference can be attached to a record. The system does not execute arbitrary repair code or rewrite source files. Future self-improvement should use controlled experiments, isolated branches, regression tests, and explicit acceptance criteria.

## Development sequence

1. [x] Project scaffold and persistent error memory
2. [x] OHLCV ingestion and validation
3. [x] Strategy interface and EMA crossover
4. [x] Spot execution and portfolio accounting
5. [x] Performance metrics
6. [x] Chronological walk-forward split utility
7. [x] Error-memory integration
8. [ ] Fold evaluation and out-of-sample reporting
9. [ ] AI critic with structured, evidence-bound output
10. [ ] Controlled experiment runner
11. [ ] Dashboard and end-to-end demo

## Validation

GitHub Actions runs Python compilation and pytest on pushes to `main` and pull requests. Every module should add focused unit tests and pass CI before further integration.

## Disclaimer

For research and education only. Backtest results do not guarantee future performance and are sensitive to data quality, execution assumptions, liquidity, and overfitting.
