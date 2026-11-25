# BackTesting Engine — Auto Improvement

A research-focused cryptocurrency strategy backtesting engine with an evidence-based AI critic and a controlled error-learning subsystem.

## Status

The project is being developed incrementally. Milestone 1 establishes Python package metadata and persistent error memory.

## Principles

- Deterministic backtests with explicit execution assumptions.
- Historical-data integrity and no-look-ahead validation.
- Transaction costs included in simulated results.
- Python computes metrics; the AI critic reviews verified evidence.
- Errors are fingerprinted and stored; confirmed resolutions can be retrieved.
- Automatic recovery is limited to safe, explicitly registered operations.

## Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## Planned milestones

1. Project scaffold and error memory.
2. OHLCV ingestion and validation.
3. Strategy interface and EMA/RSI strategies.
4. Execution simulator and portfolio accounting.
5. Performance metrics and benchmarks.
6. Out-of-sample and walk-forward validation.
7. AI critic with structured output.
8. Controlled experiment runner.
9. Streamlit dashboard.
10. Integration tests, documentation, and demo.

## Error-learning design

The error memory stores a normalized fingerprint, occurrence count, diagnostic message, and confirmed resolution. It can identify repeated failures and recommend a previously recorded resolution. It does not claim to autonomously repair arbitrary defects; code changes require tests and review.

## Disclaimer

For research and education only. Backtest results do not guarantee future performance and are sensitive to data quality, execution assumptions, and overfitting.
