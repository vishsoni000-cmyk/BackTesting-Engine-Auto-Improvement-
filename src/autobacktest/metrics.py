"""Performance metrics computed from a marked-to-market equity curve."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(
    equity_curve: pd.DataFrame,
    periods_per_year: int = 365,
    risk_free_rate: float = 0.0,
) -> dict[str, float | None]:
    """Calculate return, drawdown, volatility and Sharpe from equity values.

    risk_free_rate is an annual decimal rate (for example, 0.03 for 3%).
    Drawdown is reported as a negative percentage from the prior equity peak.
    Sharpe and annualized volatility are None when the return sample is too
    small or has zero variance.
    """
    if not isinstance(equity_curve, pd.DataFrame):
        raise TypeError("equity_curve must be a pandas DataFrame")
    if "equity" not in equity_curve.columns:
        raise ValueError("equity_curve must contain an equity column")
    if len(equity_curve) < 2:
        raise ValueError("equity_curve must contain at least two observations")
    if (
        not isinstance(periods_per_year, int)
        or isinstance(periods_per_year, bool)
        or periods_per_year < 1
    ):
        raise ValueError("periods_per_year must be a positive integer")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")

    try:
        equity = pd.to_numeric(equity_curve["equity"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("equity values must be numeric") from exc
    values = equity.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("equity values must be finite and strictly positive")

    returns = equity.pct_change().dropna()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    total_return_pct = (values[-1] / values[0] - 1.0) * 100.0

    if len(returns) >= 2:
        volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
        annualized_volatility_pct: float | None = volatility * 100.0
        excess = returns - risk_free_rate / periods_per_year
        deviation = float(excess.std(ddof=1))
        sharpe_ratio: float | None = (
            float(excess.mean() / deviation * np.sqrt(periods_per_year))
            if deviation > 0.0
            else None
        )
    else:
        annualized_volatility_pct = None
        sharpe_ratio = None

    return {
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "annualized_volatility_pct": annualized_volatility_pct,
        "sharpe_ratio": sharpe_ratio,
    }
