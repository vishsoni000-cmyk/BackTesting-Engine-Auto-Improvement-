"""Leakage-resistant chronological walk-forward data splits."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-window fold; end offsets are exclusive."""

    train: pd.DataFrame
    test: pd.DataFrame
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def walk_forward_splits(
    data: pd.DataFrame,
    initial_train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> Iterator[WalkForwardFold]:
    """Yield expanding chronological train/test folds without shuffling.

    Each training set contains only rows before its test set. Only complete
    test windows are yielded; a trailing partial window is intentionally omitted.
    Returned frames are copies so caller mutation cannot alter the source.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    for name, value in (
        ("initial_train_size", initial_train_size),
        ("test_size", test_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if step_size is None:
        step_size = test_size
    if not isinstance(step_size, int) or isinstance(step_size, bool) or step_size < 1:
        raise ValueError("step_size must be a positive integer")
    if initial_train_size + test_size > len(data):
        raise ValueError("data is too short for one complete walk-forward fold")

    test_start = initial_train_size
    while test_start + test_size <= len(data):
        test_end = test_start + test_size
        yield WalkForwardFold(
            train=data.iloc[:test_start].copy(),
            test=data.iloc[test_start:test_end].copy(),
            train_start=0,
            train_end=test_start,
            test_start=test_start,
            test_end=test_end,
        )
        test_start += step_size
