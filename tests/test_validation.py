"""Tests for chronological walk-forward splits."""

import pandas as pd
import pytest

from autobacktest.validation import walk_forward_splits


def test_expanding_walk_forward_folds_are_chronological_and_disjoint():
    data = pd.DataFrame({"value": range(10)})
    folds = list(walk_forward_splits(data, initial_train_size=4, test_size=2))

    assert len(folds) == 3
    assert folds[0].train["value"].tolist() == [0, 1, 2, 3]
    assert folds[0].test["value"].tolist() == [4, 5]
    assert folds[1].train["value"].tolist() == [0, 1, 2, 3, 4, 5]
    assert folds[1].test["value"].tolist() == [6, 7]
    assert folds[2].test["value"].tolist() == [8, 9]
    for fold in folds:
        assert fold.train_end == fold.test_start
        assert fold.test_end - fold.test_start == 2


def test_step_size_can_skip_test_windows_without_leakage():
    data = pd.DataFrame({"value": range(11)})
    folds = list(walk_forward_splits(data, initial_train_size=4, test_size=2, step_size=3))

    assert [fold.test["value"].tolist() for fold in folds] == [[4, 5], [7, 8]]


@pytest.mark.parametrize(
    "train_size,test_size,step",
    [(0, 2, None), (3, 0, None), (3, 2, 0), (True, 2, None), (3, 2, 1.5)],
)
def test_invalid_split_sizes_are_rejected(train_size, test_size, step):
    with pytest.raises(ValueError):
        list(walk_forward_splits(pd.DataFrame({"x": range(10)}), train_size, test_size, step))


def test_data_too_short_for_complete_fold_is_rejected():
    with pytest.raises(ValueError, match="too short"):
        list(walk_forward_splits(pd.DataFrame({"x": range(5)}), 4, 2))
