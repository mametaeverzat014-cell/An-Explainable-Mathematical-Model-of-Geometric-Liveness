"""Тесты метрик PAD и процедуры выбора порога."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import (  # noqa: E402
    accuracy,
    acer,
    apcer,
    bpcer,
    compute_all_metrics,
    confusion_counts,
    risk,
    select_threshold,
)

# 1 = live, 0 = attack


def test_perfect_classifier() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert apcer(y_true, y_pred) == 0.0
    assert bpcer(y_true, y_pred) == 0.0
    assert acer(y_true, y_pred) == 0.0
    assert accuracy(y_true, y_pred) == 1.0


def test_worst_classifier() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    assert apcer(y_true, y_pred) == 1.0
    assert bpcer(y_true, y_pred) == 1.0
    assert acer(y_true, y_pred) == 1.0
    assert accuracy(y_true, y_pred) == 0.0


def test_apcer_counts_only_attacks() -> None:
    # 4 атаки, 2 из них приняты за live -> APCER = 0.5; live классифицированы верно.
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 1, 1])
    assert apcer(y_true, y_pred) == pytest.approx(0.5)
    assert bpcer(y_true, y_pred) == pytest.approx(0.0)
    assert acer(y_true, y_pred) == pytest.approx(0.25)


def test_bpcer_counts_only_live() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    assert bpcer(y_true, y_pred) == pytest.approx(0.25)
    assert apcer(y_true, y_pred) == pytest.approx(0.0)


def test_metrics_are_nan_when_class_missing() -> None:
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 0])
    assert np.isnan(apcer(y_true, y_pred))       # атак нет
    assert bpcer(y_true, y_pred) == pytest.approx(1 / 3)


def test_confusion_counts_sum_to_total() -> None:
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 0, 0, 1])
    c = confusion_counts(y_true, y_pred)
    total = c["live_as_live"] + c["live_as_attack"] + c["attack_as_live"] + c["attack_as_attack"]
    assert total == len(y_true)
    assert c["n_live"] == 3 and c["n_attack"] == 2


def test_risk_weights_attacks_more() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 0])        # APCER = 0.5, BPCER = 0
    assert risk(y_true, y_pred, 1.0, 1.0) == pytest.approx(0.5)
    assert risk(y_true, y_pred, 10.0, 1.0) == pytest.approx(5.0)


def test_select_threshold_finds_separating_value() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1])        # live имеет высокий скор
    tau, obj = select_threshold(scores, y_true, rule="min_acer")
    assert obj == pytest.approx(0.0)
    y_pred = (scores >= tau).astype(int)
    assert acer(y_true, y_pred) == pytest.approx(0.0)


def test_select_threshold_respects_risk_rule() -> None:
    scores = np.array([0.1, 0.5, 0.55, 0.9])
    y_true = np.array([0, 0, 1, 1])
    tau, _ = select_threshold(scores, y_true, rule="min_risk", lam_attack=10.0, lam_live=1.0)
    # При высокой цене пропуска атаки порог не должен пропускать ни одной атаки.
    y_pred = (scores >= tau).astype(int)
    assert apcer(y_true, y_pred) == pytest.approx(0.0)


def test_compute_all_metrics_reports_auc() -> None:
    y_true = [1, 1, 0, 0]
    y_pred = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    metrics = compute_all_metrics(y_true, y_pred, scores)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["n"] == 4
