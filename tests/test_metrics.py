import numpy as np

from luem.metrics import (
    DISABLED, Thresholds, average_precision, evaluate, first_fire, precision_at_base_rate, select_thresholds,
)


def row(label, kind="normal", active="en"):
    return {"label": label, "kind": kind, "active": active}


def test_first_fire_respects_k_min():
    s = [0.999, 0.2, 0.999, 0.999]
    assert first_fire(s, 1, 0.99) == 1
    assert first_fire(s, 2, 0.99) == 3
    assert first_fire(s, 1, 1.5) is None


def test_select_thresholds_on_separable_data():
    y = np.array([1, 1, 0, 0])
    scores = [[0.6, 0.99], [0.995, 0.999], [0.1, 0.2], [0.3, 0.1]]
    th = select_thresholds(y, scores, target=0.99)
    rows = [row("wrong"), row("wrong"), row("ok"), row("ok")]
    r = evaluate(rows, scores, th)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["false_fix_rate"] == 0.0


def test_stage_b_catches_what_stage_a_misses():
    th = Thresholds(k_min=1, tau_type=0.99, tau_space=0.9)
    rows = [row("wrong"), row("wrong"), row("ok", kind="hard_negative")]
    scores = [[0.995], [0.5, 0.95], [0.1, 0.2]]
    r = evaluate(rows, scores, th)
    assert r["recall_while_typing"] == 0.5 and r["recall_on_space"] == 0.5
    assert r["false_fix_rate_hard_negatives"] == 0.0


def test_disabled_stage_never_fires():
    th = Thresholds(k_min=1, tau_type=DISABLED, tau_space=DISABLED)
    r = evaluate([row("wrong")], [[1.0]], th)
    assert r["recall"] == 0.0


def test_average_precision_perfect_ranking():
    assert average_precision(np.array([1, 1, 0]), np.array([0.9, 0.8, 0.1])) == 1.0


def test_precision_drops_at_low_base_rate():
    # 99% recall and 1% false fixes look great on 50/50 data, but only ~67% precision if 2% of words are wrong
    assert round(precision_at_base_rate(0.99, 0.01, 0.5), 2) == 0.99
    assert round(precision_at_base_rate(0.99, 0.01, 0.02), 2) == 0.67
