from dataclasses import asdict, dataclass

import numpy as np

TAUS = np.unique(
    np.concatenate([np.linspace(0.5, 0.99, 50), 1 - np.logspace(-2, -9, 36)])
)
K_RANGE = range(1, 7)
DISABLED = 2.0


@dataclass
class Thresholds:
    k_min: int
    tau_type: float
    tau_space: float


def suffix_max(prefix_scores: list[list[float]], k_min: int) -> np.ndarray:
    return np.array(
        [max(s[k_min - 1 :]) if len(s) >= k_min else 0.0 for s in prefix_scores]
    )


def _prec_rec(fire: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    tp = int((fire & y).sum())
    n_fire = int(fire.sum())
    return (tp / n_fire if n_fire else 1.0), tp / max(1, int(y.sum()))


def select_thresholds(
    y: np.ndarray, prefix_scores: list[list[float]], target: float = 0.99
) -> Thresholds:
    y = y.astype(bool)
    best_a = (-1.0, Thresholds(K_RANGE[-1], DISABLED, DISABLED))
    for k in K_RANGE:
        m = suffix_max(prefix_scores, k)
        for tau in TAUS:
            prec, rec = _prec_rec(m >= tau, y)
            if prec >= target and rec > best_a[0]:
                best_a = (rec, Thresholds(k, float(tau), DISABLED))
    th = best_a[1]
    fire_a = suffix_max(prefix_scores, th.k_min) >= th.tau_type
    full = np.array([s[-1] for s in prefix_scores])
    best_rec = _prec_rec(fire_a, y)[1]
    for tau in TAUS:
        prec, rec = _prec_rec(fire_a | (full >= tau), y)
        if prec >= target and rec > best_rec:
            best_rec, th.tau_space = rec, float(tau)
    return th


def first_fire(scores: list[float], k_min: int, tau: float) -> int | None:
    for k in range(k_min, len(scores) + 1):
        if scores[k - 1] >= tau:
            return k
    return None


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    y = y[order].astype(bool)
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / max(1, y.sum()))


def pr_curve(y: np.ndarray, score: np.ndarray, points: int = 60) -> list[dict]:
    out = []
    for tau in np.quantile(score, np.linspace(0, 1, points)):
        prec, rec = _prec_rec(score >= tau, y.astype(bool))
        out.append(
            {
                "tau": round(float(tau), 6),
                "precision": round(prec, 5),
                "recall": round(rec, 5),
            }
        )
    return out


def precision_at_base_rate(recall: float, fpr: float, base_rate: float) -> float:
    hit, false = recall * base_rate, fpr * (1 - base_rate)
    return hit / (hit + false) if hit + false else 1.0


def evaluate(
    rows: list[dict], prefix_scores: list[list[float]], th: Thresholds
) -> dict:
    y = np.array([r["label"] == "wrong" for r in rows])
    hard = np.array([r["kind"] == "hard_negative" for r in rows])
    th_intended = np.array(
        [r["active"] == "en" and r["label"] == "wrong" for r in rows]
    )  # meant Thai
    full = np.array([s[-1] for s in prefix_scores])

    fire_k = [first_fire(s, th.k_min, th.tau_type) for s in prefix_scores]
    fire_a = np.array([k is not None for k in fire_k])
    fire_b = ~fire_a & (full >= th.tau_space)
    fire = fire_a | fire_b

    prec, rec = _prec_rec(fire, y)
    ok = ~y
    fpr = float((fire & ok).sum() / max(1, ok.sum()))
    chars = np.array([k for k, wrong in zip(fire_k, y) if wrong and k is not None])
    n_wrong = int(y.sum())
    cdf = [round(float((chars <= k).sum() / max(1, n_wrong)), 5) for k in range(1, 33)]

    return {
        "thresholds": asdict(th),
        "n": len(rows),
        "n_wrong": n_wrong,
        "confusion": {
            "tp": int((fire & y).sum()),
            "fp": int((fire & ok).sum()),
            "fn": int((~fire & y).sum()),
            "tn": int((~fire & ok).sum()),
        },
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "f1": round(2 * prec * rec / (prec + rec), 5) if prec + rec else 0.0,
        "recall_while_typing": round(float((fire_a & y).sum() / max(1, n_wrong)), 5),
        "recall_on_space": round(float((fire_b & y).sum() / max(1, n_wrong)), 5),
        "false_fix_rate": round(fpr, 6),
        "false_fix_rate_hard_negatives": round(
            float((fire & hard).sum() / max(1, hard.sum())), 6
        ),
        "recall_meant_thai": round(
            float((fire & th_intended).sum() / max(1, th_intended.sum())), 5
        ),
        "recall_meant_english": round(
            float((fire & y & ~th_intended).sum() / max(1, (y & ~th_intended).sum())), 5
        ),
        "precision_at_base_rate": {
            str(b): round(precision_at_base_rate(rec, fpr, b), 5)
            for b in (0.02, 0.05, 0.1)
        },
        "chars_to_detect": {
            "median": float(np.median(chars)) if len(chars) else None,
            "mean": round(float(chars.mean()), 3) if len(chars) else None,
            "caught_by_3_chars": cdf[2],
            "caught_by_5_chars": cdf[4],
            "cdf": cdf,
        },
        "average_precision_full_text": round(average_precision(y, full), 5),
        "pr_curve_full_text": pr_curve(y, full),
    }
