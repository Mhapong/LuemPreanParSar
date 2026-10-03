"""Evaluate models with the two-moment rule: choose thresholds on val, report on test.

Usage:
    uv run python eval/evaluate.py --model ngram --model dictionary
    uv run python eval/evaluate.py --model ngram --limit 0     # all of val/test (slow)

Writes eval/results/<model>.json and prints a comparison table.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from luem.dataset import read_samples
from luem.metrics import evaluate, first_fire, select_thresholds
from luem.models import NAMES, load_model

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "eval" / "results"


def score_all(model, rows) -> list[list[float]]:
    return [model.predict_prefixes(r["text"]) for r in rows]


def latency_us(model, rows, n: int = 2000) -> dict:
    """Time of one predict() call on a full chunk (what the daemon pays per keystroke), 1 thread."""
    texts = [r["text"] for r in rows[:n]]
    times = []
    for t in texts:
        start = time.perf_counter()
        model.predict(t)
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    return {"mean": round(statistics.fmean(times), 1), "p50": round(times[len(times) // 2], 1),
            "p99": round(times[int(len(times) * 0.99)], 1)}


def run(name: str, val, test, target: float) -> dict:
    t0 = time.time()
    model = load_model(name)
    print(f"[{name}] loaded ({time.time() - t0:.0f}s)")
    val_scores = score_all(model, val)
    print(f"[{name}] scored val ({time.time() - t0:.0f}s)")
    th = select_thresholds(np.array([r["label"] == "wrong" for r in val]), val_scores, target)
    print(f"[{name}] thresholds from val: k_min={th.k_min} tau_type={th.tau_type:.6g} tau_space={th.tau_space:.6g}")
    test_scores = score_all(model, test)
    result = {"model": name, "target_precision": target, "test": evaluate(test, test_scores, th),
              "val": {k: v for k, v in evaluate(val, val_scores, th).items() if k != "pr_curve_full_text"},
              "latency_us": latency_us(model, test)}
    # a few mistakes for error analysis
    def fired(s):
        return first_fire(s, th.k_min, th.tau_type) is not None or s[-1] >= th.tau_space

    result["errors"] = {
        "false_fix": [r["text"] for r, s in zip(test, test_scores) if r["label"] == "ok" and fired(s)][:40],
        "missed": [f'{r["text"]} <- {r["intended"]}' for r, s in zip(test, test_scores)
                   if r["label"] == "wrong" and not fired(s)][:40],
    }
    print(f"[{name}] done ({time.time() - t0:.0f}s)")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", choices=NAMES, help="repeatable; default: all")
    p.add_argument("--limit", type=int, default=30_000, help="random subset of val/test (0 = all)")
    p.add_argument("--target", type=float, default=0.99, help="minimum precision when choosing thresholds")
    args = p.parse_args()

    val = read_samples(DATA / "val.jsonl", args.limit, seed=1)
    test = read_samples(DATA / "test.jsonl", args.limit, seed=2)
    print(f"val={len(val):,} test={len(test):,} samples")
    RESULTS.mkdir(parents=True, exist_ok=True)

    results = []
    for name in args.model or NAMES:
        r = run(name, val, test, args.target)
        (RESULTS / f"{name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(r)

    print("\nTest results (thresholds chosen on val, target precision %.2f)" % args.target)
    head = f"{'model':11}{'prec':>7}{'recall':>8}{'typing':>8}{'space':>7}{'falsefix':>10}{'hardneg':>9}" \
           f"{'prec@2%':>9}{'chars':>7}{'AP':>7}{'us':>8}"
    print(head + "\n" + "-" * len(head))
    for r in results:
        t = r["test"]
        print(f"{r['model']:11}{t['precision']:7.3f}{t['recall']:8.3f}{t['recall_while_typing']:8.3f}"
              f"{t['recall_on_space']:7.3f}{t['false_fix_rate']:10.4f}{t['false_fix_rate_hard_negatives']:9.4f}"
              f"{t['precision_at_base_rate']['0.02']:9.3f}{t['chars_to_detect']['median'] or 0:7.1f}"
              f"{t['average_precision_full_text']:7.3f}{r['latency_us']['p50']:8.0f}")


if __name__ == "__main__":
    main()
