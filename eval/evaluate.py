"""Evaluate models with the two-moment rule: choose thresholds on val, report on test.

Usage:
    uv run python eval/evaluate.py                                # all models, targets 0.99 and 0.999
    uv run python eval/evaluate.py --model cnn --target 0.995
    uv run python eval/evaluate.py --model ngram --limit 0        # all of val/test (slow)

Each model scores val and test once; thresholds are then chosen on val separately for every target.
Writes eval/results/<model>.json, eval/results/scores/<model>.npz (test full-text scores, for the
plots) and prints a comparison table. Latency is measured by eval/latency.py on the demo machine.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from luem.dataset import read_samples
from luem.metrics import average_precision, evaluate, first_fire, pr_curve, select_thresholds
from luem.models import MODELS_DIR, NAMES, load_model, model_path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "eval" / "results"
MAX_ERRORS = 500


def score_all(model, rows) -> list[list[float]]:
    return [model.predict_prefixes(r["text"]) for r in rows]


def errors(rows, scores, th) -> dict:
    """Every false fix and miss on test with what the model saw, for error analysis."""
    out = {"false_fix": [], "missed": []}
    for r, s in zip(rows, scores):
        k = first_fire(s, th.k_min, th.tau_type)
        fired = k is not None or s[-1] >= th.tau_space
        if fired == (r["label"] == "wrong"):
            continue
        out["false_fix" if fired else "missed"].append({
            "text": r["text"], "intended": r["intended"], "active": r["active"], "kind": r["kind"],
            "source": r["source"], "full": round(s[-1], 6), "max_prefix": round(max(s), 6),
            "stage": None if not fired else ("typing" if k is not None else "space"), "at_char": k,
        })
    return {key: v[:MAX_ERRORS] for key, v in out.items()} | {f"n_{key}": len(v) for key, v in out.items()}


def run(name: str, val, test, targets: list[float], models_dir: Path = MODELS_DIR,
        scores_dir: Path | None = None) -> dict:
    t0 = time.time()
    model = load_model(name, models_dir)
    print(f"[{name}] loaded ({time.time() - t0:.0f}s)")
    val_scores = score_all(model, val)
    test_scores = score_all(model, test)
    print(f"[{name}] scored val and test ({time.time() - t0:.0f}s)")

    y_test = np.array([r["label"] == "wrong" for r in test])
    full_test = np.array([s[-1] for s in test_scores])
    result = {"model": name, "file_bytes": model_path(name, models_dir).stat().st_size,
              "n_val": len(val), "n_test": len(test),
              "test_average_precision_full_text": round(average_precision(y_test, full_test), 5),
              "test_pr_curve_full_text": pr_curve(y_test, full_test), "targets": {}}
    y_val = np.array([r["label"] == "wrong" for r in val])
    for target in targets:
        th = select_thresholds(y_val, val_scores, target)
        print(f"[{name}] target {target}: k_min={th.k_min} tau_type={th.tau_type:.6g} tau_space={th.tau_space:.6g}")
        drop = ("pr_curve_full_text", "average_precision_full_text")
        result["targets"][str(target)] = {
            "test": {k: v for k, v in evaluate(test, test_scores, th).items() if k not in drop},
            "val": {k: v for k, v in evaluate(val, val_scores, th).items() if k not in drop},
            "errors": errors(test, test_scores, th),
        }
    if scores_dir is not None:
        scores_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(scores_dir / f"{name}.npz", y=y_test, full=full_test.astype(np.float32),
                            hard=np.array([r["kind"] == "hard_negative" for r in test]))
    print(f"[{name}] done ({time.time() - t0:.0f}s)")
    return result


def print_table(results: list[dict]) -> None:
    head = f"{'model':11}{'target':>7}{'prec':>7}{'recall':>8}{'typing':>8}{'space':>7}{'falsefix':>10}" \
           f"{'hardneg':>9}{'prec@2%':>9}{'chars':>7}{'AP':>8}{'KB':>8}"
    print("\nTest results (thresholds chosen on val)\n" + head + "\n" + "-" * len(head))
    for r in results:
        for target, res in r["targets"].items():
            t = res["test"]
            print(f"{r['model']:11}{target:>7}{t['precision']:7.4f}{t['recall']:8.4f}{t['recall_while_typing']:8.3f}"
                  f"{t['recall_on_space']:7.3f}{t['false_fix_rate']:10.4f}{t['false_fix_rate_hard_negatives']:9.4f}"
                  f"{t['precision_at_base_rate']['0.02']:9.3f}{t['chars_to_detect']['median'] or 0:7.1f}"
                  f"{r['test_average_precision_full_text']:8.4f}{r['file_bytes'] / 1024:8.0f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", choices=NAMES, help="repeatable; default: all")
    p.add_argument("--limit", type=int, default=30_000, help="random subset of val/test (0 = all)")
    p.add_argument("--target", type=float, action="append",
                   help="minimum precision when choosing thresholds; repeatable; default: 0.99 and 0.999")
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="e.g. models/runs/b to compare a variant")
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    args = p.parse_args()

    val = read_samples(DATA / "val.jsonl", args.limit, seed=1)
    test = read_samples(DATA / "test.jsonl", args.limit, seed=2)
    print(f"val={len(val):,} test={len(test):,} samples")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in args.model or NAMES:
        r = run(name, val, test, args.target or [0.99, 0.999], args.models_dir, args.results_dir / "scores")
        (args.results_dir / f"{name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(r)
    print_table(results)


if __name__ == "__main__":
    main()
