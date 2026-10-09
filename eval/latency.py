import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from luem.dataset import read_samples
from luem.models import MODELS_DIR, NAMES, load_model

ROOT = Path(__file__).resolve().parent.parent
BUDGET_MS = 5.0


def cpu_name() -> str:
    if sys.platform == "linux":
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def measure(model, texts: list[str], warmup: int = 200) -> dict:
    calls = [t[:k] for t in texts for k in range(1, len(t) + 1)]
    for c in calls[:warmup]:
        model.predict(c)
    times = []
    for c in calls:
        start = time.perf_counter()
        model.predict(c)
        times.append((time.perf_counter() - start) * 1e3)
    times.sort()
    return {"calls": len(times), "mean_ms": round(statistics.fmean(times), 4),
            "p50_ms": round(times[len(times) // 2], 4), "p99_ms": round(times[int(len(times) * 0.99)], 4),
            "max_ms": round(times[-1], 4), "under_budget": times[int(len(times) * 0.99)] < BUDGET_MS}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", choices=NAMES, help="repeatable; default: all")
    p.add_argument("--chunks", type=int, default=1000, help="test chunks to type")
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / f"latency_{sys.platform}.json")
    args = p.parse_args()

    texts = [r["text"] for r in read_samples(ROOT / "data" / "processed" / "test.jsonl", args.chunks, seed=3)]
    out = {"machine": {"os": platform.platform(), "cpu": cpu_name(), "python": platform.python_version()},
           "budget_ms": BUDGET_MS, "models": {}}
    print(f"{out['machine']['cpu']} | {out['machine']['os']}")
    print(f"{'model':11}{'load s':>8}{'mean ms':>9}{'p50 ms':>8}{'p99 ms':>8}{'max ms':>8}")
    for name in args.model or NAMES:
        t0 = time.perf_counter()
        model = load_model(name, args.models_dir)
        r = {"load_s": round(time.perf_counter() - t0, 3)} | measure(model, texts)
        out["models"][name] = r
        print(f"{name:11}{r['load_s']:8.2f}{r['mean_ms']:9.3f}{r['p50_ms']:8.3f}{r['p99_ms']:8.3f}{r['max_ms']:8.2f}")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
