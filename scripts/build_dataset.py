import argparse
import hashlib
import json
import random
import string
import time
from collections import Counter, defaultdict
from pathlib import Path

from luem.dataset import SPLITS, chunks, hard_negatives, make_sample, split_of

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
CAPS = {"wiki_th": 750_000, "wisesight": 300_000, "wiki_en": 550_000}
SPLIT_FRACTION = {"train": 0.8, "val": 0.1, "test": 0.1}


def reservoir_chunks(path: Path, cap: int, rng: random.Random):
    caps = {s: max(1, int(cap * SPLIT_FRACTION[s])) for s in SPLITS}
    kept = {s: [] for s in SPLITS}
    seen = Counter()
    docs, dup_docs = Counter(), 0
    doc_hashes = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            digest = hashlib.md5(row["text"].encode("utf-8")).digest()
            if digest in doc_hashes:
                dup_docs += 1
                continue
            doc_hashes.add(digest)
            split = split_of(row["id"])
            docs[split] += 1
            res, k = kept[split], caps[split]
            for chunk, script in chunks(row["text"]):
                seen[split] += 1
                if len(res) < k:
                    res.append((chunk, script, row["id"]))
                else:
                    j = rng.randrange(seen[split])
                    if j < k:
                        res[j] = (chunk, script, row["id"])
    return kept, seen, docs, dup_docs


def english_words(samples: list) -> list[str]:
    words = {
        c.lower()
        for c, script, _ in samples
        if script == "en" and c.isalpha() and 3 <= len(c) <= 10
    }
    return sorted(words)


def describe(samples) -> dict:
    by = lambda key: dict(Counter(key(s) for s in samples).most_common())
    lengths = [len(s.text) for s in samples]
    hist = Counter(min(n // 4 * 4, 32) for n in lengths)
    return {
        "total": len(samples),
        "label": by(lambda s: s.label),
        "active_label": by(lambda s: f"{s.active}/{s.label}"),
        "kind": by(lambda s: s.kind),
        "source": by(lambda s: s.source),
        "mean_len": round(sum(lengths) / max(1, len(lengths)), 2),
        "len_hist": {f"{b}-{b + 3}": hist[b] for b in sorted(hist)},
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply per-source caps (0.01 = quick test)",
    )
    p.add_argument(
        "--hard-frac",
        type=float,
        default=0.05,
        help="hard negatives as a fraction of normal samples",
    )
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    rng = random.Random(args.seed)
    t0 = time.time()
    per_split = defaultdict(list)
    corpus_stats = {}
    en_pool = []

    for source, cap in CAPS.items():
        path = args.raw_dir / f"{source}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}: run scripts/download_corpus.py first")
        kept, seen, docs, dup_docs = reservoir_chunks(path, int(cap * args.scale), rng)
        corpus_stats[source] = {
            "docs": dict(docs),
            "duplicate_docs": dup_docs,
            "usable_chunks": dict(seen),
            "kept_chunks": {s: len(kept[s]) for s in SPLITS},
        }
        print(
            f"{source:10} docs={sum(docs.values()):>6,} dup={dup_docs:>5,} "
            f"usable={sum(seen.values()):>10,} kept={sum(len(v) for v in kept.values()):>9,}  ({time.time() - t0:.0f}s)"
        )
        en_pool += kept["train"]
        for split in SPLITS:
            for chunk, script, doc in kept[split]:
                s = make_sample(chunk, script, rng.random() < 0.5, source, doc)
                if s:
                    per_split[split].append(s)

    words = english_words(en_pool)
    del en_pool
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "seed": args.seed,
        "scale": args.scale,
        "hard_frac": args.hard_frac,
        "corpus": corpus_stats,
        "splits": {},
    }

    for split in SPLITS:
        samples = per_split[split]
        samples += hard_negatives(int(len(samples) * args.hard_frac), rng, words, split)
        rng.shuffle(samples)
        out = args.out_dir / f"{split}.jsonl"
        with out.open("w", encoding="utf-8", newline="\n") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        stats["splits"][split] = describe(samples)
        print(
            f"{split:5} {len(samples):>9,} samples -> {out} ({out.stat().st_size / 1e6:.1f} MB)"
        )

    train = per_split["train"]
    ok_texts = {s.text for s in train if s.label == "ok"}
    wrong = [s for s in train if s.label == "wrong"]
    collide = [s for s in wrong if s.text in ok_texts]
    stats["train_wrong_also_ok_text"] = {
        "rate": round(len(collide) / max(1, len(wrong)), 4),
        "examples": sorted(
            {f"{s.text} <- {s.intended}" for s in collide[:2000]}, key=len
        )[:20],
    }
    stats["examples"] = {
        k: [s.to_dict() for s in train if f"{s.kind}/{s.label}" == k][:5]
        for k in ("normal/ok", "normal/wrong", "hard_negative/ok", "tech_wrong/wrong")
    }
    (args.out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"wrong-layout strings that are also real chunks: {stats['train_wrong_also_ok_text']['rate']:.2%}"
    )
    print(f"done in {time.time() - t0:.0f}s, stats -> {args.out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
