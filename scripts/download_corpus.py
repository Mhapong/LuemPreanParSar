"""Download raw text corpora to data/raw/<source>.jsonl ({"id", "text"} per line).

Usage:
    uv run python scripts/download_corpus.py                  # all sources, default limits
    uv run python scripts/download_corpus.py --only wiki_th --limit 1000
    uv run python scripts/download_corpus.py --limit 0        # no limit (wiki_en is ~20 GB!)

Streams from the Hugging Face Hub, so only the rows we keep are downloaded.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# name -> (hf repo, config, text column, default article limit)
SOURCES = {
    "wisesight": ("pythainlp/wisesight_sentiment", "wisesight_sentiment", "texts", 0),
    # ~10 KB/article (th), ~3 KB/article (en): plenty of segments for 1-2M samples
    "wiki_th": ("wikimedia/wikipedia", "20231101.th", "text", 20_000),
    "wiki_en": ("wikimedia/wikipedia", "20231101.en", "text", 20_000),
}
SEED = 42


def download(name: str, limit: int | None) -> None:
    repo, config, column, default_limit = SOURCES[name]
    limit = default_limit if limit is None else limit
    out = RAW_DIR / f"{name}.jsonl"

    # Wisesight is tiny: take every split. Wikipedia has only "train".
    splits = ["train", "validation", "test"] if name == "wisesight" else ["train"]
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for split in splits:
            ds = load_dataset(repo, config, split=split, streaming=True)
            if name.startswith("wiki"):
                # Shuffle so a limit doesn't keep only alphabetically-first articles
                ds = ds.shuffle(seed=SEED, buffer_size=10_000)
            for i, row in enumerate(ds):
                text = (row[column] or "").strip()
                if not text:
                    continue
                doc_id = row.get("id") or f"{split}-{i}"
                f.write(json.dumps({"id": f"{name}:{doc_id}", "text": text}, ensure_ascii=False) + "\n")
                n += 1
                if n % 10_000 == 0:
                    print(f"  {name}: {n:,}")
                if limit and n >= limit:
                    break
            if limit and n >= limit:
                break
    print(f"{name}: wrote {n:,} docs, {out.stat().st_size / 1e6:.1f} MB -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=SOURCES, action="append", help="source to download (repeatable)")
    p.add_argument("--limit", type=int, default=None, help="max docs per source; 0 = no limit")
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.only or SOURCES:
        download(name, args.limit)


if __name__ == "__main__":
    main()
