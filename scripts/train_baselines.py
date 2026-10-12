import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from luem.dataset import script_of_chunk
from luem.models import MODELS_DIR
from luem.models.ngram import CharNgramLM, NgramModel

ROOT = Path(__file__).resolve().parent.parent


def intended_texts(train_path: Path) -> dict[str, list[str]]:
    by_lang = {"en": [], "th": []}
    with train_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["kind"] == "normal":
                by_lang[script_of_chunk(row["intended"])].append(row["intended"])
    return by_lang


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, default=ROOT / "data" / "processed" / "train.jsonl")
    p.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--only", choices=("dictionary", "ngram"))
    p.add_argument("--order", type=int, default=5, help="n-gram order")
    p.add_argument("--min-count", type=int, default=3, help="dictionary: min occurrences of an English word")
    args = p.parse_args()

    t0 = time.time()
    texts = intended_texts(args.train)
    print(f"train texts: en={len(texts['en']):,} th={len(texts['th']):,} ({time.time() - t0:.0f}s)")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "dictionary"):
        counts = Counter(w.lower() for t in texts["en"] for w in re.findall(r"[A-Za-z]+", t))
        words = sorted(w for w, c in counts.items() if c >= args.min_count)
        out = args.out_dir / "dictionary_en_words.txt"
        out.write_text("\n".join(words) + "\n", encoding="utf-8")
        print(f"dictionary: {len(words):,} English words (seen >= {args.min_count}x) -> {out}")

    if args.only in (None, "ngram"):
        lms = {}
        for lang in ("th", "en"):
            lms[lang] = CharNgramLM(args.order).fit(texts[lang])
            print(f"ngram {lang}: order={args.order} {len(lms[lang].ngram):,} n-grams ({time.time() - t0:.0f}s)")
        out = args.out_dir / "ngram.pkl"
        NgramModel(lms).save(out)
        print(f"ngram -> {out} ({out.stat().st_size / 1e6:.0f} MB)")

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
