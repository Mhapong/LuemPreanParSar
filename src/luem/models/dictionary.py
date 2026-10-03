"""Model 1 (baseline): look both readings up in a dictionary.

For "l;ylfu" compare how much of "l;ylfu" is English words with how much of "สวัสดี" is Thai words.
The last word may still be half-typed, so it also counts if it is the start of a dictionary word.
"""

import re
from pathlib import Path

from luem.models.base import Model, views

_EN_WORD = re.compile(r"[A-Za-z]+")


def _prefixes(words) -> set[str]:
    return {w[:k] for w in words for k in range(1, len(w) + 1)}


class DictionaryModel(Model):
    name = "dictionary"

    def __init__(self, en_words: set[str], th_words: set[str]):
        from pythainlp.tokenize import word_tokenize  # heavy import, only needed here

        self._tokenize = word_tokenize
        self.en_words, self.th_words = en_words, th_words
        self.en_prefixes, self.th_prefixes = _prefixes(en_words), _prefixes(th_words)

    @classmethod
    def load(cls, en_words_path: Path) -> "DictionaryModel":
        from pythainlp.corpus import thai_words

        en = set(en_words_path.read_text(encoding="utf-8").split())
        return cls(en, set(thai_words()))

    def coverage(self, text: str, lang: str) -> float:
        """Fraction of letters that belong to dictionary words (last word: prefix of one is enough)."""
        if lang == "en":
            tokens = [t.lower() for t in _EN_WORD.findall(text)]
            words, prefixes = self.en_words, self.en_prefixes
        else:
            tokens = [t for t in self._tokenize(text, engine="newmm", keep_whitespace=False) if t.strip()]
            words, prefixes = self.th_words, self.th_prefixes
        total = sum(len(t) for t in tokens)
        if total == 0:
            return 0.0
        hit = sum(len(t) for t in tokens[:-1] if t in words)
        last = tokens[-1]
        hit += len(last) if (last in words or last in prefixes) else 0
        return hit / total

    def predict(self, text: str) -> float:
        v = views(text)
        if v is None:
            return 0.0
        active, typed, other, converted = v
        return 0.5 + 0.5 * (self.coverage(converted, other) - self.coverage(typed, active))
