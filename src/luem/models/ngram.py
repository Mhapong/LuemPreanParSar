"""Model 2: character n-gram language models + Bayes rule (a noisy channel model).

One LM per language says how "Thai-like" or "English-like" a string is. For what is on screen x:

    P(wrong | x) = sigmoid( log P_other(convert(x)) - log P_active(x) )

i.e. which reading of the same key presses is more plausible language, with equal priors.
Smoothing: interpolated Witten-Bell, so unseen n-grams still get a small probability.
"""

import math
import pickle
from collections import Counter
from pathlib import Path

from luem.models.base import Model, views

BOS = "\x02"   # start-of-chunk marker: chunks are words, so "how words start" is informative
VOCAB = 200    # base distribution: uniform over roughly the typeable characters


class CharNgramLM:
    def __init__(self, order: int = 5):
        self.n = order
        self.ngram = Counter()     # h + c -> count, for every context length 0..n-1
        self.ctx_total = Counter() # h -> count
        self.ctx_types = Counter() # h -> number of distinct chars seen after h

    def fit(self, texts) -> "CharNgramLM":
        n, ngram, total, types = self.n, self.ngram, self.ctx_total, self.ctx_types
        pad = BOS * (n - 1)
        for text in texts:
            s = pad + text
            for i in range(n - 1, len(s)):
                c = s[i]
                for k in range(n):
                    h = s[i - k:i]
                    key = h + c
                    ngram[key] += 1
                    total[h] += 1
                    if ngram[key] == 1:
                        types[h] += 1
        return self

    def prob(self, h: str, c: str) -> float:
        """Witten-Bell: P(c|h) = (C(hc) + T(h) * P(c|h[1:])) / (C(h) + T(h))."""
        p = 1.0 / VOCAB
        for k in range(len(h) + 1):  # shortest context first, each level smooths with the one below
            ctx = h[len(h) - k:]
            t = self.ctx_total.get(ctx, 0)
            if t == 0:
                continue
            types = self.ctx_types[ctx]
            p = (self.ngram.get(ctx + c, 0) + types * p) / (t + types)
        return p

    def logprobs(self, text: str) -> list[float]:
        """log P of each character given the previous n-1 characters."""
        s = BOS * (self.n - 1) + text
        return [math.log(self.prob(s[i - self.n + 1:i], s[i])) for i in range(self.n - 1, len(s))]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x > -500 else 0.0


class NgramModel(Model):
    name = "ngram"

    def __init__(self, lms: dict[str, CharNgramLM]):
        self.lms = lms  # {"en": ..., "th": ...}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({lang: (lm.n, lm.ngram, lm.ctx_total, lm.ctx_types) for lang, lm in self.lms.items()}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "NgramModel":
        with path.open("rb") as f:
            raw = pickle.load(f)
        lms = {}
        for lang, (n, ngram, total, types) in raw.items():
            lm = CharNgramLM(n)
            lm.ngram, lm.ctx_total, lm.ctx_types = ngram, total, types
            lms[lang] = lm
        return cls(lms)

    def predict(self, text: str) -> float:
        return self.predict_prefixes(text)[-1] if text else 0.0

    def predict_prefixes(self, text: str) -> list[float]:
        """One pass: the log-likelihood ratio of prefix k is a running sum over its characters."""
        v = views(text)
        if v is None:
            # the full text has no letter, so no prefix has one either
            return [0.0] * len(text)
        active, typed, other, converted = v
        lp_typed = self.lms[active].logprobs(typed)
        lp_conv = self.lms[other].logprobs(converted)
        out, llr = [], 0.0
        for k in range(len(text)):
            llr += lp_conv[k] - lp_typed[k]
            # a prefix without a letter yet (e.g. "(" of "(director") carries no evidence
            out.append(_sigmoid(llr) if views(text[:k + 1]) else 0.0)
        return out
