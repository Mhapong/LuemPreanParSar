import math

import pytest

from luem.models.base import views
from luem.models.ngram import VOCAB, CharNgramLM, NgramModel

EN = ["hello", "help", "hell", "world", "word", "the", "there", "this", "that", "status", "station"] * 20
TH = ["สวัสดี", "สวัส", "ครับ", "ดีครับ", "ภาษา", "ไทย", "ภาษาไทย", "สถานี", "ที่", "นี่"] * 20


@pytest.fixture(scope="module")
def model():
    return NgramModel({"en": CharNgramLM(4).fit(EN), "th": CharNgramLM(4).fit(TH)})


def test_views():
    assert views("hello") == ("en", "hello", "th", "้ำสสน")
    assert views("สวัสดี") == ("th", "สวัสดี", "en", "l;ylfu")
    assert views("555") is None and views("เ") is None


def test_witten_bell_is_a_distribution():
    lm = CharNgramLM(3).fit(EN)
    seen = sorted({c for w in EN for c in w})
    alphabet = seen + [chr(0x2500 + i) for i in range(VOCAB - len(seen))]
    for h in ("", "h", "he", "zz"):
        assert math.isclose(sum(lm.prob(h, c) for c in alphabet), 1.0, rel_tol=1e-9)


def test_ngram_detects_wrong_layout(model):
    assert model.predict("l;ylfu") > 0.9
    assert model.predict("้ำสสน") > 0.9
    assert model.predict("hello") < 0.1
    assert model.predict("สวัสดี") < 0.1


def test_no_letters_means_no_fix(model):
    assert model.predict("555") == 0.0
    assert model.predict_prefixes("(hello") [0] == 0.0


def test_prefix_scores_match_predict(model):
    text = "l;ylfu"
    pre = model.predict_prefixes(text)
    assert len(pre) == len(text)
    for k in range(1, len(text) + 1):
        assert math.isclose(pre[k - 1], model.predict(text[:k]), rel_tol=1e-9)


def test_save_load_roundtrip(model, tmp_path):
    path = tmp_path / "ngram.pkl"
    model.save(path)
    assert NgramModel.load(path).predict("l;ylfu") == model.predict("l;ylfu")
