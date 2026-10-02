import random
from collections import Counter

from luem.dataset import (
    EN_SET, MAX_LEN, MIN_LEN, chunks, hard_negatives, make_sample, normalize, script_of_chunk, split_of,
)
from luem.layout import en_to_th, th_to_en


def test_split_is_stable_and_about_80_10_10():
    assert split_of("wiki_th:320526") == split_of("wiki_th:320526")
    counts = Counter(split_of(f"doc:{i}") for i in range(20_000))
    assert 0.78 < counts["train"] / 20_000 < 0.82
    assert 0.08 < counts["val"] / 20_000 < 0.12


def test_normalize_maps_untypeable_lookalikes():
    assert normalize("a​b") == "ab"
    assert normalize("“hi” – it’s") == '"hi" - it\'s'
    assert normalize("a b") == "a b"


def test_script_requires_single_layout_and_a_letter():
    assert script_of_chunk("สวัสดี") == "th"
    assert script_of_chunk("hello,") == "en"
    assert script_of_chunk("ไปWiFi") is None      # mixed layouts in one chunk
    assert script_of_chunk("ค.ศ.1926") is None    # th layout has no ASCII digits
    assert script_of_chunk("1926") is None        # no letters
    assert script_of_chunk("café") is None        # é is not on either layout


def test_chunks_filter_and_truncate():
    text = "สวัสดี 😂 hello ไปWiFi a " + "x" * 40
    got = list(chunks(text))
    assert ("สวัสดี", "th") in got and ("hello", "en") in got
    assert all(MIN_LEN <= len(c) <= MAX_LEN for c, _ in got)
    assert ("x" * MAX_LEN, "en") in got
    assert not any(c in ("😂", "ไปWiFi", "a") for c, _ in got)


def test_wrong_samples_map_back_to_intended():
    th = make_sample("สวัสดี", "th", True, "t", "d")
    assert (th.text, th.active, th.label) == ("l;ylfu", "en", "wrong")
    assert en_to_th(th.text) == th.intended
    en = make_sample("hello", "en", True, "t", "d")
    assert (en.text, en.active) == ("้ำสสน", "th")
    assert th_to_en(en.text) == en.intended


def test_ok_sample_is_unchanged():
    s = make_sample("hello", "en", False, "t", "d")
    assert (s.text, s.active, s.label, s.intended) == ("hello", "en", "ok", "hello")


def test_wrong_sample_dropped_when_screen_shows_no_letters():
    assert th_to_en("คต") == "89"
    assert make_sample("คต", "th", True, "t", "d") is None


def test_hard_negatives_are_typeable_and_ok():
    samples = hard_negatives(2000, random.Random(0), ["alpha", "beta", "gamma"], "train")
    oks = [s for s in samples if s.label == "ok"]
    assert len(oks) == 2000
    assert all(s.kind == "hard_negative" and s.active == "en" and EN_SET.issuperset(s.text) for s in oks)
    assert all(MIN_LEN <= len(s.text) <= MAX_LEN for s in samples)
    wrongs = [s for s in samples if s.label == "wrong"]
    assert wrongs and all(s.kind == "tech_wrong" and th_to_en(s.text) == s.intended for s in wrongs)
