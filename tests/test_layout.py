import string

import pytest

from luem.layout import EN2TH, EN_KEYS, KEYCODES, TH2EN, char_to_key, en_to_th, key_to_char, script_of, th_to_en


def test_keycodes_match_evdev_names():
    ecodes = pytest.importorskip("evdev.ecodes")
    names = ["GRAVE", *"1234567890", "MINUS", "EQUAL", *"QWERTYUIOP", "LEFTBRACE", "RIGHTBRACE",
             *"ASDFGHJKL", "SEMICOLON", "APOSTROPHE", "BACKSLASH", *"ZXCVBNM", "COMMA", "DOT", "SLASH"]
    assert KEYCODES == tuple(getattr(ecodes, f"KEY_{n}") for n in names)


def test_key_char_round_trip():
    for layout, chars in (("en", EN_KEYS), ("th", "".join(EN2TH.values()))):
        for c in chars:
            assert key_to_char(*char_to_key(c, layout), layout) == c


def test_shared_symbols_differ_per_layout():
    assert char_to_key("/", "en") == (53, False)
    assert char_to_key("/", "th") == (3, False)


def test_bijection_covers_all_printable_ascii():
    printable = set(string.printable) - set(string.whitespace)
    assert set(EN2TH) == printable
    assert len(set(EN2TH.values())) == len(EN2TH)


def test_known_examples():
    assert en_to_th("l;ylfu") == "สวัสดี"
    assert th_to_en("้ำสสน") == "hello"
    assert th_to_en("ภาษาไทย") == "4kKkwmp"
    assert en_to_th("git status") == "เระ หะฟะีห"


def test_round_trip():
    th = "ภาษาไทยง่ายนิดเดียว ๑๒๓ ฿"
    en = "The quick brown fox jumps over the lazy dog! @#$%^&*()"
    assert en_to_th(th_to_en(th)) == th
    assert th_to_en(en_to_th(en)) == en


def test_passthrough():
    assert en_to_th(" \n\t") == " \n\t"
    assert th_to_en("😀") == "😀"


def test_script_of():
    assert script_of("สวัสดี") == "th"
    assert script_of("hello") == "en"
    assert script_of("1234 !!") == "other"
