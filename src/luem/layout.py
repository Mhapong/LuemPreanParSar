"""Kedmanee (Thai) <-> US QWERTY key mapping.

Each physical key produces one character per (layout, shift) pair, so the
mapping is a bijection between the 94 printable US characters and the 94
characters the Thai Kedmanee layout produces. Characters outside the table
(space, newline, ...) pass through unchanged.

Source of truth: /usr/share/X11/xkb/symbols/th, section "basic".
Verified by scripts/verify_layout.py.
"""

# Rows in physical key order: TLDE AE01..AE12 / AD01..AD12 / AC01..AC11 BKSL / AB01..AB10
_EN_LOWER = "`1234567890-=" "qwertyuiop[]" "asdfghjkl;'\\" "zxcvbnm,./"
_EN_UPPER = '~!@#$%^&*()_+' "QWERTYUIOP{}" 'ASDFGHJKL:"|' "ZXCVBNM<>?"
_TH_LOWER = "_ๅ/-ภถุึคตจขช" "ๆไำพะัีรนยบล" "ฟหกดเ้่าสวงฃ" "ผปแอิืทมใฝ"
_TH_UPPER = "%+๑๒๓๔ู฿๕๖๗๘๙" '๐"ฎฑธํ๊ณฯญฐ,' "ฤฆฏโฌ็๋ษศซ.ฅ" "()ฉฮฺ์?ฒฬฦ"

# Linux evdev keycodes (linux/input-event-codes.h) in the same physical key order
KEYCODES: tuple[int, ...] = (
    (41, *range(2, 12), 12, 13)                 # ` 1..0 - =
    + (*range(16, 26), 26, 27)                  # q..p [ ]
    + (*range(30, 39), 39, 40, 43)              # a..l ; ' \
    + (*range(44, 51), 51, 52, 53)              # z..m , . /
)
assert len(KEYCODES) == len(_EN_LOWER)

EN_KEYS = _EN_LOWER + _EN_UPPER
TH_KEYS = _TH_LOWER + _TH_UPPER

EN2TH: dict[str, str] = dict(zip(EN_KEYS, TH_KEYS, strict=True))
TH2EN: dict[str, str] = dict(zip(TH_KEYS, EN_KEYS, strict=True))

assert len(EN2TH) == len(TH2EN) == len(EN_KEYS), "layout must be a bijection"

# (keycode, shift) <-> character, per layout
_KEY_TO_EN = {(k, s): c for s, row in ((False, _EN_LOWER), (True, _EN_UPPER)) for k, c in zip(KEYCODES, row)}
_KEY_TO_TH = {(k, s): c for s, row in ((False, _TH_LOWER), (True, _TH_UPPER)) for k, c in zip(KEYCODES, row)}
_EN_TO_KEY = {c: ks for ks, c in _KEY_TO_EN.items()}
_TH_TO_KEY = {c: ks for ks, c in _KEY_TO_TH.items()}

_EN2TH_TABLE = str.maketrans(EN2TH)
_TH2EN_TABLE = str.maketrans(TH2EN)


def en_to_th(text: str) -> str:
    """Text typed while the US layout was active, as if Thai had been active.

    >>> en_to_th("l;ylfu")
    'สวัสดี'
    """
    return text.translate(_EN2TH_TABLE)


def th_to_en(text: str) -> str:
    """Text typed while the Thai layout was active, as if US had been active.

    >>> th_to_en("้ำสสน")
    'hello'
    """
    return text.translate(_TH2EN_TABLE)


def key_to_char(keycode: int, shift: bool, layout: str) -> str | None:
    """Character produced by an evdev keycode in layout 'en' or 'th' (None if not a printable key).

    >>> key_to_char(38, False, "en"), key_to_char(38, False, "th")
    ('l', 'ส')
    """
    return (_KEY_TO_EN if layout == "en" else _KEY_TO_TH).get((keycode, shift))


def char_to_key(c: str, layout: str) -> tuple[int, bool] | None:
    """(evdev keycode, shift) that types character c in layout 'en' or 'th'.

    Layout matters: '/' is KEY_SLASH in 'en' but KEY_2 in 'th'.
    """
    return (_EN_TO_KEY if layout == "en" else _TH_TO_KEY).get(c)


def is_thai_char(c: str) -> bool:
    return "฀" <= c <= "๿"


def script_of(text: str) -> str:
    """'th', 'en', or 'other' by majority of Thai vs ASCII-letter characters."""
    th = sum(is_thai_char(c) for c in text)
    en = sum(c.isascii() and c.isalpha() for c in text)
    if th == en == 0:
        return "other"
    return "th" if th >= en else "en"
