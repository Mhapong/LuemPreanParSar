_EN_LOWER = "`1234567890-=" "qwertyuiop[]" "asdfghjkl;'\\" "zxcvbnm,./"
_EN_UPPER = "~!@#$%^&*()_+" "QWERTYUIOP{}" 'ASDFGHJKL:"|' "ZXCVBNM<>?"
_TH_LOWER = "_ๅ/-ภถุึคตจขช" "ๆไำพะัีรนยบล" "ฟหกดเ้่าสวงฃ" "ผปแอิืทมใฝ"
_TH_UPPER = "%+๑๒๓๔ู฿๕๖๗๘๙" '๐"ฎฑธํ๊ณฯญฐ,' "ฤฆฏโฌ็๋ษศซ.ฅ" "()ฉฮฺ์?ฒฬฦ"
KEYCODES: tuple[int, ...] = (
    (41, *range(2, 12), 12, 13)
    + (*range(16, 26), 26, 27)
    + (*range(30, 39), 39, 40, 43)
    + (*range(44, 51), 51, 52, 53)
)
assert len(KEYCODES) == len(_EN_LOWER)

EN_KEYS = _EN_LOWER + _EN_UPPER
TH_KEYS = _TH_LOWER + _TH_UPPER
EN2TH: dict[str, str] = dict(zip(EN_KEYS, TH_KEYS, strict=True))
TH2EN: dict[str, str] = dict(zip(TH_KEYS, EN_KEYS, strict=True))

assert len(EN2TH) == len(TH2EN) == len(EN_KEYS), "layout must be a bijection"

_KEY_TO_EN = {
    (k, s): c
    for s, row in ((False, _EN_LOWER), (True, _EN_UPPER))
    for k, c in zip(KEYCODES, row)
}
_KEY_TO_TH = {
    (k, s): c
    for s, row in ((False, _TH_LOWER), (True, _TH_UPPER))
    for k, c in zip(KEYCODES, row)
}
_EN_TO_KEY = {c: ks for ks, c in _KEY_TO_EN.items()}
_TH_TO_KEY = {c: ks for ks, c in _KEY_TO_TH.items()}

_EN2TH_TABLE = str.maketrans(EN2TH)
_TH2EN_TABLE = str.maketrans(TH2EN)


def en_to_th(text: str) -> str:
    return text.translate(_EN2TH_TABLE)


def th_to_en(text: str) -> str:
    return text.translate(_TH2EN_TABLE)


def key_to_char(keycode: int, shift: bool, layout: str) -> str | None:
    return (_KEY_TO_EN if layout == "en" else _KEY_TO_TH).get((keycode, shift))


def char_to_key(c: str, layout: str) -> tuple[int, bool] | None:
    return (_EN_TO_KEY if layout == "en" else _TH_TO_KEY).get(c)


def is_thai_char(c: str) -> bool:
    return "฀" <= c <= "๿"


def script_of(text: str) -> str:
    th = sum(is_thai_char(c) for c in text)
    en = sum(c.isascii() and c.isalpha() for c in text)
    if th == en == 0:
        return "other"
    return "th" if th >= en else "en"
