import re
import sys
import unicodedata

from luem.layout import EN_KEYS, TH_KEYS

KEY_ORDER = (
    ["TLDE"] + [f"AE{i:02}" for i in range(1, 13)]
    + [f"AD{i:02}" for i in range(1, 13)]
    + [f"AC{i:02}" for i in range(1, 12)] + ["BKSL"]
    + [f"AB{i:02}" for i in range(1, 11)]
)
ASCII_KEYSYMS = {
    "underscore": "_", "percent": "%", "plus": "+", "slash": "/", "minus": "-",
    "quotedbl": '"', "comma": ",", "period": ".", "parenleft": "(",
    "parenright": ")", "question": "?",
}
THAI_DIGITS = ["sun", "nung", "song", "sam", "si", "ha", "hok", "chet", "paet", "kao"]


def keysym_to_char(name: str) -> str:
    if name in ASCII_KEYSYMS:
        return ASCII_KEYSYMS[name]
    if name == "Thai_baht":
        return "฿"
    if name.startswith("Thai_lek"):
        return chr(0x0E50 + THAI_DIGITS.index(name.removeprefix("Thai_lek")))
    want = name.removeprefix("Thai_").lower()
    for cp in range(0x0E01, 0x0E5C):
        uname = unicodedata.name(chr(cp), "")
        if uname.removeprefix("THAI CHARACTER ").replace(" ", "").replace("-", "").lower() == want:
            return chr(cp)
    raise KeyError(name)


def main(path: str) -> int:
    src = open(path, encoding="utf-8").read()
    basic = src.split('xkb_symbols "basic"')[1].split("\n};")[0]
    pattern = r"key <(\w+)>\s*\{\[\s*(\w+),\s*(\w+)\s*\]\}"
    keys = {k: (lo, up) for k, lo, up in re.findall(pattern, basic)}
    lower = "".join(keysym_to_char(keys[k][0]) for k in KEY_ORDER)
    upper = "".join(keysym_to_char(keys[k][1]) for k in KEY_ORDER)
    expected = lower + upper

    bad = [(EN_KEYS[i], TH_KEYS[i], expected[i]) for i in range(len(expected)) if TH_KEYS[i] != expected[i]]
    if len(TH_KEYS) != len(expected):
        print(f"length mismatch: layout.py={len(TH_KEYS)} xkb={len(expected)}")
        return 1
    for en, got, want in bad:
        print(f"key {en!r}: layout.py has {got!r}, xkb has {want!r}")
    print("OK: all 94 keys match xkb" if not bad else f"{len(bad)} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/usr/share/X11/xkb/symbols/th"))
