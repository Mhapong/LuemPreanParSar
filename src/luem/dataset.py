"""Turn raw corpus text into labeled samples: typed in the right layout ("ok") or the wrong one ("wrong").

A sample is what the user sees on screen while typing one chunk (text between spaces):
    ok     "สวัสดี" typed with the th layout active
    wrong  "l;ylfu" typed with the en layout active while meaning "สวัสดี"

Labels come for free: a wrong sample is an ok chunk pushed through the key mapping in layout.py.
"""

import hashlib
import random
import string
import unicodedata
from dataclasses import asdict, dataclass

from luem.layout import EN_KEYS, TH_KEYS, en_to_th, th_to_en

MAX_LEN = 32  # same cap as the daemon's typing buffer
MIN_LEN = 2

EN_SET = frozenset(EN_KEYS)
TH_SET = frozenset(TH_KEYS)

# Characters editors/Wikipedia insert that a keyboard can't type: map to the typeable form or drop
_NORMALIZE = str.maketrans({
    "​": None, "‌": None, "‍": None, "﻿": None, "­": None, "️": None,
    " ": " ", " ": " ", "　": " ", "\t": " ",
    "–": "-", "—": "-", "−": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "…": "...",
})

SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class Sample:
    text: str      # what the screen shows
    active: str    # layout that was active: "en" | "th"
    label: str     # "ok" | "wrong"
    intended: str  # what the user meant to type
    source: str
    kind: str      # "normal" | "hard_negative" | "tech_wrong"
    doc: str

    def to_dict(self) -> dict:
        return asdict(self)


def read_samples(path, limit: int = 0, seed: int = 0) -> list[dict]:
    """Load a split written by scripts/build_dataset.py; limit>0 takes a fixed random subset."""
    import json

    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


def split_of(doc_id: str) -> str:
    """Stable 80/10/10 split by document, identical on every machine (unlike hash())."""
    bucket = int(hashlib.md5(doc_id.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_NORMALIZE)


def _has_thai_consonant(s: str) -> bool:
    return any("ก" <= c <= "ฮ" for c in s)


def _has_ascii_letter(s: str) -> bool:
    return any(c in string.ascii_letters for c in s)


def script_of_chunk(chunk: str) -> str | None:
    """Layout that typed this chunk, or None if no single layout can type it as a real word.

    Requires every character to be on that layout (so "ค.ศ.1926" is rejected: the th layout
    has no ASCII digits) and at least one letter (so "1926" and "()" are rejected).

    >>> script_of_chunk("สวัสดี"), script_of_chunk("hello"), script_of_chunk("ค.ศ.1926"), script_of_chunk("42")
    ('th', 'en', None, None)
    """
    if TH_SET.issuperset(chunk) and _has_thai_consonant(chunk):
        return "th"
    if EN_SET.issuperset(chunk) and _has_ascii_letter(chunk):
        return "en"
    return None


def chunks(text: str):
    """Yield (chunk, script) for every usable space-separated chunk of a document."""
    for raw in normalize(text).split():
        chunk = raw[:MAX_LEN]
        if len(chunk) < MIN_LEN:
            continue
        script = script_of_chunk(chunk)
        if script:
            yield chunk, script


def make_sample(chunk: str, script: str, wrong: bool, source: str, doc: str, kind: str = "normal") -> Sample | None:
    """ok: the chunk as typed in its own layout. wrong: the same keys pressed in the other layout.

    Returns None when the wrong version has no letters on screen (e.g. "คต" -> "89"): such
    strings look like numbers, and labeling numbers "wrong" would teach the model to fix them.
    """
    if not wrong:
        return Sample(chunk, script, "ok", chunk, source, kind, doc)
    shown, active = (th_to_en(chunk), "en") if script == "th" else (en_to_th(chunk), "th")
    if script_of_chunk(shown) != active:
        return None
    return Sample(shown, active, "wrong", chunk, source, kind, doc)


# ---------------------------------------------------------------- hard negatives
# Correctly typed strings that look odd: the model must learn "unusual" is not "wrong layout".

COMMANDS = (
    "git sudo apt ls cd grep cat echo pip python python3 npm npx node make cmake docker ssh scp curl wget "
    "vim nano code uv pytest chmod chown mkdir rm mv cp tar unzip kill ps top htop man systemctl journalctl "
    "export source fish bash zsh exit clear history ffmpeg gcc cargo rustc go java javac kubectl yarn pnpm "
    "conda jupyter tmux less head tail sed awk find xargs sort uniq wc diff ping ip df du free whoami uname "
    "reboot nvidia-smi"
).split()
FLAGS = "-l -la -a -h -v -r -rf -f -n -i -x --help --version --force --all --verbose -m -p -u -d".split()
ABBREVIATIONS = (
    "ok okay OK lol LOL btw pls plz thx ty np idk omg brb asap fyi imo gg wtf lmao kk jk ez afk dm irl "
    "tbh smh nvm hbd xd XD wifi WiFi usb USB pdf PDF url api API cpu GPU"
).split()
EMOTICONS = ":) :( :D ;) <3 ^^ T_T -_- :P =) 555 5555 55555 +1".split()
TLDS = "com org net io dev co.th ac.th go.th".split()


def _word(rng: random.Random, words: list[str]) -> str:
    return rng.choice(words) if words else "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))


def _identifier(rng, words):
    parts = [_word(rng, words) for _ in range(rng.randint(2, 3))]
    style = rng.randrange(4)
    if style == 0:
        return "_".join(parts)
    if style == 1:
        return parts[0] + "".join(p.capitalize() for p in parts[1:])
    if style == 2:
        return "".join(p.capitalize() for p in parts)
    return "_".join(parts).upper()


def _password(rng):
    pool = string.ascii_letters + string.digits + "!@#$%^&*_-.?"
    while True:
        s = "".join(rng.choices(pool, k=rng.randint(6, 16)))
        if _has_ascii_letter(s):
            return s


def _hex(rng):
    return "".join(rng.choices("0123456789abcdef", k=rng.choice((7, 8, 12, 16, 32))))


def _url(rng, words):
    w1, w2 = _word(rng, words), _word(rng, words)
    return rng.choice((
        f"https://{w1}.{rng.choice(TLDS)}/{w2}",
        f"www.{w1}.{rng.choice(TLDS)}",
        f"github.com/{w1}/{w2}",
        f"{w1}{rng.randint(1, 999)}@gmail.com",
        f"{w1}.{rng.choice(TLDS)}",
    ))


def _path(rng, words):
    w1, w2 = _word(rng, words), _word(rng, words)
    return rng.choice((
        f"/usr/share/{w1}", f"~/.config/{w1}", f"./{w1}.sh", f"src/{w1}/{w2}.py",
        f"/home/{w1}", f"{w1}.txt", f"{w1}_{w2}.json", f"/dev/{w1}",
    ))


def _number_like(rng, words):
    return rng.choice((
        f"{rng.randint(0, 99)}.{rng.randint(0, 99)}", f"{rng.randint(1, 100)}%", f"#{rng.randint(1, 9999)}",
        f"@{_word(rng, words)}", f"{rng.randint(2000, 2030)}-{rng.randint(1, 12):02}-{rng.randint(1, 28):02}",
        f"{rng.randint(0, 23):02}:{rng.randint(0, 59):02}", f"{rng.randint(1, 999)},{rng.randint(0, 999):03}",
        f"v{rng.randint(0, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 20)}", "x86_64", "arm64", "utf-8",
        str(rng.randint(0, 10**rng.randint(1, 9))),
    ))


def _command(rng, words):
    cmd = rng.choice(COMMANDS)
    return rng.choice((cmd, cmd, rng.choice(FLAGS), f"{cmd}.py", f"{_word(rng, words)}.{rng.choice(('py', 'js', 'c', 'md'))}"))


# (generator, also add a wrong-layout version?) Word-like strings get one: typing `git` with the
# th layout active ("เระ") is a classic real mistake. Random strings (passwords, hashes) don't.
_GENERATORS = (
    (lambda rng, w: rng.choice(ABBREVIATIONS), True),
    (lambda rng, w: rng.choice(EMOTICONS), False),
    (_command, True),
    (_identifier, True),
    (lambda rng, w: _password(rng), False),
    (lambda rng, w: _hex(rng), False),
    (_url, False),
    (_path, False),
    (_number_like, False),
)


def hard_negatives(n: int, rng: random.Random, words: list[str], split: str) -> list[Sample]:
    """n correctly typed tech/odd strings (label ok), plus wrong-layout versions of the word-like ones."""
    out, n_ok = [], 0
    while n_ok < n:
        gen, add_wrong = rng.choice(_GENERATORS)
        s = gen(rng, words)[:MAX_LEN]
        if len(s) < MIN_LEN or not EN_SET.issuperset(s):
            continue
        out.append(Sample(s, "en", "ok", s, "synthetic", "hard_negative", f"synthetic:{split}"))
        n_ok += 1
        if add_wrong and rng.random() < 0.5:
            wrong = make_sample(s, "en", True, "synthetic", f"synthetic:{split}", kind="tech_wrong")
            if wrong:
                out.append(wrong)
    return out
