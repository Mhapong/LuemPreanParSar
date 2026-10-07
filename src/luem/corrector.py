from dataclasses import dataclass, field
from typing import Protocol

from luem.dataset import MAX_LEN
from luem.layout import KEYCODES, key_to_char

KEY_ESC, KEY_1, KEY_2, KEY_BACKSPACE, KEY_TAB = 1, 2, 3, 14, 15
KEY_P, KEY_ENTER, KEY_LEFTCTRL, KEY_F, KEY_LEFTSHIFT = 25, 28, 29, 33, 42
KEY_RIGHTSHIFT, KEY_LEFTALT, KEY_SPACE, KEY_CAPSLOCK = 54, 56, 57, 58
KEY_KPENTER, KEY_RIGHTCTRL, KEY_RIGHTALT = 96, 97, 100
KEY_LEFTMETA, KEY_RIGHTMETA = 125, 126

SHIFTS = {KEY_LEFTSHIFT, KEY_RIGHTSHIFT}
CTRLS = {KEY_LEFTCTRL, KEY_RIGHTCTRL}
ALTS = {KEY_LEFTALT, KEY_RIGHTALT}
METAS = {KEY_LEFTMETA, KEY_RIGHTMETA}
MODIFIERS = SHIFTS | CTRLS | ALTS | METAS
PRINTABLE = frozenset(KEYCODES)
HOTKEY_FIX, HOTKEY_PAUSE = KEY_F, KEY_P
LAYOUT_KEYS = {KEY_1: "en", KEY_2: "th"}

Key = tuple[int, bool]  # (keycode, shift)
_THAI_DEPENDENT = frozenset(map(chr, range(0x0E30, 0x0E4F))) - set("เแโใไๆ฿")


def impossible_thai(word: str, at_start: bool = True) -> bool:
    marks = []
    for ch in word:
        if "ก" <= ch <= "ฮ":
            break
        if ch in _THAI_DEPENDENT:
            marks.append(ch)
    if at_start:
        return bool(marks)
    return len(marks) >= 3 and len(set(marks)) == len(marks) and "ํ" not in marks


class Injector(Protocol):
    def backspace(self, n: int) -> None: ...
    def select_layout(self, layout: str) -> None: ...
    def press_keys(self, keys: list[Key]) -> None: ...


class Scorer(Protocol):
    def predict(self, text: str) -> float: ...


@dataclass
class Thresholds:
    k_min: int = 1
    tau_type: float = 0.999
    tau_space: float = 0.96


def other(layout: str) -> str:
    return "th" if layout == "en" else "en"


def render(keys: list[Key], layout: str) -> str:
    return "".join(key_to_char(code, shift, layout) for code, shift in keys)


@dataclass
class Event:
    kind: str  # "fix" | "would-fix" | "score" | "layout" | "reset" | "pause"
    detail: str = ""
    p: float | None = None


@dataclass
class Corrector:
    model: Scorer
    injector: Injector
    th: Thresholds = field(default_factory=Thresholds)
    layout: str = "en"  # active layout as we track it
    enabled: bool = True
    dry_run: bool = False  # only report what would be fixed, never touch the text
    caps_lock: bool = False  # set by the daemon from the keyboard LED
    keys: list[Key] = field(default_factory=list)  # word being typed
    fixed: bool = False  # current word already converted: never auto-convert it again
    last_word: list[Key] | None = None  # word before the last Space (for manual fix)
    last_fixed: bool = False
    pending: tuple[str, str, float | None] | None = (
        None  # ("word"|"last", why, p): run at flush()
    )
    at_start: bool = (
        False  # the word began right after a Space/Enter/Tab we saw (not after a click)
    )
    last_start: bool = False
    held: set[int] = field(default_factory=set)
    log: list[Event] = field(default_factory=list)

    # ------------------------------------------------------------------ input

    def start(self) -> None:
        self.injector.select_layout("en")
        self.layout = "en"
        self._emit("layout", "start: en")

    def on_key(self, code: int, value: int) -> None:
        if code in MODIFIERS:
            (self.held.add if value else self.held.discard)(code)
            return
        if value == 0:
            return
        shift = bool(self.held & SHIFTS)
        meta, alt, ctrl = (
            bool(self.held & METAS),
            bool(self.held & ALTS),
            bool(self.held & CTRLS),
        )
        if value == 2 and (meta or alt or ctrl):
            return  # a held shortcut repeats; counting each repeat as a Super+Space toggle drifted the layout

        if meta and alt and code in LAYOUT_KEYS:  # user selects a layout directly
            self.layout = LAYOUT_KEYS[code]
            self._reset(f"user selected {self.layout}", forget_last=True)
        elif meta and alt and code == HOTKEY_FIX:
            self._manual_fix()
        elif meta and alt and code == HOTKEY_PAUSE:
            self.enabled = not self.enabled
            self._emit("pause", "resumed" if self.enabled else "paused")
        elif meta and code == KEY_SPACE:  # user toggles the layout
            self.layout = other(self.layout)
            self._reset(f"user toggled to {self.layout}", forget_last=True)
        elif ctrl or alt or meta:  # any other shortcut: cursor/text may change
            self._reset("shortcut")
            self.at_start = False
        elif code == KEY_BACKSPACE:
            self._backspace()
        elif code == KEY_SPACE:
            self._space()
        elif code in PRINTABLE:
            self._letter(code, shift)
        else:  # Enter, Tab, arrows, numpad, F-keys, ...
            self._reset("non-text key", forget_last=True)
            self.at_start = code in (
                KEY_ENTER,
                KEY_KPENTER,
                KEY_TAB,
            )  # a new line/field starts a word

    def on_layout(self, layout: str) -> None:
        if layout != self.layout:
            self.layout = layout
            self._reset(f"desktop switched to {layout}", forget_last=True)
            self._emit("layout", f"desktop says {layout}")

    def on_click(self) -> None:
        self._reset("mouse click", forget_last=True)
        self.at_start = False

    # ------------------------------------------------------------------ typing

    def _letter(self, code: int, shift: bool) -> None:
        if (
            self.pending and self.pending[0] == "last"
        ):  # a new word started before the fix ran
            self._drop_pending("next word started")
        self.last_word = None
        if self.caps_lock:  # Caps Lock changes what keys produce; don't guess
            self._reset("caps lock on")
            return
        self.keys.append((code, shift))
        if (
            self.pending
            or not self.enabled
            or self.fixed
            or len(self.keys) < self.th.k_min
            or len(self.keys) > MAX_LEN
        ):
            return
        text = render(self.keys, self.layout)
        p = self._score(text)
        if p >= self.th.tau_type:
            self.pending = ("word", "while typing", p)

    def _space(self) -> None:
        if (
            self.pending and self.pending[0] == "word"
        ):  # decided while typing, Space came first
            self.pending = ("last", *self.pending[1:])
        elif self.pending:  # second Space before the fix ran
            self._drop_pending("kept typing")
        elif (
            self.keys and self.enabled and not self.fixed and len(self.keys) <= MAX_LEN
        ):
            p = self._score(render(self.keys, self.layout), "␣")
            if p >= self.th.tau_space:
                # the app already received this Space: the fix deletes it too and types it again
                self.pending = ("last", "on Space", p)
        self.last_word, self.last_fixed, self.last_start = (
            (self.keys or None),
            self.fixed,
            self.at_start,
        )
        self.keys, self.fixed, self.at_start = [], False, True

    def _backspace(self) -> None:
        if self.pending:
            self._drop_pending("user is editing")
        if self.keys:
            self.keys.pop()
            if not self.keys:
                self.fixed = False
        elif self.last_word:  # deleted the Space: we are back inside the previous word
            self.keys, self.fixed, self.at_start = (
                self.last_word,
                self.last_fixed,
                self.last_start,
            )
            self.last_word = None

    def _score(self, text: str, suffix: str = "") -> float:
        """P(wrong layout) for the word on screen: Thai spelling rule first, then the model."""
        if self.layout == "th" and impossible_thai(text, self.at_start):
            p, why = 1.0, "  (Thai word can't start like this)"
        else:
            p, why = self.model.predict(text), ""
        self._emit("score", text + suffix + why, p)
        return p

    def _manual_fix(self) -> None:
        """Convert the word being typed, or the one just finished. Converting twice undoes it."""
        if self.keys:
            self.pending = ("word", "manual", None)
        elif self.last_word:
            self.pending = ("last", "manual (last word)", None)
        else:
            self._emit("reset", "manual fix: nothing to convert")

    # ------------------------------------------------------------------ output

    def flush(self) -> None:
        """Run the pending fix. Call only while no key is held (see module docstring)."""
        if not self.pending:
            return
        target, why, p = self.pending
        self.pending = None
        if target == "word" and self.keys:
            self._convert(self.keys, with_space=False, why=why, p=p)
            self.fixed = True  # the user keeps typing this word in the right layout now
        elif target == "last" and self.last_word:
            self._convert(self.last_word, with_space=True, why=why, p=p)
            self.last_fixed = True

    def _drop_pending(self, why: str) -> None:
        self._emit("reset", f"fix not done: {why}")
        self.pending = None

    def _convert(
        self, keys: list[Key], with_space: bool, why: str, p: float | None = None
    ) -> None:
        before, target = render(keys, self.layout), other(self.layout)
        if self.dry_run:
            self._emit(
                "would-fix", f"{before!r} -> {render(keys, target)!r} ({why})", p
            )
            return
        self.injector.backspace(len(keys) + with_space)
        self.injector.select_layout(
            target
        )  # absolute selection: correct even if tracking drifted
        self.layout = target
        self.injector.press_keys(keys + ([(KEY_SPACE, False)] if with_space else []))
        self._emit("fix", f"{before!r} -> {render(keys, target)!r} ({why})", p)

    def _reset(self, why: str, forget_last: bool = False) -> None:
        if self.keys or self.pending or (forget_last and self.last_word):
            self._emit("reset", why)
        if self.keys:  # what follows continues a word already on screen
            self.at_start = False
        self.keys, self.fixed, self.pending = [], False, None
        if forget_last:
            self.last_word = None

    def _emit(self, kind: str, detail: str, p: float | None = None) -> None:
        self.log.append(Event(kind, detail, p))
