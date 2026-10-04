"""Corrector against a simulated desktop: what ends up on screen after typing, not just which calls ran."""

import pytest

from luem import corrector as c
from luem.corrector import Corrector, Thresholds
from luem.layout import char_to_key, key_to_char


class Desktop:
    """Fake app + keyboard layout. Physical keys and injected keys both land on `screen`."""

    def __init__(self, layout="en"):
        self.layout, self.screen, self.ops = layout, "", []

    def _type(self, code, shift):
        self.screen += " " if code == c.KEY_SPACE else key_to_char(code, shift, self.layout)

    # Injector interface
    def backspace(self, n):
        self.ops.append(("backspace", n))
        self.screen = self.screen[:-n] if n else self.screen

    def select_layout(self, layout):
        self.ops.append(("layout", layout))
        self.layout = layout

    def press_keys(self, keys):
        self.ops.append(("keys", len(keys)))
        for code, shift in keys:
            self._type(code, shift)


class Model:
    """Scores from a table; anything not listed is 0 (typed correctly)."""

    def __init__(self, scores=None, default=0.0):
        self.scores, self.default = scores or {}, default

    def predict(self, text):
        return self.scores.get(text, self.default)


def setup(scores=None, default=0.0, desktop_layout="en", **th):
    d = Desktop(desktop_layout)
    k = Corrector(Model(scores, default), d, Thresholds(**{"k_min": 1, "tau_type": 0.999, "tau_space": 0.96, **th}))
    return d, k


def press(d, k, code, shift=False, mods=(), flush=True):
    """Press and release a key; then, like the daemon, flush once no key is held."""
    for m in mods:
        k.on_key(m, 1)
    if shift:
        k.on_key(c.KEY_LEFTSHIFT, 1)
    if code == c.KEY_BACKSPACE:
        d.screen = d.screen[:-1]
    elif not mods and (code == c.KEY_SPACE or code in c.PRINTABLE):
        d._type(code, shift)
    k.on_key(code, 1)
    k.on_key(code, 0)
    if shift:
        k.on_key(c.KEY_LEFTSHIFT, 0)
    for m in mods:
        k.on_key(m, 0)
    if flush:
        k.flush()


def type_meaning(d, k, word, meant_layout, flush=True):
    """Press the keys that would produce `word` in `meant_layout` (whatever layout is really active)."""
    for ch in word:
        if ch == " ":
            press(d, k, c.KEY_SPACE, flush=flush)
        else:
            press(d, k, *char_to_key(ch, meant_layout), flush=flush)


SUPER_ALT = (c.KEY_LEFTMETA, c.KEY_LEFTALT)


def test_keycodes_match_evdev():
    e = pytest.importorskip("evdev.ecodes")
    for name in ("ESC", "1", "2", "BACKSPACE", "TAB", "P", "ENTER", "LEFTCTRL", "F", "LEFTSHIFT", "RIGHTSHIFT",
                 "LEFTALT", "SPACE", "CAPSLOCK", "KPENTER", "RIGHTCTRL", "RIGHTALT", "LEFTMETA", "RIGHTMETA"):
        assert getattr(c, f"KEY_{name}") == getattr(e, f"KEY_{name}"), name


def test_start_selects_english():
    d, k = setup(desktop_layout="th")
    k.start()
    assert d.layout == "en" and k.layout == "en"


def test_fix_while_typing():
    d, k = setup({"l;y": 0.9995})  # meant สวัสดี, layout was en
    type_meaning(d, k, "สวัสดี", "th")
    assert d.screen == "สวัสดี" and d.layout == "th" and k.layout == "th"
    assert d.ops == [("backspace", 3), ("layout", "th"), ("keys", 3)]  # fixed after 3 keys, then typed on


def test_fix_on_space_also_retypes_the_space():
    d, k = setup({"้ำสสน": 0.97}, desktop_layout="th")  # meant hello, layout was th
    k.layout = "th"
    type_meaning(d, k, "hello ", "en")
    assert d.screen == "hello " and d.layout == "en"
    assert d.ops == [("backspace", 6), ("layout", "en"), ("keys", 6)]


def test_correct_text_is_left_alone():
    d, k = setup()
    type_meaning(d, k, "hello world ", "en")
    assert d.screen == "hello world " and d.ops == []


def test_converted_word_is_never_converted_back():
    d, k = setup(default=0.9999)  # model would flag everything
    type_meaning(d, k, "สวัสดี", "th")
    assert d.screen.startswith("ส") and d.layout == "th"
    assert sum(op[0] == "layout" for op in d.ops) == 1


def test_manual_fix_converts_last_word_and_undoes():
    d, k = setup()  # model never fires
    type_meaning(d, k, "l;ylfu ", "en")
    press(d, k, c.KEY_F, mods=SUPER_ALT)
    assert d.screen == "สวัสดี " and d.layout == "th"
    press(d, k, c.KEY_F, mods=SUPER_ALT)      # pressing again undoes a wrong conversion
    assert d.screen == "l;ylfu " and d.layout == "en"


def test_manual_fix_mid_word():
    d, k = setup()
    type_meaning(d, k, "l;y", "en")
    press(d, k, c.KEY_F, mods=SUPER_ALT)
    type_meaning(d, k, "สดี", "th")
    assert d.screen == "สวัสดี"


def test_backspace_over_space_returns_to_previous_word():
    d, k = setup({"l;ylfu": 0.97})
    k.th.tau_space = 0.99                     # Space does not fire
    type_meaning(d, k, "l;ylfu ", "en")
    press(d, k, c.KEY_BACKSPACE)
    k.th.tau_space = 0.96
    press(d, k, c.KEY_SPACE)                  # back in the word: Space checks it again
    assert d.screen == "สวัสดี "


def test_mouse_click_forgets_the_word():
    d, k = setup({"l;ylfu": 0.97})
    type_meaning(d, k, "l;ylfu", "en")
    k.on_click()
    press(d, k, c.KEY_SPACE)
    assert d.ops == []


def test_shortcut_and_enter_reset():
    d, k = setup({"l;ylfu": 0.97})
    type_meaning(d, k, "l;yl", "en")
    press(d, k, char_to_key("a", "en")[0], mods=(c.KEY_LEFTCTRL,))  # Ctrl+A
    type_meaning(d, k, "fu", "en")
    press(d, k, c.KEY_SPACE)
    assert d.ops == []                        # only "fu" was in the buffer


def test_user_layout_keys_update_tracking():
    d, k = setup()
    press(d, k, c.KEY_SPACE, mods=(c.KEY_LEFTMETA,))
    assert k.layout == "th"
    press(d, k, c.KEY_1, mods=SUPER_ALT)
    assert k.layout == "en"
    press(d, k, c.KEY_2, mods=SUPER_ALT)
    assert k.layout == "th"


def test_pause_stops_automatic_fixes():
    d, k = setup({"l;y": 0.9995})
    press(d, k, c.KEY_P, mods=SUPER_ALT)
    type_meaning(d, k, "l;ylfu ", "en")
    assert d.ops == []


def test_caps_lock_disables_fixing():
    d, k = setup({"L;Y": 0.9995, "l;y": 0.9995})
    k.caps_lock = True
    type_meaning(d, k, "l;y", "en")
    assert d.ops == []


def test_dry_run_reports_but_never_touches_text():
    d, k = setup({"l;y": 0.9995})
    k.dry_run = True
    type_meaning(d, k, "l;ylfu ", "en")
    assert d.screen == "l;ylfu " and d.ops == []
    assert [e.kind for e in k.log].count("would-fix") == 1


# --- found on the real desktop: typing "hello" with th active gave "lheo" (fix ran while keys were held)

def test_fix_waits_for_release_and_includes_keys_typed_meanwhile():
    d, k = setup({"้ำส": 0.9999}, desktop_layout="th")
    k.layout = "th"
    type_meaning(d, k, "hell", "en", flush=False)   # fast typing: never a moment with no key held
    assert d.ops == []                              # decided at "hel" but nothing injected yet
    k.flush()
    type_meaning(d, k, "o", "en")
    assert d.screen == "hello"
    assert d.ops == [("backspace", 4), ("layout", "en"), ("keys", 4)]


def test_manual_fix_not_injected_while_hotkey_held():
    d, k = setup()
    type_meaning(d, k, "l;ylfu ", "en")
    for m in SUPER_ALT:
        k.on_key(m, 1)
    k.on_key(c.KEY_F, 1)
    assert d.ops == []          # Super+Alt still down: injecting now would send shortcuts
    k.on_key(c.KEY_F, 0)
    for m in SUPER_ALT:
        k.on_key(m, 0)
    k.flush()
    assert d.screen == "สวัสดี "


def test_space_fix_dropped_when_next_word_started_before_flush():
    d, k = setup({"l;ylfu": 0.97})
    type_meaning(d, k, "l;ylfu", "en")
    press(d, k, c.KEY_SPACE, flush=False)
    type_meaning(d, k, "a", "en", flush=False)      # next word already on screen: deleting is unsafe
    k.flush()
    assert d.ops == [] and d.screen == "l;ylfu a"


def test_backspace_cancels_pending_fix():
    d, k = setup({"l;y": 0.9995})
    type_meaning(d, k, "l;y", "en", flush=False)
    press(d, k, c.KEY_BACKSPACE, flush=False)
    k.flush()
    assert d.ops == []


# --- found on the real desktop: screen showed "l;l;" while the daemon thought Thai was active

def test_desktop_reported_layout_wins_over_counting():
    d, k = setup()
    k.layout = "th"                             # our guess drifted
    k.on_layout("en")                           # Cinnamon says what is really active
    assert k.layout == "en"
    type_meaning(d, k, "l;l;", "en")
    assert [e.detail for e in k.log if e.kind == "score"][-1] == "l;l;"


def test_held_shortcut_autorepeat_is_not_counted_again():
    d, k = setup()
    k.on_key(c.KEY_LEFTMETA, 1)
    k.on_key(c.KEY_SPACE, 1)
    for _ in range(5):
        k.on_key(c.KEY_SPACE, 2)                # Super+Space held down
    k.on_key(c.KEY_SPACE, 0)
    k.on_key(c.KEY_LEFTMETA, 0)
    assert k.layout == "th"
    press(d, k, c.KEY_P, mods=SUPER_ALT)
    for _ in range(3):
        k.on_key(c.KEY_P, 2)
    assert not k.enabled


# --- found on the real desktop: "did u see" typed with th active; "u" (ี) was never fixed

@pytest.mark.parametrize("word", ["u", "the", "he", "but"])
def test_english_word_without_thai_consonant_is_fixed(word):
    d, k = setup(desktop_layout="th")   # model abstains (0.0) on all of them
    k.layout = "th"
    type_meaning(d, k, "ดี ", "th")     # any earlier word: we saw a Space, so a new word starts
    type_meaning(d, k, word, "en")
    assert d.screen.endswith(" " + word) and d.layout == "en"


def test_spelling_rule_not_used_after_click_into_a_word():
    d, k = setup(desktop_layout="th")
    k.layout = "th"
    type_meaning(d, k, "ดี ", "th")
    k.on_click()                        # cursor may now be right after a Thai consonant
    type_meaning(d, k, "ี", "th")
    assert d.ops == []


def test_spelling_rule_not_used_mid_word_after_layout_toggle():
    d, k = setup(desktop_layout="th")
    k.layout = "th"
    type_meaning(d, k, "x ", "en")
    type_meaning(d, k, "ม", "th")
    press(d, k, c.KEY_SPACE, mods=(c.KEY_LEFTMETA,))
    press(d, k, c.KEY_SPACE, mods=(c.KEY_LEFTMETA,))   # toggled away and back: "ม" still on screen
    type_meaning(d, k, "ี", "th")
    assert d.ops == []


def test_three_marks_in_a_row_fixed_even_after_click():
    d, k = setup(desktop_layout="th")   # "the" typed right after clicking into a text box
    k.layout = "th"
    k.on_click()
    type_meaning(d, k, "the", "en")
    assert d.screen == "the" and d.layout == "en"
