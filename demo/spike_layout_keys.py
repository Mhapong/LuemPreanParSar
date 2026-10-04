"""Spike: do Cinnamon's "switch to input source N" shortcuts select a layout absolutely?

Run in a terminal, then don't touch the keyboard or mouse for ~10 s:
    uv run --extra demo python demo/spike_layout_keys.py

Needs the shortcuts bound first (once):
    gsettings set org.cinnamon.desktop.keybindings.wm switch-input-source-0 "['<Super><Alt>1']"
    gsettings set org.cinnamon.desktop.keybindings.wm switch-input-source-1 "['<Super><Alt>2']"

How it reads the real layout: after each shortcut it injects KEY_F through uinput into this
(focused) terminal and reads what arrives on stdin: "f" means en is active, "ด" means th.
Throwaway code.
"""

import os
import select
import sys
import termios
import time
import tty

from evdev import UInput, ecodes as e

PROBE_TIMEOUT = 1.0


def press(ui: UInput, *keys: int) -> None:
    """Press keys in order (modifiers first), release in reverse."""
    for k in keys:
        ui.write(e.EV_KEY, k, 1)
        ui.syn()
        time.sleep(0.02)
    for k in reversed(keys):
        ui.write(e.EV_KEY, k, 0)
        ui.syn()
        time.sleep(0.02)
    time.sleep(0.3)  # let the compositor apply a layout change


def drain() -> None:
    while select.select([sys.stdin], [], [], 0)[0]:
        os.read(sys.stdin.fileno(), 1024)


def probe(ui: UInput) -> str:
    """Type KEY_F into this terminal and see which character arrives."""
    drain()
    press(ui, e.KEY_F)
    deadline = time.time() + PROBE_TIMEOUT
    buf = b""
    while time.time() < deadline:
        if select.select([sys.stdin], [], [], 0.05)[0]:
            buf += os.read(sys.stdin.fileno(), 16)
            try:
                text = buf.decode("utf-8")
            except UnicodeDecodeError:
                continue  # "ด" is 3 bytes; wait for the rest
            return {"f": "en", "ด": "th"}.get(text, f"?{text!r}")
    return "none (no key arrived: is this terminal focused?)"


STEPS = [
    ("Super+Alt+1", (e.KEY_LEFTMETA, e.KEY_LEFTALT, e.KEY_1), "en"),
    ("Super+Alt+2", (e.KEY_LEFTMETA, e.KEY_LEFTALT, e.KEY_2), "th"),
    ("Super+Alt+2 again", (e.KEY_LEFTMETA, e.KEY_LEFTALT, e.KEY_2), "th"),
    ("Super+Alt+1", (e.KEY_LEFTMETA, e.KEY_LEFTALT, e.KEY_1), "en"),
    ("Super+Alt+1 again", (e.KEY_LEFTMETA, e.KEY_LEFTALT, e.KEY_1), "en"),
    ("Super+Space (toggle, for comparison)", (e.KEY_LEFTMETA, e.KEY_SPACE), "th"),
]


def main() -> None:
    if not sys.stdin.isatty():
        raise SystemExit("run this in a terminal")
    print(f"session={os.environ.get('XDG_SESSION_TYPE')}. Hands off the keyboard for ~10 s...")
    old = termios.tcgetattr(sys.stdin)
    ui = UInput(name="luem-spike-layout")
    results = []
    try:
        tty.setcbreak(sys.stdin.fileno())  # read keys one by one, no echo
        time.sleep(1.0)  # give the compositor time to register the new virtual keyboard
        results.append(("start", probe(ui), None))
        for label, keys, expect in STEPS:
            press(ui, *keys)
            results.append((label, probe(ui), expect))
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        ui.close()

    print(f"\n{'step':40} {'layout after':14} expected")
    for label, got, expect in results:
        mark = "" if expect is None else ("OK" if got == expect else "MISMATCH")
        print(f"{label:40} {got:14} {expect or '':8} {mark}")
    absolute = all(got == expect for _, got, expect in results[1:6])
    print("\nRESULT:", "shortcuts select a layout absolutely" if absolute
          else "shortcuts do NOT select absolutely (see MISMATCH rows)")


if __name__ == "__main__":
    main()
