"""LuemPreanParSar daemon: fixes words typed in the wrong keyboard layout, on X11 and Wayland.

Usage:
    uv run --extra demo python demo/daemon.py                 # fix for real (CNN, target precision 99.9%)
    uv run --extra demo python demo/daemon.py --dry-run -v    # only show what it would fix
    uv run --extra demo python demo/daemon.py --setup         # bind Super+Alt+1/2 first, then run

Hotkeys:  Super+Alt+F  convert the current / last word (press again to undo)
          Super+Alt+P  pause / resume automatic fixing
          Super+Alt+1 / Super+Alt+2  switch to English / Thai (the daemon follows)

How it works: reads key codes from /dev/input (evdev, below the display server, so Wayland does
not block it), asks the model about the word being typed, and when it is in the wrong layout
types BackSpace, selects the other layout and presses the same keys again through a virtual
keyboard (uinput). Logic lives in src/luem/corrector.py.

Key codes do not say which layout is active, so the daemon asks Cinnamon over D-Bus (works on
X11 and Wayland) and follows its CurrentInputSourceChanged signal; layouts are also selected
through D-Bus. Without Cinnamon it falls back to injecting Super+Alt+1/2 and counting Super+Space.

Privacy: nothing typed is written to disk. With -v the current word is printed to this terminal.
Needs read access to /dev/input/event* and /dev/uinput (group `input`, see docs/IMPLEMENTATION_PLAN.md).
"""

import argparse
import json
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

from evdev import InputDevice, UInput, ecodes as e, list_devices

from luem.corrector import KEY_1, KEY_2, Corrector, Thresholds
from luem.models import NAMES, load_model

ROOT = Path(__file__).resolve().parent.parent
UINPUT_NAME = "luem-daemon"
OWN_DEVICES = {UINPUT_NAME, "luem-spike", "luem-spike-layout"}
KB_SCHEMA, SRC_SCHEMA = "org.cinnamon.desktop.keybindings.wm", "org.cinnamon.desktop.input-sources"
SHORTCUTS = {"switch-input-source-0": "['<Super><Alt>1']", "switch-input-source-1": "['<Super><Alt>2']"}
CLICKS = {e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE, e.BTN_TOUCH}  # BTN_TOUCH: touchpad tap/move
CINNAMON = ["--session", "--dest", "org.Cinnamon", "--object-path", "/org/Cinnamon"]
SOURCE_LAYOUT = {"us": "en", "th": "th"}  # Cinnamon input-source id -> our layout name
LAYOUT_INDEX = {"en": 0, "th": 1}         # position in the sources list (checked by check_desktop)


# ------------------------------------------------------------------ desktop checks

def gsettings(*args: str) -> str | None:
    try:
        return subprocess.run(["gsettings", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_desktop(setup: bool) -> bool:
    """Layouts must be [us, th] and Super+Alt+1/2 bound to them; --setup binds the shortcuts."""
    sources = gsettings("get", SRC_SCHEMA, "sources")
    if sources is None:
        print("! gsettings/Cinnamon not found: make sure Super+Alt+1 selects English and Super+Alt+2 Thai")
        return True
    if not sources.startswith("[('xkb', 'us'), ('xkb', 'th')"):
        print(f"! input sources are {sources}; expected English (us) first and Thai (th) second")
        return False
    ok = True
    for key, want in SHORTCUTS.items():
        have = gsettings("get", KB_SCHEMA, key)
        if have == want:
            continue
        if setup:
            gsettings("set", KB_SCHEMA, key, want)
            print(f"  bound {key} = {want}")
        else:
            print(f"! {key} is {have}, expected {want}. Run once with --setup (or bind it in Keyboard settings).")
            ok = False
    return ok


def load_thresholds(model: str, target: str, args) -> Thresholds:
    """Thresholds chosen on val by eval/evaluate.py; command-line values override them."""
    path = ROOT / "eval" / "results" / f"{model}.json"
    th = Thresholds()
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))["targets"][target]["test"]["thresholds"]
        th = Thresholds(saved["k_min"], saved["tau_type"], saved["tau_space"])
    except (OSError, KeyError) as err:
        print(f"! no thresholds for {model}@{target} in {path} ({err!r}); using defaults")
    for name in ("k_min", "tau_type", "tau_space"):
        if getattr(args, name) is not None:
            setattr(th, name, getattr(args, name))
    return th


class CinnamonLayout:
    """The real active layout, from Cinnamon's own state over D-Bus (not gsettings, which lies).

    Counting Super+Space presses drifted (seen on the real desktop: screen showed "l;l;" while the
    daemon thought Thai). Cinnamon knows the truth and signals every change, whoever made it.
    """

    def __init__(self):
        self.proc = subprocess.Popen(["gdbus", "monitor", *CINNAMON], stdout=subprocess.PIPE)
        self.fd = self.proc.stdout.fileno()
        self.buf = ""

    @staticmethod
    def call(method: str, *args: str) -> str | None:
        try:
            return subprocess.run(["gdbus", "call", *CINNAMON, "--method", f"org.Cinnamon.{method}", *args],
                                  capture_output=True, text=True, check=True, timeout=2).stdout
        except (OSError, subprocess.SubprocessError):
            return None

    @classmethod
    def current(cls) -> str | None:
        """GetInputSources returns tuples (type, id, ..., is_current); pick the current one."""
        out = cls.call("GetInputSources")
        found = re.search(r"\('xkb', '(\w+)'[^()]*, true\)", out or "")
        return SOURCE_LAYOUT.get(found.group(1)) if found else None

    def select(self, layout: str) -> None:
        self.call("ActivateInputSourceIndex", str(LAYOUT_INDEX[layout]))

    def changes(self) -> list[str]:
        """Layouts announced since the last call. Raw os.read: select() can't see Python-buffered lines."""
        data = os.read(self.fd, 4096)
        if not data:
            raise EOFError("gdbus monitor exited")
        self.buf += data.decode(errors="replace")
        *lines, self.buf = self.buf.split("\n")
        ids = re.findall(r"CurrentInputSourceChanged \('(\w+)',\)", "\n".join(lines))
        return [SOURCE_LAYOUT[i] for i in ids if i in SOURCE_LAYOUT]

    def close(self) -> None:
        self.proc.terminate()


# ------------------------------------------------------------------ devices

def find_devices() -> tuple[list[InputDevice], list[InputDevice]]:
    keyboards, pointers = [], []
    for path in list_devices():
        dev = InputDevice(path)
        if dev.name in OWN_DEVICES:  # never read our own injected keys back
            continue
        keys = set(dev.capabilities().get(e.EV_KEY, []))
        if e.KEY_A in keys and e.KEY_SPACE in keys:
            keyboards.append(dev)
        if keys & CLICKS:
            pointers.append(dev)
    return keyboards, pointers


class UinputInjector:
    """Implements corrector.Injector with a virtual keyboard."""

    def __init__(self, ui: UInput, key_delay: float, switch_delay: float, cinnamon: CinnamonLayout | None):
        self.ui, self.key_delay, self.switch_delay, self.cinnamon = ui, key_delay, switch_delay, cinnamon

    def _key(self, code: int, value: int) -> None:
        self.ui.write(e.EV_KEY, code, value)
        self.ui.syn()
        time.sleep(self.key_delay)

    def _tap(self, code: int, mods: tuple[int, ...] = ()) -> None:
        """Press and release like a person: each change in its own report, with time in between.

        Press and release in one report (zero-length key press) worked in xed but Brave's address
        bar dropped some of them: the BackSpaces deleted only part of the word, the retyped keys
        never appeared.
        """
        for m in mods:
            self._key(m, 1)
        self._key(code, 1)
        self._key(code, 0)
        for m in reversed(mods):
            self._key(m, 0)

    def backspace(self, n: int) -> None:
        for _ in range(n):
            self._tap(e.KEY_BACKSPACE)

    def select_layout(self, layout: str) -> None:
        if self.cinnamon:
            self.cinnamon.select(layout)
        else:
            self._tap(KEY_1 if layout == "en" else KEY_2, (e.KEY_LEFTMETA, e.KEY_LEFTALT))
        time.sleep(self.switch_delay)  # retyping before the switch lands would use the old layout

    def press_keys(self, keys) -> None:
        for code, shift in keys:
            self._tap(code, (e.KEY_LEFTSHIFT,) if shift else ())


def keys_down(keyboards: list[InputDevice]) -> bool:
    """Ask the kernel (not our own bookkeeping) whether any key is physically down right now."""
    return any(d.active_keys() for d in keyboards)


def fix_with_keyboards_grabbed(k: Corrector, keyboards: list[InputDevice], ui: UInput,
                               timeout: float = 2.0) -> None:
    """Run the pending fix while the physical keyboards are grabbed (EVIOCGRAB).

    Without this, a key typed during the fix reached the app in the middle of it ("hello" became
    "lheo"). While grabbed, the user's keys reach only us; we replay them through the virtual
    keyboard *after* the fix. Call only when no key is down: a key held while grabbing would never
    deliver its release to the desktop and stay stuck.
    """
    grabbed = []
    for d in keyboards:
        try:
            d.grab()
            grabbed.append(d)
        except OSError:
            pass
    down = set()  # replayed keys whose release we have not replayed yet
    try:
        k.flush()
        deadline = time.monotonic() + timeout
        while True:
            ready, _, _ = select.select(grabbed, [], [], 0.05 if down else 0)
            got = False
            for dev in ready:
                try:
                    events = list(dev.read())
                except BlockingIOError:
                    continue
                for ev in events:
                    if ev.type != e.EV_KEY or ev.value == 2 or ev.code in CLICKS:
                        continue  # autorepeat is generated by the desktop itself
                    ui.write(e.EV_KEY, ev.code, ev.value)
                    ui.syn()
                    (down.add if ev.value else down.discard)(ev.code)
                    k.on_key(ev.code, ev.value)  # keep the corrector in step with the screen
                    got = True
            # keep forwarding until every replayed key is released again, then hand the keyboard back
            if (not down and not got) or time.monotonic() > deadline:
                break
    finally:
        for code in down:  # timed out with a key down: release it rather than leave it stuck
            ui.write(e.EV_KEY, code, 0)
        ui.syn()
        for d in grabbed:
            try:
                d.ungrab()
            except OSError:
                pass


# ------------------------------------------------------------------ main loop

def show(events, verbose: bool) -> None:
    for ev in events:
        if ev.kind == "score" and not verbose:
            continue
        p = f" p={ev.p:.4f}" if ev.p is not None else ""
        mark = {"fix": "FIX ", "would-fix": "WOULD FIX ", "score": "  ", "layout": "layout: ",
                "reset": "  reset: ", "pause": ""}[ev.kind]
        print(f"{time.strftime('%H:%M:%S')} {mark}{ev.detail}{p}", flush=True)
    events.clear()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=NAMES, default="cnn")
    p.add_argument("--target", default="0.999", help="precision target whose val thresholds to use (0.99 or 0.999)")
    p.add_argument("--k-min", type=int)
    p.add_argument("--tau-type", type=float)
    p.add_argument("--tau-space", type=float)
    p.add_argument("--dry-run", action="store_true", help="only print what would be fixed")
    p.add_argument("--setup", action="store_true", help="bind Super+Alt+1/2 to English/Thai via gsettings")
    p.add_argument("-v", "--verbose", action="store_true", help="print the model score after every key")
    p.add_argument("--key-delay", type=float, default=0.008,
                   help="seconds between injected key presses and releases (raise it if an app drops keys)")
    p.add_argument("--switch-delay", type=float, default=0.15)
    args = p.parse_args()

    if not check_desktop(args.setup):
        raise SystemExit(1)
    th = load_thresholds(args.model, args.target, args)
    t0 = time.perf_counter()
    model = load_model(args.model)
    print(f"model {args.model} loaded in {time.perf_counter() - t0:.2f}s; "
          f"k_min={th.k_min} tau_type={th.tau_type:g} tau_space={th.tau_space:g}")

    keyboards, pointers = find_devices()
    if not keyboards:
        raise SystemExit("no readable keyboard: is your user in group `input` (log out/in after adding)?")
    for d in keyboards:
        print(f"keyboard: {d.path} {d.name}")
    try:
        ui = UInput(name=UINPUT_NAME)
    except PermissionError:
        raise SystemExit("cannot open /dev/uinput: add the udev rule from docs/IMPLEMENTATION_PLAN.md")

    real = CinnamonLayout.current()
    cinnamon = CinnamonLayout() if real else None
    print(f"active layout: {real} (followed via Cinnamon D-Bus)" if real else
          "! cannot read the layout from Cinnamon D-Bus: guessing it by counting Super+Space")
    k = Corrector(model, UinputInjector(ui, args.key_delay, args.switch_delay, cinnamon), th,
                  dry_run=args.dry_run, layout=real or "en")
    devices = {d.fd: d for d in keyboards + pointers}
    kb_fds = {d.fd for d in keyboards}
    try:
        time.sleep(0.5)  # let the compositor register the virtual keyboard
        if not args.dry_run and not cinnamon:
            k.start()
        mode = "fixing" if not args.dry_run else "DRY RUN (no changes)" if cinnamon else \
            "DRY RUN (no changes; assumes English is active, press Super+Alt+1 to be sure)"
        print(f"\nready, {mode}. Super+Alt+F convert word, Super+Alt+P pause, Ctrl+C quit\n")
        show(k.log, args.verbose)
        while True:
            ready, _, _ = select.select([*devices, *([cinnamon.fd] if cinnamon else [])], [], [])
            for fd in ready:
                if cinnamon and fd == cinnamon.fd:
                    try:
                        for layout in cinnamon.changes():
                            k.on_layout(layout)
                    except EOFError:
                        print("! lost the Cinnamon D-Bus monitor: guessing the layout by counting Super+Space")
                        cinnamon.close()
                        cinnamon = k.injector.cinnamon = None
                    show(k.log, args.verbose)
                    continue
                dev = devices[fd]
                try:
                    events = list(dev.read())
                except OSError:  # device unplugged
                    devices.pop(fd, None)
                    continue
                for ev in events:
                    if ev.type != e.EV_KEY:
                        continue
                    if fd in kb_fds and ev.code not in CLICKS:
                        if ev.value == 1:
                            k.caps_lock = e.LED_CAPSL in dev.leds()
                        k.on_key(ev.code, ev.value)
                    elif ev.code in CLICKS and ev.value == 1:
                        k.on_click()
                # a decided fix runs only once every key is up (see corrector.py docstring)
                if k.pending and not keys_down(keyboards):
                    if args.dry_run:
                        k.flush()
                    else:
                        fix_with_keyboards_grabbed(k, keyboards, ui)
                show(k.log, args.verbose)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        ui.close()
        if cinnamon:
            cinnamon.close()


if __name__ == "__main__":
    main()
