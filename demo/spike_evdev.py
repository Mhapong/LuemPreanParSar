import os
import select
import subprocess
import time

from evdev import InputDevice, UInput, ecodes, list_devices

from luem.layout import char_to_key, key_to_char

SCHEMA = "org.cinnamon.desktop.input-sources"
LAYOUTS = ["en", "th"]
UINPUT_NAME = "luem-spike"


def get_layout() -> str:
    out = subprocess.run(["gsettings", "get", SCHEMA, "current"], capture_output=True, text=True).stdout
    return LAYOUTS[int(out.split()[-1])]


def set_layout(layout: str) -> None:
    subprocess.run(["gsettings", "set", SCHEMA, "current", str(LAYOUTS.index(layout))], check=True)


def find_keyboards() -> list[InputDevice]:
    kbds = []
    for path in list_devices():
        dev = InputDevice(path)
        keys = dev.capabilities().get(ecodes.EV_KEY, [])
        if dev.name != UINPUT_NAME and ecodes.KEY_A in keys and ecodes.KEY_SPACE in keys:
            kbds.append(dev)
    return kbds


def tap(ui: UInput, code: int, shift: bool = False) -> None:
    if shift:
        ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTSHIFT, 1)
    ui.write(ecodes.EV_KEY, code, 1)
    ui.write(ecodes.EV_KEY, code, 0)
    if shift:
        ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTSHIFT, 0)
    ui.syn()
    time.sleep(0.005)


def inject_test(ui: UInput) -> None:
    for c in "hello":
        tap(ui, *char_to_key(c, "en"))
    for _ in range(2):
        tap(ui, ecodes.KEY_BACKSPACE)


def super_space(ui: UInput) -> None:
    ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 1)
    tap(ui, ecodes.KEY_SPACE)
    ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 0)
    ui.syn()


def main() -> None:
    print(f"session={os.environ.get('XDG_SESSION_TYPE')}  groups ok={'input' in os.popen('id -nG').read()}")
    kbds = find_keyboards()
    if not kbds:
        raise SystemExit("No readable keyboard. In `input` group? Re-logged in?  ls -l /dev/input/event*")
    for d in kbds:
        print(f"reading: {d.path}  {d.name}")
    ui = UInput(name=UINPUT_NAME)
    print(f"uinput ok: {ui.device.path}\nlayout (gsettings) = {get_layout()}\n")

    shift_down = set()
    meta_down = set()
    tracked = get_layout()
    fds = {d.fd: d for d in kbds}
    try:
        while True:
            r, _, _ = select.select(fds, [], [])
            for fd in r:
                for ev in fds[fd].read():
                    if ev.type != ecodes.EV_KEY:
                        continue
                    if ev.code in (ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT):
                        (shift_down.add if ev.value else shift_down.discard)(ev.code)
                        continue
                    if ev.code in (ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA):
                        (meta_down.add if ev.value else meta_down.discard)(ev.code)
                        continue
                    if ev.value != 1:
                        continue
                    if ev.code == ecodes.KEY_SPACE and meta_down:
                        tracked = "th" if tracked == "en" else "en"
                        print(f">> Super+Space seen: tracked -> {tracked}, gsettings = {get_layout()}")
                        continue
                    if ev.code == ecodes.KEY_F8:
                        print(">> F8: injecting 'hello' + 2x BackSpace")
                        inject_test(ui)
                    elif ev.code == ecodes.KEY_F9:
                        new = "th" if get_layout() == "en" else "en"
                        set_layout(new)
                        print(f">> F9: gsettings set -> {new}, readback = {get_layout()}")
                    elif ev.code == ecodes.KEY_F10:
                        super_space(ui)
                        tracked = "th" if tracked == "en" else "en"
                        time.sleep(0.1)
                        print(f">> F10: sent Super+Space, tracked -> {tracked}, gsettings = {get_layout()}")
                    else:
                        shift = bool(shift_down)
                        name = ecodes.KEY.get(ev.code, ev.code)
                        print(f"{ev.code:3} {str(name):16} shift={int(shift)}  "
                              f"en={key_to_char(ev.code, shift, 'en')!r:6} th={key_to_char(ev.code, shift, 'th')!r:6} "
                              f"tracked={tracked} gsettings={get_layout()}")
    except KeyboardInterrupt:
        pass
    finally:
        ui.close()


if __name__ == "__main__":
    main()
