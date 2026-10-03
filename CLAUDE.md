# CLAUDE.md

Course project (PSU): detect text typed in the wrong keyboard layout (Thai Kedmanee <-> US QWERTY) and fix it on Linux. Deadline 2026-10-08. Plan and daily checklist: docs/IMPLEMENTATION_PLAN.md (Thai). Flow diagrams: docs/FLOW.md. Report figure checklist: docs/FIGURES.md. Work log with rationale and problems: docs/WORKLOG.md. Glossary: docs/GLOSSARY.md.

Docs (README.md, docs/*) are Thai and must be readable by a non-technical reader: explain terms on first use or link docs/GLOSSARY.md, use concrete examples, put technical detail inside `<details>` blocks. This file (CLAUDE.md) stays technical English.

## Commands
- `uv sync` (dev), `uv sync --extra data --extra ml` (dataset/training phase), `--extra demo` for the evdev/uinput daemon
- `uv run pytest` runs tests and doctests in `src/`
- `uv run python scripts/verify_layout.py` checks `src/luem/layout.py` against `/usr/share/X11/xkb/symbols/th`
- Pipeline: `scripts/download_corpus.py` -> `scripts/build_dataset.py` -> `scripts/train_baselines.py` -> `eval/evaluate.py --model ngram` (thresholds chosen on val, reported on test; `--limit 30000` default)
- Models live in `src/luem/models/` and are loaded by name via `luem.models.load_model`; add new ones to `NAMES`.

## Conventions
- Layout mapping in `src/luem/layout.py` is the single source of truth; never duplicate the table elsewhere.
- Every model exposes `predict(text: str) -> float` = P(typed in the wrong layout), so eval and demo can swap models.
- Problem is binary: the script of the typed text reveals the active layout, so the model only decides ok vs wrong.
- Main neural model is a char-CNN (GRU optional, for comparison). Dataset stores full segments, not prefixes: the CNN DataLoader samples a random prefix each epoch; `evaluate.py` expands prefixes 1..n itself.
- Detection runs at two moments with the same model: while typing (`τ_type`, high, after `k_min` chars) and on Space (`τ_space`, lower, full word). On-Space correction must also delete and retype the Space, since the daemon reads keys passively.
- `data/` and `models/` are regenerable and not committed. Never put personal shell history or keystroke logs into committed data.

## Environment
- Dev/demo: notebook, Linux Mint 22.3 Cinnamon 6.6.9, target session **Wayland** (X11 as fallback), layouts `us,th,us`, IBus running. Latency is measured here (CPU).
- Demo reads keys via evdev and injects via uinput (no X11 APIs), so it runs on both Wayland and X11. evdev gives keycodes, not characters: the daemon must track the active layout itself. Spike result (X11): `gsettings org.cinnamon.desktop.input-sources current` neither follows nor controls the real layout; injecting Super+Space via uinput does switch it. So: switch by injecting Super+Space, track state by counting Super+Space presses.
- Training: separate **Windows** PC with RTX 3070 (8 GB). Neural training scripts must run on both CPU and CUDA, on Windows and Linux: always `open(..., encoding="utf-8")`, use `pathlib`, guard entry points with `if __name__ == "__main__":`. Models move to the notebook as ONNX.
