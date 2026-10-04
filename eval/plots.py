"""Figures and the comparison table for the report (docs/FIGURES.md numbers), from eval/results only.

Usage:
    uv run python eval/plots.py                  # main operating point: target 0.999
    uv run python eval/plots.py --target 0.99

Needs eval/evaluate.py (results + scores/) and, for the latency figure, eval/latency.py
(the Linux notebook's numbers are preferred over the training PC's).
Writes docs/figures/fig14_training.png, fig16_pr.png, fig17_chars.png, fig19_latency.png,
fig20_confusion.png and eval/results/summary.md (figure 18, Thai table to paste into the report).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
FIGURES = ROOT / "docs" / "figures"
MODELS = ("dictionary", "ngram", "cnn", "gru")
LABELS = {"dictionary": "พจนานุกรม", "ngram": "สถิติกลุ่มตัวอักษร (n-gram)", "cnn": "CNN", "gru": "GRU"}
COLORS = {"dictionary": "#9e9e9e", "ngram": "#ef8a17", "cnn": "#1f6fb2", "gru": "#4caf50"}
RUNS = {"cnn_a_base": "CNN (ใช้จริง)", "cnn_b_hard3": "CNN น้ำหนักตัวอย่างหลอก x3",
        "cnn_c_wide": "CNN ขนาดใหญ่ขึ้น", "gru": "GRU"}
THAI_FONTS = ("Leelawadee UI", "Tahoma", "Noto Sans Thai", "Noto Sans Thai Looped", "Loma", "Garuda", "Sarabun")


def setup_fonts() -> None:
    """Latin first, then the first installed Thai font as a per-glyph fallback (matplotlib >= 3.6)."""
    installed = {f.name for f in fm.fontManager.ttflist}
    thai = [f for f in THAI_FONTS if f in installed]
    if not thai:
        print("warning: no Thai font found, Thai labels will show as boxes (Linux: apt install fonts-noto-core)")
    plt.rcParams.update({"font.family": ["DejaVu Sans", *thai[:1]], "figure.dpi": 150, "savefig.bbox": "tight", "figure.constrained_layout.use": True,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.3})


def load_results() -> dict:
    out = {}
    for m in MODELS:
        path = RESULTS / f"{m}.json"
        if path.exists():
            out[m] = json.loads(path.read_text(encoding="utf-8"))
    return out


def fig_training(path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for run, label in RUNS.items():
        f = RESULTS / "training" / f"{run}.json"
        if not f.exists():
            continue
        h = json.loads(f.read_text(encoding="utf-8"))["history"]
        ep = [e["epoch"] for e in h]
        style = {"label": label, "marker": "o", "ms": 3, "lw": 2.2 if run == "cnn_a_base" else 1.2}
        a.plot(ep, [e["loss_full"] for e in h], **style)
        b.plot(ep, [e["loss_prefix"] for e in h], **style)
    a.set(title="ทายจากคำเต็ม (ตอนกด Space)", xlabel="รอบการฝึก (epoch)", ylabel="loss บนข้อสอบย่อย (ยิ่งต่ำยิ่งดี)")
    b.set(title="ทายจากส่วนต้นของคำ (ระหว่างพิมพ์)", xlabel="รอบการฝึก (epoch)")
    a.set_yscale("log")
    a.legend(fontsize=8)
    fig.suptitle("รูปที่ 14: AI เก่งขึ้นทีละรอบ และหยุดดีขึ้นหลังรอบที่ 8")
    fig.savefig(path)
    plt.close(fig)


def fig_pr(results: dict, target: str, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, r in results.items():
        s = np.load(RESULTS / "scores" / f"{m}.npz")
        y, full = s["y"].astype(bool), s["full"]
        order = np.argsort(-full, kind="stable")
        tp = np.cumsum(y[order])
        prec, rec = tp / np.arange(1, len(y) + 1), tp / y.sum()
        for ax in (a, b):
            ax.plot(rec, prec, color=COLORS[m], label=LABELS[m], lw=1.8)
        t = r["targets"][target]["test"]
        if t["recall"] > 0:  # a model that never fixes anything has no operating point to mark
            for ax in (a, b):
                ax.plot(t["recall"], t["precision"], "o", color=COLORS[m], mec="black", ms=7, zorder=5)
    a.set(xlabel="จับได้กี่ % (recall)", ylabel="แก้ถูกกี่ % (precision)", title="ภาพรวม", ylim=(0.5, 1.005))
    b.set(xlabel="จับได้กี่ % (recall)", title="ขยายมุมขวาบน", xlim=(0.85, 1.002), ylim=(0.98, 1.0005))
    a.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"รูปที่ 16: แก้ถูกกี่ % เทียบกับจับได้กี่ % (ข้อสอบจริง)  ● = เกณฑ์ที่เลือก (เป้า {float(target):.1%})",
                 fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def fig_chars(results: dict, targets: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 4), sharey=True, squeeze=False)
    for ax, target in zip(axes[0], targets):
        for m, r in results.items():
            if target not in r["targets"]:
                continue
            cdf = r["targets"][target]["test"]["chars_to_detect"]["cdf"][:16]
            ax.plot(range(1, len(cdf) + 1), cdf, color=COLORS[m], label=LABELS[m], marker="o", ms=3, lw=1.8)
        ax.set(title=f"เป้าแก้ถูก {float(target):.1%}", xlabel="จำนวนตัวอักษรที่พิมพ์ไปแล้ว", xticks=range(1, 17, 1),
               ylim=(0, 1.0))
    axes[0][0].set_ylabel("% ของคำที่พิมพ์ผิดแป้น ที่จับได้แล้ว")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("รูปที่ 17: ต้องพิมพ์กี่ตัวถึงรู้ว่าพิมพ์ผิดแป้น (เฉพาะการแก้ระหว่างพิมพ์)")
    fig.savefig(path)
    plt.close(fig)


def fig_latency(path: Path) -> str | None:
    for os_name in ("linux", "win32", "darwin"):
        f = RESULTS / f"latency_{os_name}.json"
        if f.exists():
            break
    else:
        return None
    lat = json.loads(f.read_text(encoding="utf-8"))
    ms = [m for m in MODELS if m in lat["models"]]
    x = np.arange(len(ms))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - 0.2, [lat["models"][m]["p50_ms"] for m in ms], 0.4, label="ปกติ (p50)",
           color=[COLORS[m] for m in ms])
    ax.bar(x + 0.2, [lat["models"][m]["p99_ms"] for m in ms], 0.4, label="ช้าสุด 1% (p99)",
           color=[COLORS[m] for m in ms], alpha=0.45, hatch="//")
    ax.axhline(lat["budget_ms"], color="red", ls="--", lw=1)
    ax.text(len(ms) - 0.5, lat["budget_ms"], f" เป้า < {lat['budget_ms']:g} ms", color="red", va="bottom", ha="right")
    ax.set(xticks=x, xticklabels=[LABELS[m].split(" (")[0] for m in ms], ylabel="มิลลิวินาทีต่อการกด 1 ปุ่ม",
           yscale="log")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"รูปที่ 19: ความเร็ว (CPU 1 thread)\n{lat['machine']['cpu']}", fontsize=10)
    fig.savefig(path)
    plt.close(fig)
    return f.name


def fig_confusion(result: dict, target: str, path: Path) -> None:
    c = result["targets"][target]["test"]["confusion"]
    grid = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    share = grid / grid.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{grid[i, j]:,}\n({share[i, j]:.2%})", ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else "black")
    ax.set(xticks=[0, 1], xticklabels=["ไม่แก้", "แก้"], yticks=[0, 1],
           yticklabels=["พิมพ์ถูกอยู่แล้ว", "พิมพ์ผิดแป้น"], xlabel="โปรแกรมตัดสิน", ylabel="ความจริง")
    ax.set_title(f"รูปที่ 20: ทายถูก/ทายผิดของ {LABELS[result['model']]} (เป้า {float(target):.1%})", fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def summary_table(results: dict, targets: list[str], latency_file: str | None) -> str:
    lat = json.loads((RESULTS / latency_file).read_text(encoding="utf-8"))["models"] if latency_file else {}
    rows = ["| วิธี | เป้า | แก้ถูก (precision) | จับได้ (recall) | แก้ระหว่างพิมพ์ | แก้ตอนกด Space "
            "| แก้ผิดกับข้อความปกติ | แก้ผิดกับตัวอย่างหลอก | แก้ถูกในชีวิตจริง* | พิมพ์กี่ตัวถึงรู้ (มัธยฐาน) "
            "| ความเร็ว p50 (ms) | ขนาดไฟล์ |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for target in targets:
        for m, r in results.items():
            if target not in r["targets"]:
                continue
            t = r["targets"][target]["test"]
            size = r["file_bytes"] / 1e6
            speed = f"{lat[m]['p50_ms']:.3f}" if m in lat else "-"
            chars = t["chars_to_detect"]["median"]
            if t["recall"] == 0:
                rows.append(f"| {LABELS[m]} | {float(target):.1%} | ไม่มีเกณฑ์ที่ถึงเป้า (ไม่แก้เลย) "
                            + "| - " * 7 + f"| {speed} | {size:.2f} MB |")
                continue
            rows.append(f"| {LABELS[m]} | {float(target):.1%} | {t['precision']:.2%} | {t['recall']:.2%} "
                        f"| {t['recall_while_typing']:.1%} | {t['recall_on_space']:.1%} | {t['false_fix_rate']:.2%} "
                        f"| {t['false_fix_rate_hard_negatives']:.2%} | {t['precision_at_base_rate']['0.02']:.1%} "
                        f"| {chars if chars is not None else '-'} | {speed} | {size:.2f} MB |")
    note = ("\n\n\\* ถ้าในการพิมพ์จริงมีแค่ 2% ที่พิมพ์ผิดแป้น (ข้อสอบมีครึ่งต่อครึ่ง) "
            "แก้ถูก/จับได้/แก้ผิด วัดบนข้อสอบจริง เกณฑ์เลือกจากข้อสอบย่อย")
    if latency_file:
        note += f"; ความเร็วจาก `{latency_file}`"
    return "\n".join(rows) + note + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="0.999", help="operating point for figures 16 and 20")
    args = p.parse_args()
    setup_fonts()
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = load_results()
    targets = sorted({t for r in results.values() for t in r["targets"]}, key=float)

    fig_training(FIGURES / "fig14_training.png")
    fig_pr(results, args.target, FIGURES / "fig16_pr.png")
    fig_chars(results, targets, FIGURES / "fig17_chars.png")
    latency_file = fig_latency(FIGURES / "fig19_latency.png")
    fig_confusion(results["cnn"], args.target, FIGURES / "fig20_confusion.png")
    (RESULTS / "summary.md").write_text(summary_table(results, targets, latency_file), encoding="utf-8")
    print(f"figures -> {FIGURES}, table -> {RESULTS / 'summary.md'}"
          + ("" if latency_file else " (no latency results yet: run eval/latency.py)"))


if __name__ == "__main__":
    main()
