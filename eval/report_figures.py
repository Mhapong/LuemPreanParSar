import json
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch

from luem.layout import _EN_LOWER, _EN_UPPER, _TH_LOWER, _TH_UPPER
from plots import FIGURES, ROOT, setup_fonts

THOUSANDS = FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k" if v else "0")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_TINT, ORANGE_TINT = "#cde2fb", "#fde3d6"


def show(ch: str) -> str:
    return "◌" + ch if unicodedata.combining(ch) else ch


def fig_keyboard(path: Path) -> None:
    rows = [(0, 13), (13, 25), (25, 37), (37, 47)]
    offsets = [0.0, 1.5, 1.8, 2.3]
    example = set("l;yfu")
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.set_xlim(-0.2, 16.4)
    ax.set_ylim(-0.35, 4.25)
    ax.axis("off")
    ax.grid(False)
    for r, ((a, b), off) in enumerate(zip(rows, offsets)):
        y = 3 - r
        for i in range(a, b):
            x = off + (i - a)
            hot = _EN_LOWER[i] in example
            ax.add_patch(FancyBboxPatch((x + 0.05, y + 0.05), 0.9, 0.9, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc=BLUE_TINT if hot else "white", ec=BLUE if hot else "#b5b4af", lw=1.2))
            kw = {"fontsize": 9}
            ax.text(x + 0.17, y + 0.72, _EN_UPPER[i], color=MUTED, ha="left", va="center", **kw)
            ax.text(x + 0.17, y + 0.30, _EN_LOWER[i], color=INK, ha="left", va="center", fontsize=12,
                    fontweight="bold")
            ax.text(x + 0.83, y + 0.72, show(_TH_UPPER[i]), color=MUTED, ha="right", va="center", **kw)
            ax.text(x + 0.83, y + 0.30, show(_TH_LOWER[i]), color=BLUE, ha="right", va="center", fontsize=13)
    lx, ly, lw = 13.2, 0.05, 3.0
    ax.add_patch(FancyBboxPatch((lx, ly), lw, 0.9, boxstyle="round,pad=0,rounding_size=0.08", fc="white",
                                ec="#b5b4af", lw=1.2))
    for (dx, dy, txt, col, ha) in ((0.12, 0.67, "อังกฤษ + Shift", MUTED, "left"),
                                   (0.12, 0.25, "อังกฤษ", INK, "left"),
                                   (lw - 0.12, 0.67, "ไทย + Shift", MUTED, "right"),
                                   (lw - 0.12, 0.25, "ไทย", BLUE, "right")):
        ax.text(lx + dx, ly + dy, txt, color=col, ha=ha, va="center", fontsize=9)
    ax.text(lx + lw, ly + 1.05, "อ่านปุ่มอย่างไร", ha="right", fontsize=9, color=MUTED)
    ax.set_title("แป้นพิมพ์อังกฤษ (US) กับแป้นไทย (เกษมณี) ปุ่มเดียวกันให้ตัวอักษรต่างกัน\n"
                 "ปุ่มสีฟ้า = ปุ่มที่กดเมื่อพิมพ์ \"สวัสดี\" ถ้าลืมเปลี่ยนภาษาจะได้ l;ylfu",
                 fontsize=11, loc="left")
    fig.savefig(path)
    plt.close(fig)


SOURCE_LABELS = {"wiki_th": "Wikipedia ไทย", "wiki_en": "Wikipedia อังกฤษ", "wisesight": "Wisesight (แชท/โซเชียล)",
                 "synthetic": "ตัวอย่างหลอกที่สร้างเอง"}


def fig_dataset(path: Path) -> None:
    stats = json.loads((ROOT / "data" / "processed" / "stats.json").read_text(encoding="utf-8"))
    train = stats["splits"]["train"]
    totals = {s: stats["splits"][s]["total"] for s in ("train", "val", "test")}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.15, 1, 1.25]})

    src = sorted(train["source"].items(), key=lambda kv: kv[1])
    ys = range(len(src))
    a.barh(ys, [v for _, v in src], color=BLUE, height=0.6)
    for y, (_, v) in zip(ys, src):
        a.text(v, y, f" {v:,}", va="center", fontsize=9, color=INK)
    a.set_yticks(list(ys), [SOURCE_LABELS[k] for k, _ in src])
    a.set_xlim(0, max(v for _, v in src) * 1.3)
    a.set_title("(ก) แหล่งข้อความ (ข้อมูลฝึก)", loc="left", fontsize=10)
    a.set_xlabel("จำนวนตัวอย่าง")
    a.xaxis.set_major_formatter(THOUSANDS)
    a.grid(axis="y", visible=False)

    al = train["active_label"]
    x = [0, 1]
    ok = [al["en/ok"], al["th/ok"]]
    wrong = [al["en/wrong"], al["th/wrong"]]
    w = 0.36
    b.bar([i - w / 2 - 0.01 for i in x], ok, w, color=BLUE, label="พิมพ์ถูก (ok)")
    b.bar([i + w / 2 + 0.01 for i in x], wrong, w, color=ORANGE, label="พิมพ์ผิดแป้น (wrong)")
    for i in x:
        b.text(i - w / 2 - 0.01, ok[i], f"{ok[i]:,}", ha="center", va="bottom", fontsize=8, color=INK)
        b.text(i + w / 2 + 0.01, wrong[i], f"{wrong[i]:,}", ha="center", va="bottom", fontsize=8, color=INK)
    b.set_xticks(x, ["เครื่องตั้งเป็นอังกฤษ", "เครื่องตั้งเป็นไทย"])
    b.set_ylim(0, max(ok + wrong) * 1.18)
    b.yaxis.set_major_formatter(THOUSANDS)
    b.legend(fontsize=8, loc="upper right", frameon=False)
    b.set_title("(ข) สมดุลของเฉลย (ข้อมูลฝึก)", loc="left", fontsize=10)
    b.grid(axis="x", visible=False)

    hist = train["len_hist"]
    labels = list(hist)
    vals = [hist[k] for k in labels]
    c.bar(range(len(labels)), vals, color=BLUE, width=0.8)
    c.set_xticks(range(len(labels)), [k.replace("32-35", "32 (ตัด)") for k in labels], rotation=35, fontsize=8)
    c.set_xlabel("ความยาว (ตัวอักษร)")
    c.yaxis.set_major_formatter(THOUSANDS)
    c.set_title(f"(ค) ความยาวข้อความ (เฉลี่ย {train['mean_len']} ตัว)", loc="left", fontsize=10)
    c.grid(axis="x", visible=False)
    c.annotate("ข้อความยาวกว่า 32 ตัว\nถูกตัดเหลือ 32", xy=(len(labels) - 1, vals[-1]),
               xytext=(len(labels) - 3.6, max(vals) * 0.62), fontsize=8, color=MUTED,
               arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.8})

    fig.suptitle(f"ข้อมูลทั้งหมด {sum(totals.values()):,} ตัวอย่าง: ฝึก {totals['train']:,} / "
                 f"ตรวจสอบ {totals['val']:,} / ทดสอบ {totals['test']:,} (แบ่งตามบทความ 80/10/10)", fontsize=11)
    fig.savefig(path)
    plt.close(fig)


CNN = {"emb": 32, "channels": 96, "kernels": (2, 3, 4, 5), "hidden": 128, "max_len": 32}


def cnn_params(vocab: int) -> int:
    e, ch, ks, h = CNN["emb"], CNN["channels"], CNN["kernels"], CNN["hidden"]
    conv = sum(e * ch * k + ch for k in ks)
    feat = 2 * ch * len(ks)
    return vocab * e + conv + feat * h + h + h + 1


def box(ax, x, y, w, h, title, sub="", fc="white", ec=GRID, tc=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x, y + (0.13 if sub else 0), title, ha="center", va="center", fontsize=10, color=tc,
            fontweight="bold")
    if sub:
        ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=8, color=MUTED)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.1,
                                                              "shrinkA": 2, "shrinkB": 2})


def fig_cnn(path: Path) -> None:
    from luem.models.cnn import VOCAB_SIZE
    n_params = cnn_params(VOCAB_SIZE)
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(1.0, 6)
    ax.axis("off")
    ax.grid(False)

    ax.text(0.95, 5.55, "ข้อความบนจอ", ha="center", fontsize=9, color=MUTED)
    for i, ch in enumerate("l;y"):
        box(ax, 0.95, 4.8 - i * 0.62, 0.62, 0.5, ch, fc=BLUE_TINT, ec=BLUE)
    ax.text(0.95, 2.75, "(สูงสุด 32 ตัว)", ha="center", fontsize=8, color=MUTED)
    arrow(ax, 1.3, 4.2, 1.95, 4.2)
    box(ax, 2.95, 4.2, 1.8, 0.95, "แปลงเป็นตัวเลข", f"{VOCAB_SIZE} ตัวอักษรที่รู้จัก")
    arrow(ax, 3.85, 4.2, 4.35, 4.2)
    box(ax, 5.3, 4.2, 1.7, 0.95, "Embedding", f"ตัวละ {CNN['emb']} ค่า")

    ys = [5.35, 4.6, 3.85, 3.1]
    for k, y in zip(CNN["kernels"], ys):
        arrow(ax, 6.15, 4.2, 6.75, y)
        box(ax, 7.65, y, 1.75, 0.6, f"Conv k={k}", f"{CNN['channels']} ตัวกรอง + ReLU", fc="#eef5fd", ec=BLUE)
    ax.text(7.65, 2.45, f"ตัวกรองมองทีละ 2–5 ตัวอักษร\nเช่น \"l;y\" (k=3) ไม่เคยพบในภาษาอังกฤษ\n"
                        f"มองแค่ตัวที่พิมพ์แล้ว (causal)", ha="center", va="top", fontsize=8, color=MUTED)

    for y in ys:
        arrow(ax, 8.55, y, 9.1, 4.65 if y > 4.2 else 3.75)
    box(ax, 9.9, 4.65, 1.5, 0.62, "Max-pool", "เจอรูปแปลกไหม")
    box(ax, 9.9, 3.75, 1.5, 0.62, "Mean-pool", "แปลกมากแค่ไหน")
    arrow(ax, 10.65, 4.65, 11.05, 4.25)
    arrow(ax, 10.65, 3.75, 11.05, 4.15)
    box(ax, 11.85, 4.2, 1.5, 0.95, "Dense", f"{2 * CNN['channels'] * len(CNN['kernels'])} → {CNN['hidden']} → 1")
    arrow(ax, 12.6, 4.2, 12.95, 4.2)
    box(ax, 13.45, 4.2, 0.95, 0.95, "0.999", "p(ผิดแป้น)", fc=ORANGE_TINT, ec=ORANGE)

    ax.text(0.2, 1.65, f"พารามิเตอร์ทั้งหมด {n_params:,} ค่า  ·  ไฟล์ ONNX 0.6 MB  ·  ใช้เวลาประมาณ 0.04 ms ต่อการกด 1 ปุ่ม (CPU)",
            fontsize=9, color=INK)
    ax.text(0.2, 1.2, "ข้อความเดียวกันถูกส่งเข้าไปใหม่ทุกครั้งที่กดปุ่ม คะแนนของส่วนต้นคำจึงไม่ขึ้นกับตัวที่ยังไม่ได้พิมพ์",
            fontsize=9, color=MUTED)
    ax.set_title("โครงสร้าง CNN: จากตัวอักษรบนจอ เป็นความน่าจะเป็นที่พิมพ์ผิดแป้น", fontsize=11, loc="left")
    fig.savefig(path)
    plt.close(fig)
    return n_params


def main() -> None:
    setup_fonts()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_keyboard(FIGURES / "fig03_keyboard.png")
    fig_dataset(FIGURES / "fig10_dataset.png")
    n = fig_cnn(FIGURES / "fig12_cnn.png")
    print(f"figures -> {FIGURES} (CNN parameters: {n:,})")


if __name__ == "__main__":
    main()
