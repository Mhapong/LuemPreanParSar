"""Train the char-CNN (or, with --arch gru, the comparison GRU) on data/processed/train.jsonl, pick the best epoch on val, save models/cnn.pt.

Usage:
    uv run python scripts/train_cnn.py --limit 20000 --epochs 2          # quick check on any machine
    uv run python scripts/train_cnn.py                                  # full run (GPU if available)
    uv run python scripts/train_cnn.py --channels 128 --hard-weight 3 --out models/cnn_b.pt
    uv run python scripts/train_cnn.py --arch gru --out models/gru.pt
then: uv run python scripts/export_onnx.py && uv run python eval/evaluate.py --model cnn

The dataset stores full chunks. Every time a sample is drawn, the batch is cut to a random prefix
(what the screen shows while the user is still typing), so each epoch sees different prefixes.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

from luem.dataset import MAX_LEN, read_samples, script_of_chunk
from luem.metrics import average_precision
from luem.models import MODELS_DIR
from luem.models.cnn import VOCAB, encode
from luem.models.cnn_net import DEFAULT_CONFIG, build

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"


def first_judgeable(text: str) -> int:
    """Shortest prefix length with a letter: shorter prefixes are always scored 0 by the wrapper."""
    for k in range(1, len(text) + 1):
        if script_of_chunk(text[:k]):
            return k
    return len(text)


def tensors(rows: list[dict], hard_weight: float) -> TensorDataset:
    ids = torch.tensor([encode(r["text"], MAX_LEN) for r in rows], dtype=torch.uint8)
    lens = torch.tensor([min(len(r["text"]), MAX_LEN) for r in rows])
    first = torch.tensor([first_judgeable(r["text"][:MAX_LEN]) for r in rows])
    y = torch.tensor([r["label"] == "wrong" for r in rows], dtype=torch.float32)
    w = torch.tensor([hard_weight if r["kind"] == "hard_negative" else 1.0 for r in rows])
    return TensorDataset(ids, lens, first, y, w)


def cut_to_prefix(ids: torch.Tensor, lens: torch.Tensor, first: torch.Tensor, full_prob: float,
                  gen: torch.Generator | None = None) -> torch.Tensor:
    """Keep a random prefix of each row: the full chunk with prob full_prob, else k ~ U[first, len]."""
    u = torch.rand(len(ids), device=ids.device, generator=gen)
    span = (lens - first + 1).float()
    k = first + (torch.rand(len(ids), device=ids.device, generator=gen) * span).long()
    k = torch.where(u < full_prob, lens, torch.minimum(k, lens))
    keep = torch.arange(ids.shape[1], device=ids.device) < k.unsqueeze(1)
    return ids.long() * keep


@torch.no_grad()
def validate(net, val: TensorDataset, device, batch_size: int, hard: torch.Tensor) -> dict:
    """Loss on full chunks and on fixed random prefixes, AP, and false fixes on hard negatives."""
    net.eval()
    gen = torch.Generator(device=device).manual_seed(0)  # same prefixes every epoch
    out = {"full": [], "prefix": []}
    for i in range(0, len(val), batch_size):
        ids, lens, first, y, _ = (t[i:i + batch_size].to(device) for t in val.tensors)
        out["full"].append(torch.sigmoid(net(ids.long())))
        out["prefix"].append(torch.sigmoid(net(cut_to_prefix(ids, lens, first, 0.0, gen))))
    y = val.tensors[3].to(device)
    p_full, p_pre = torch.cat(out["full"]), torch.cat(out["prefix"])
    yy, hh = y.cpu().numpy().astype(bool), hard.numpy()
    pf = p_full.cpu().numpy()
    return {
        "loss_full": F.binary_cross_entropy(p_full, y).item(),
        "loss_prefix": F.binary_cross_entropy(p_pre, y).item(),
        "ap_full": average_precision(yy, pf),
        # threshold-free view of the problem n-gram had: how many hard negatives score above
        # the score that catches 95% of wrong chunks
        "hardneg_fp_at_r95": float((pf[hh] >= np.quantile(pf[yy], 0.05)).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, default=DATA / "train.jsonl")
    p.add_argument("--val", type=Path, default=DATA / "val.jsonl")
    p.add_argument("--out", type=Path, default=MODELS_DIR / "cnn.pt")
    p.add_argument("--limit", type=int, default=0, help="random subset of train (0 = all); val uses limit/5")
    p.add_argument("--val-limit", type=int, default=50_000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024, help="lower it on 'CUDA out of memory'")
    p.add_argument("--workers", type=int, default=0, help="DataLoader subprocesses (0 = none, safest on Windows)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--full-prob", type=float, default=0.4, help="share of samples shown as the full chunk")
    p.add_argument("--hard-weight", type=float, default=1.0, help="loss weight of hard negatives")
    p.add_argument("--arch", choices=("cnn", "gru"), default="cnn")
    p.add_argument("--emb", type=int, default=DEFAULT_CONFIG["emb"])
    p.add_argument("--channels", type=int, default=DEFAULT_CONFIG["channels"])
    p.add_argument("--kernels", type=int, nargs="+", default=list(DEFAULT_CONFIG["kernels"]))
    p.add_argument("--hidden", type=int, default=DEFAULT_CONFIG["hidden"])
    p.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.time()

    train_rows = read_samples(args.train, args.limit, seed=args.seed)
    val_rows = read_samples(args.val, args.limit // 5 if args.limit else args.val_limit, seed=1)
    train, val = tensors(train_rows, args.hard_weight), tensors(val_rows, 1.0)
    val_hard = torch.tensor([r["kind"] == "hard_negative" for r in val_rows])
    del train_rows, val_rows
    print(f"train={len(train):,} val={len(val):,} device={device}"
          f"{' (' + torch.cuda.get_device_name(device) + ')' if device.type == 'cuda' else ''} ({time.time() - t0:.0f}s)")

    # sampler yields whole index lists: one tensor gather per batch instead of batch_size small ones
    loader = DataLoader(train, sampler=BatchSampler(RandomSampler(train), args.batch_size, drop_last=False),
                        batch_size=None, num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=device.type == "cuda")

    config = {"emb": args.emb, "hidden": args.hidden, "dropout": args.dropout}
    if args.arch == "cnn":
        config |= {"channels": args.channels, "kernels": args.kernels}
    else:
        config["arch"] = "gru"
    net = build(config).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"{type(net).__name__} {config}: {n_params:,} parameters")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(loader)
    warmup = min(500, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))

    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        net.train()
        total, seen, te = 0.0, 0, time.time()
        for ids, lens, first, y, w in loader:
            ids, lens, first, y, w = (t.to(device, non_blocking=True) for t in (ids, lens, first, y, w))
            logits = net(cut_to_prefix(ids, lens, first, args.full_prob))
            loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
            seen += len(y)
        v = validate(net, val, device, 8192, val_hard)
        v = {"epoch": epoch, "train_loss": total / seen, **{k: round(x, 6) for k, x in v.items()},
             "seconds": round(time.time() - te, 1)}
        history.append(v)
        score = v["loss_full"] + v["loss_prefix"]
        mark = ""
        if best is None or score < best:
            best, mark = score, " *"
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": config, "state_dict": net.state_dict(), "vocab": VOCAB, "epoch": epoch,
                        "args": {k: str(x) if isinstance(x, Path) else x for k, x in vars(args).items()},
                        "n_params": n_params}, args.out)
        print(f"epoch {epoch:2}: train {v['train_loss']:.4f} | val full {v['loss_full']:.4f} "
              f"prefix {v['loss_prefix']:.4f} AP {v['ap_full']:.5f} hardneg@r95 {v['hardneg_fp_at_r95']:.4f} "
              f"| {v['seconds']:.0f}s{mark}")

    hist_path = args.out.with_name(args.out.stem + "_history.json")
    hist_path.write_text(json.dumps({"config": config, "n_params": n_params, "history": history}, indent=2),
                         encoding="utf-8")
    print(f"best checkpoint -> {args.out}, history -> {hist_path} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
