import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import torch

from luem.dataset import read_samples
from luem.models import MODELS_DIR
from luem.models.cnn import VOCAB, CNNModel, encode
from luem.models.cnn_net import WithSigmoid, load_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=MODELS_DIR / "cnn.pt")
    p.add_argument("--out", type=Path, default=None, help="default: checkpoint path with .onnx")
    p.add_argument("--check", type=int, default=2000, help="val chunks to compare")
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args()
    out = args.out or args.ckpt.with_suffix(".onnx")

    net, ckpt = load_checkpoint(args.ckpt)
    if ckpt["vocab"] != VOCAB:
        raise SystemExit("checkpoint vocabulary differs from layout.py: retrain")
    model = WithSigmoid(net).eval()
    example = torch.tensor([encode("l;ylfu")])
    torch.onnx.export(model, (example,), str(out), input_names=["ids"], output_names=["p_wrong"],
                      dynamic_axes={"ids": {0: "batch", 1: "length"}, "p_wrong": {0: "batch"}},
                      opset_version=17, dynamo=False)
    m = onnx.load(str(out))
    onnx.checker.check_model(m)
    for key, value in {"vocab": VOCAB, "epoch": str(ckpt["epoch"]), "config": str(ckpt["config"])}.items():
        m.metadata_props.add(key=key, value=value)
    onnx.save(m, str(out))
    print(f"{args.ckpt} (epoch {ckpt['epoch']}, {ckpt['n_params']:,} params) -> {out} "
          f"({out.stat().st_size / 1024:.0f} KB)")

    rows = read_samples(ROOT / "data" / "processed" / "val.jsonl", args.check, seed=3)
    runtime = CNNModel(out)
    worst, n, t0 = 0.0, 0, time.time()
    with torch.no_grad():
        for r in rows:
            ids = np.array(encode(r["text"]), dtype=np.int64)
            prefixes = np.tril(np.broadcast_to(ids, (len(ids), len(ids))))
            ref = model(torch.from_numpy(prefixes.copy())).numpy()
            got = runtime._run(prefixes)
            worst = max(worst, float(np.abs(ref - got).max()))
            n += len(got)
    print(f"checked {len(rows):,} chunks / {n:,} prefixes: max |torch - onnx| = {worst:.2e} ({time.time() - t0:.1f}s)")
    if worst > args.tol:
        raise SystemExit(f"FAIL: difference above {args.tol}")
    print("OK: ONNX gives the same scores")


if __name__ == "__main__":
    main()
