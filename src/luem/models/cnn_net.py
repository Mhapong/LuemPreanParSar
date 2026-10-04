"""The neural networks in PyTorch (used by scripts/train_cnn.py and scripts/export_onnx.py only).

Main model, char-CNN:

    ids [B, L] -> embedding -> causal Conv1d, one per kernel size, ReLU
               -> max-pool and mean-pool over the real (non-PAD) positions
               -> MLP -> logit [B]

Max-pool answers "is there any n-gram that never happens in this language?" (e.g. `l;y`);
mean-pool answers "how much of the text looks odd?", which helps tell passwords from typos.

Comparison model, char-GRU: reads left to right; the state after the last real character -> logit.
Both take ids [B, L] (PAD on the right) and give one logit per row, so training, ONNX export and
the runtime wrapper are shared and only the architecture differs.
"""

import torch
import torch.nn.functional as F
from torch import nn

from luem.models.cnn import PAD, VOCAB_SIZE

DEFAULT_CONFIG = {"emb": 32, "channels": 96, "kernels": (2, 3, 4, 5), "hidden": 128, "dropout": 0.1}


class CharCNN(nn.Module):
    def __init__(self, emb=32, channels=96, kernels=(2, 3, 4, 5), hidden=128, dropout=0.1):
        super().__init__()
        self.kernels = tuple(kernels)
        self.emb = nn.Embedding(VOCAB_SIZE, emb, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(emb, channels, k) for k in self.kernels)
        feat = 2 * channels * len(self.kernels)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != PAD).unsqueeze(1)                     # [B, 1, L]
        x = self.emb(ids).transpose(1, 2)                    # [B, E, L]
        # left padding only: position t sees characters t-k+1..t, never the ones typed later
        h = torch.cat([F.relu(conv(F.pad(x, (k - 1, 0)))) for k, conv in zip(self.kernels, self.convs)], 1)
        h = h * mask                                         # ReLU >= 0, so 0 is neutral for max
        n = mask.sum(2).clamp(min=1)
        return self.head(torch.cat([h.amax(2), h.sum(2) / n], 1)).squeeze(1)


class CharGRU(nn.Module):
    def __init__(self, emb=32, hidden=128, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.emb(ids))                       # [B, L, H]; PAD comes after, so it never
        last = (ids != PAD).sum(1).clamp(min=1) - 1          # reaches the state of a real character
        h = h.gather(1, last.view(-1, 1, 1).expand(-1, 1, h.shape[2])).squeeze(1)
        return self.head(h).squeeze(1)


def build(config: dict) -> nn.Module:
    """config as saved in a checkpoint; "arch" defaults to cnn."""
    config = dict(config)
    return (CharGRU if config.pop("arch", "cnn") == "gru" else CharCNN)(**config)


class WithSigmoid(nn.Module):
    """What gets exported: probability instead of logit, so the runtime needs no math."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, ids):
        return torch.sigmoid(self.net(ids))


def load_checkpoint(path, device="cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = build(ckpt["config"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    return net.eval(), ckpt
