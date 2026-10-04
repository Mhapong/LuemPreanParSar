"""Model 3 (main): character CNN, run from an ONNX file so the demo needs no PyTorch.

The comparison GRU exports to the same ONNX interface (ids -> p_wrong), so this wrapper runs both.

The network itself lives in cnn_net.py (training and export only). Here: the shared vocabulary
and the runtime wrapper that eval/evaluate.py and the daemon use.

Convolutions are causal (each position sees only itself and the characters before it), so a
prefix padded inside a batch scores exactly like the prefix on its own: predict_prefixes() scores
all prefixes of a chunk in one batched call.
"""

from pathlib import Path

import numpy as np

from luem.dataset import MAX_LEN
from luem.layout import EN_KEYS, TH_KEYS
from luem.models.base import Model, views

PAD, UNK = 0, 1
VOCAB = "".join(sorted(set(EN_KEYS) | set(TH_KEYS)))  # every character either layout can type
_INDEX = {c: i + 2 for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB) + 2


def encode(text: str, length: int | None = None) -> list[int]:
    """Character ids, cut to MAX_LEN and right-padded with PAD to `length`.

    >>> encode("ab", 4)[2:]
    [0, 0]
    """
    ids = [_INDEX.get(c, UNK) for c in text[:MAX_LEN]]
    return ids + [PAD] * ((length or len(ids)) - len(ids))


class CNNModel(Model):
    name = "cnn"

    def __init__(self, path: Path, threads: int = 1, name: str = "cnn"):
        import onnxruntime as ort

        self.name = name
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads  # the daemon scores one short chunk at a time
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        vocab = self.session.get_modelmeta().custom_metadata_map.get("vocab")
        if vocab is not None and vocab != VOCAB:
            raise ValueError(f"{path} was trained with a different vocabulary than layout.py gives now")

    @classmethod
    def load(cls, path: Path, name: str = "cnn") -> "CNNModel":
        return cls(path, name=name)

    def _run(self, ids: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"ids": ids})[0]

    def predict(self, text: str) -> float:
        if views(text) is None:
            return 0.0
        return float(self._run(np.array([encode(text)], dtype=np.int64))[0])

    def predict_prefixes(self, text: str) -> list[float]:
        n = min(len(text), MAX_LEN)
        if n == 0 or views(text) is None:
            return [0.0] * len(text)
        ids = np.array(encode(text, n), dtype=np.int64)
        batch = np.tril(np.broadcast_to(ids, (n, n)))  # row k = first k+1 characters, then PAD
        probs = self._run(batch)
        out = [float(p) if views(text[:k + 1]) else 0.0 for k, p in enumerate(probs)]
        return out + out[-1:] * (len(text) - n)  # past MAX_LEN the daemon's buffer stops growing
