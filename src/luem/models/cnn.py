from pathlib import Path

import numpy as np

from luem.dataset import MAX_LEN
from luem.layout import EN_KEYS, TH_KEYS
from luem.models.base import Model, views

PAD, UNK = 0, 1
VOCAB = "".join(sorted(set(EN_KEYS) | set(TH_KEYS)))
_INDEX = {c: i + 2 for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB) + 2


def encode(text: str, length: int | None = None) -> list[int]:
    ids = [_INDEX.get(c, UNK) for c in text[:MAX_LEN]]
    return ids + [PAD] * ((length or len(ids)) - len(ids))


class CNNModel(Model):
    name = "cnn"

    def __init__(self, path: Path, threads: int = 1, name: str = "cnn"):
        import onnxruntime as ort

        self.name = name
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
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
        batch = np.tril(np.broadcast_to(ids, (n, n)))
        probs = self._run(batch)
        out = [float(p) if views(text[:k + 1]) else 0.0 for k, p in enumerate(probs)]
        return out + out[-1:] * (len(text) - n)
