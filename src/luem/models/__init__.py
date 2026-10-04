"""Model registry: load any trained model by name, so eval and demo can swap models freely."""

from pathlib import Path

from luem.models.base import Model

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
NAMES = ("dictionary", "ngram", "cnn", "gru")
FILES = {"dictionary": "dictionary_en_words.txt", "ngram": "ngram.pkl", "cnn": "cnn.onnx", "gru": "gru.onnx"}


def model_path(name: str, models_dir: Path = MODELS_DIR) -> Path:
    if name not in FILES:
        raise ValueError(f"unknown model {name!r}, choose from {NAMES}")
    return models_dir / FILES[name]


def load_model(name: str, models_dir: Path = MODELS_DIR) -> Model:
    path = model_path(name, models_dir)
    if name == "dictionary":
        from luem.models.dictionary import DictionaryModel
        return DictionaryModel.load(path)
    if name == "ngram":
        from luem.models.ngram import NgramModel
        return NgramModel.load(path)
    from luem.models.cnn import CNNModel  # cnn and gru: same ONNX interface, see cnn.py
    return CNNModel.load(path, name)
