"""Model registry: load any trained model by name, so eval and demo can swap models freely."""

from pathlib import Path

from luem.models.base import Model

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
NAMES = ("dictionary", "ngram")


def load_model(name: str, models_dir: Path = MODELS_DIR) -> Model:
    if name == "dictionary":
        from luem.models.dictionary import DictionaryModel
        return DictionaryModel.load(models_dir / "dictionary_en_words.txt")
    if name == "ngram":
        from luem.models.ngram import NgramModel
        return NgramModel.load(models_dir / "ngram.pkl")
    raise ValueError(f"unknown model {name!r}, choose from {NAMES}")
