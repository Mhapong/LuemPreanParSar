from luem.dataset import script_of_chunk
from luem.layout import en_to_th, th_to_en


def views(text: str) -> tuple[str, str, str, str] | None:
    active = script_of_chunk(text)
    if active is None:
        return None
    if active == "en":
        return "en", text, "th", en_to_th(text)
    return "th", text, "en", th_to_en(text)


class Model:
    name = "base"

    def predict(self, text: str) -> float:
        raise NotImplementedError

    def predict_prefixes(self, text: str) -> list[float]:
        return [self.predict(text[:k]) for k in range(1, len(text) + 1)]
