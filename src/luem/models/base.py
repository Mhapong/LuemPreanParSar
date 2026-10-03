"""Common interface: every model answers P(typed in the wrong layout) for what is on screen."""

from luem.dataset import script_of_chunk
from luem.layout import en_to_th, th_to_en


def views(text: str) -> tuple[str, str, str, str] | None:
    """(active layout, text as typed, other layout, same keys read in the other layout).

    None when the text has no letter yet (e.g. "555", "(", "เ"): there is no evidence to judge,
    and models must answer 0.0 so numbers and punctuation are never "fixed".

    >>> views("l;y")
    ('en', 'l;y', 'th', 'สวั')
    """
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
        """Score of every prefix text[:1], text[:2], ... (what the model sees while the user types).

        Models that can do it in one pass (n-gram, GRU) override this.
        """
        return [self.predict(text[:k]) for k in range(1, len(text) + 1)]
