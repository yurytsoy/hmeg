from dataclasses import dataclass


@dataclass
class TranslationExercise:
    """
    A translation exercise consisting of source text and its optional translation.
    """
    sentence_en: str
    sentence_kr: str | None = None
