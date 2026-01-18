from dataclasses import dataclass


@dataclass
class TranslationExercise:
    """
    A translation exercise consisting of source text and its optional translation.
    """
    source_text: str
    translation: str | None = None
