from __future__ import annotations
import dataclasses


class VocabularyPlaceholders:
    Adjective = "{adj}"  # adjective
    Adverb = "{adverb}"  # adverb
    Noun = "{noun}"  # noun, singular
    # Pronoun = "{pronoun}"  # personal pronoun: I, we, they, you
    # PronounSingular3rd = "{pronoun:3s}"  # personal pronoun for 3rd person, singular: he, she, it
    Season = "{season}"  # season
    Verb = "{verb}"  # verb, present simple
    VerbSingular3rd = "{verb:3s}"  # verb in the present simple for 3rd person, singular
    VerbPast = "{verb:past}"  # verb, past
    VerbProgressive = "{verb:ing}"  # verb, progressive
    Weekday = "{weekday}"  # day of the week

    @staticmethod
    def to_list() -> list[str]:
        return [
            VocabularyPlaceholders.Adjective,
            VocabularyPlaceholders.Adverb,
            VocabularyPlaceholders.Noun,
            # MinilexPlaceholders.Pronoun,
            # MinilexPlaceholders.PronounSingular3rd,
            VocabularyPlaceholders.Verb,
            VocabularyPlaceholders.VerbSingular3rd,
            VocabularyPlaceholders.VerbPast,
            VocabularyPlaceholders.VerbProgressive,
        ]


@dataclasses.dataclass
class GrammarDescription:
    name: str
    links: list[str]
    exercises: list[str]
