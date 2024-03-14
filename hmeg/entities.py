from __future__ import annotations
import dataclasses


class MinilexPlaceholders:
    Adjective = "{adj}"  # adjective
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
            MinilexPlaceholders.Adjective,
            MinilexPlaceholders.Noun,
            # MinilexPlaceholders.Pronoun,
            # MinilexPlaceholders.PronounSingular3rd,
            MinilexPlaceholders.Verb,
            MinilexPlaceholders.VerbSingular3rd,
            MinilexPlaceholders.VerbPast,
            MinilexPlaceholders.VerbProgressive,
        ]


@dataclasses.dataclass
class GrammarDescription:
    name: str
    links: list[str]
    exercises: list[str]
