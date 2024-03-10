from __future__ import annotations
import dataclasses


class MinilexPlaceholders:
    Noun = "{noun}"  # noun, singular
    Pronoun = "{pronoun}"  # personal pronoun: I, we, they, you
    PronounSingular3rd = "{pronoun:3s}"  # personal pronoun for 3rd person, singular: he, she, it
    Verb = "{verb}"  # verb, present simple
    VerbSingular3rd = "{verb:3s}"  # verb in the present simple for 3rd person, singular
    VerbPast = "{verb:past}"  # verb, past

    @staticmethod
    def to_list() -> list[str]:
        return [
            MinilexPlaceholders.Noun,
            MinilexPlaceholders.Pronoun,
            MinilexPlaceholders.PronounSingular3rd,
            MinilexPlaceholders.Verb,
            MinilexPlaceholders.VerbSingular3rd,
            MinilexPlaceholders.VerbPast,
        ]


@dataclasses.dataclass
class GrammarDescription:
    name: str
    links: list[str]
    exercises: list[str]
