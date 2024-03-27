from __future__ import annotations
import dataclasses


VOWELS = "aeiou"


class VocabularyPlaceholders:
    Adjective = "{adj}"  # adjective
    Adverb = "{adverb}"  # adverb
    ANoun = "{a:noun}"  # noun, singular, with a preceding a/an article
    Country = "{country}"  # name of the country
    Month = "{month}"  # month
    Nationality = "{nationality}"  # nationality
    Noun = "{noun}"  # noun, singular
    Number100 = "{number:100}"  # number below 100
    Number1000 = "{number:1000}"  # number below 1000
    Number100k = "{number:100000}"  # number below 100000
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
            VocabularyPlaceholders.ANoun,
            VocabularyPlaceholders.Country,
            VocabularyPlaceholders.Nationality,
            VocabularyPlaceholders.Noun,
            VocabularyPlaceholders.Number100,
            VocabularyPlaceholders.Number1000,
            VocabularyPlaceholders.Number100k,
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
