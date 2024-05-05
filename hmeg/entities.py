from __future__ import annotations
import dataclasses


VOWELS = "aeiou"


@dataclasses.dataclass
class VocabularyInfo:
    name: str
    num_words: int
    num_nouns: int
    num_verbs: int
    num_adjectives: int
    num_adverbs: int


class VocabularyPlaceholders:
    Adjective = "{adj}"  # adjective
    Adverb = "{adverb}"  # adverb
    ANoun = "{a:noun}"  # noun, singular, with a preceding a/an article
    ANounNonPerson = "{a:noun:nonperson}"  # noun, singular, with a preceding a/an article, excluding people
    City = "{city}"  # name of the country
    Country = "{country}"  # name of the country
    Month = "{month}"  # month
    Nationality = "{nationality}"  # nationality
    Noun = "{noun}"  # noun, singular
    NounNonPerson = "{noun:nonperson}"  # noun, singular, except of people related
    NounPlural = "{noun:plural}"  # noun, plural
    Number100 = "{number:100}"  # number below 100
    Number1000 = "{number:1000}"  # number below 1000
    Number100k = "{number:100000}"  # number below 100000
    Person = "{person}"  # place or location
    Place = "{place}"  # place or location
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
            VocabularyPlaceholders.City,
            VocabularyPlaceholders.Country,
            VocabularyPlaceholders.Nationality,
            VocabularyPlaceholders.Noun,
            VocabularyPlaceholders.Number100,
            VocabularyPlaceholders.Number1000,
            VocabularyPlaceholders.Number100k,
            VocabularyPlaceholders.Place,
            # MinilexPlaceholders.Pronoun,
            # MinilexPlaceholders.PronounSingular3rd,
            VocabularyPlaceholders.Verb,
            VocabularyPlaceholders.VerbSingular3rd,
            VocabularyPlaceholders.VerbPast,
            VocabularyPlaceholders.VerbProgressive,
        ]


@dataclasses.dataclass
class TopicLevelInfo:
    resource_name: str
    level: int | str | None = None


@dataclasses.dataclass
class GrammarDescription:
    name: str
    links: list[str]
    exercises: list[str]
    levels: list[TopicLevelInfo]

    @staticmethod
    def from_dict(d: dict) -> GrammarDescription:
        res = GrammarDescription(
            name=d["name"],
            links=d["links"],
            exercises=d["exercises"],
            levels=[TopicLevelInfo(**level_descr) for level_descr in d.get("levels", [])],
        )
        return res
