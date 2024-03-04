import dataclasses


class MinilexPlaceholders:
    Noun = "{noun}"  # noun, singular
    Pronoun = "{pronoun}"  # personal pronoun: I, we, they, you
    PronounSingular3rd = "{pronoun:s}"  # personal pronoun for 3rd person, singular: he, she, it
    Verb = "{verb}"  # verb, present simple
    VerbSingular3rd = "{verb:s}"  # verb in the present simple for 3rd person, singular


@dataclasses.dataclass
class GrammarDescription:
    name: str
    links: list[str]
    exercises: list[str]
