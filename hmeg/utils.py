import os
import random
import tomllib


from .entities import GrammarDescription, MinilexPlaceholders
from .registry import GrammarRegistry
from .minilex import nouns, verbs


def register_grammar_topics(grammar_dir: str | None = None):
    """
    Read and register descriptions of grammar exercises.
    """

    # iterate over files in hmeg.grammar, load descriptions and register in the factory.
    grammar_dir = grammar_dir or "hmeg/grammar/"

    for file in os.listdir(grammar_dir):
        if not file.endswith(".toml"):
            continue
        with open(os.path.join(grammar_dir, file), "r") as f:
            grammar_descr_dict = tomllib.loads(f.read())
            grammar_descr = GrammarDescription(
                name=grammar_descr_dict["name"],
                links=grammar_descr_dict["links"],
                exercises=grammar_descr_dict["exercises"],
            )
            GrammarRegistry.register_grammar_topic(grammar_descr)


def apply_minilex(s: str) -> str:
    """
    Takes input string and replaces placeholders with respective minilex entities.
    """

    while MinilexPlaceholders.Verb in s:
        s = s.replace(MinilexPlaceholders.Verb, random.choice(verbs), 1)

    while MinilexPlaceholders.Noun in s:
        s = s.replace(MinilexPlaceholders.Noun, random.choice(nouns), 1)

    return s