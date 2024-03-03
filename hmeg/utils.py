import os
import tomllib


from .entities import GrammarDescription
from .factory import GrammarFactory


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
            GrammarFactory.register_grammar_topic(grammar_descr)
