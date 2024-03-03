"""
Factory for generation of exercises for the given topic.
"""

from __future__ import annotations

import os
import tomllib

from .entities import GrammarDescription


def register_grammar_topics():
    # iterate over files in hmeg.grammar, load descriptions and register in the factory.
    grammar_dir = "hmeg/grammar/"
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


class GrammarFactory:
    topics: dict[str, GrammarDescription] = dict()

    @staticmethod
    def register_grammar_topic(grammar_descr: GrammarDescription):
        if grammar_descr.name in GrammarFactory.topics:
            return
        GrammarFactory.topics[grammar_descr.name] = grammar_descr

    @staticmethod
    def get_registered_topics() -> list[str]:
        """
        Returns names of registered topics.
        """
        ...

    @staticmethod
    def generate_exercises(topic_name: str, num: str) -> list[str]:
        """
        Generates list of random translation exercises for the given topic.
        """
        ...

    @staticmethod
    def generate_answers(exercises: list[str], grammar_topic: str | None = None) -> list[list[str]]:
        """
        Generate list of possible answers for the given exercises.
        Each exercise can have several possible answers.
        """
        ...
