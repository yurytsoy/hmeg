"""
Factory for generation of exercises for the given topic.
"""

from __future__ import annotations


def register_grammar_topics():
    # iterate over files in hmeg.grammar, load descriptions and register in the factory.
    ...


class GrammarFactory:
    @staticmethod
    def register_grammar_topic(topic_description):
        ...

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
