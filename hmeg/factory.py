"""
Factory for generation of exercises for the given topic.
"""

from __future__ import annotations

from .entities import GrammarDescription


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
