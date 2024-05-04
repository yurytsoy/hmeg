"""
Factory for generation of exercises for the given topic.
"""

from __future__ import annotations

from .entities import GrammarDescription


class GrammarRegistry:
    topics: dict[str, GrammarDescription] = dict()

    @staticmethod
    def reset():
        GrammarRegistry.topics = dict()

    @staticmethod
    def register_grammar_topic(grammar_descr: GrammarDescription):
        if grammar_descr.name in GrammarRegistry.topics:
            return
        GrammarRegistry.topics[grammar_descr.name] = grammar_descr

    @staticmethod
    def get_registered_topics() -> list[str]:
        """
        Returns names of registered topics.
        """
        return list(GrammarRegistry.topics)

    @staticmethod
    def get_registered_levels() -> dict[str, list[int]]:
        """
        Returns information about registered levels for topics.
        The levels are grouped by the resource and sorted in the ascending order.
        """
        res = dict()
        for topic in GrammarRegistry.topics.values():
            for level_descr in topic.levels:
                if level_descr.resource_name not in res:
                    res[level_descr.resource_name] = list()
                if level_descr.level:
                    res[level_descr.resource_name].append(level_descr.level)

        for key in res:
            res[key] = sorted(set(res[key]))
        return res

    @staticmethod
    def find_topics(topic_name: str) -> list[str]:
        """
        Find registered topic that fully or partially matches the provided name.

        Args
        ----
        topic_name: str,
            Full or partial name of the topic to find.

        Returns
        -------
        list[str]
            List of found registered topics. If nothing is found then empty list is returned.
        """

        if not topic_name:
            return []

        res = []
        lower_topic_name = topic_name.lower()
        for cur_topic in GrammarRegistry.topics:
            if lower_topic_name in cur_topic.lower():
                res.append(cur_topic)
        return res

    @staticmethod
    def generate_answers(exercises: list[str], grammar_topic: str | None = None) -> list[list[str]]:
        """
        Generate list of possible answers for the given exercises.
        Each exercise can have several possible answers.
        """
        ...
