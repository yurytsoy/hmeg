from __future__ import annotations

from langchain.tools import tool

from hmeg import GrammarRegistry
from hmeg import usecases as uc


@tool
def list_grammar_topics() -> list[str]:
    """
    Lists available grammar topics for translation exercises.

    Returns
    -------
    list[str]
        A list of available grammar topics.
    """

    uc.register_grammar_topics(force=False)
    topics = GrammarRegistry.get_registered_topics()
    return topics
