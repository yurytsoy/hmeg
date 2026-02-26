from __future__ import annotations

from langchain.tools import tool

from hmeg.exercise_generator import ExerciseGenerator


@tool
def exercises_generator(grammar_topic: str, num: int, vocab_level: str) -> list[str]:
    """
    Generates translation exercises for the user to practice translating sentences into Korean.

    Parameters
    ----------
    grammar_topic : str
        The Korean grammar topic to focus on (e.g., "주세요", "-아/어/여도", etc.)
    num : int
        The number of exercises to generate.
    vocab_level : str
        CEFR-compatible vocabulary level to use (e.g., "A2", "B1", etc.)

    Returns
    -------
    list[str]
        List of generated exercises for translation to Korean.
    """

    exercises = ExerciseGenerator.generate_exercises_ollama(
        topic_name=grammar_topic,
        num=num,
        vocab_level=vocab_level,
    )
    return [ex.sentence_en for ex in exercises]
