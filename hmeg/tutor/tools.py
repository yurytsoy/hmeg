from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain.tools import tool

from hmeg import GrammarRegistry
from hmeg import usecases as uc
from hmeg.exercise_generator import ExerciseGenerator


@tool
def list_grammar_topics() -> list[str]:
    """
    Lists available grammar topics for translation exercises.

    Returns
    -------
    list[str]
        A list of available grammar topics.
    """

    uc.register_grammar_topics()
    topics = GrammarRegistry.get_registered_topics()
    return topics


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

    Returns:
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


@tool
def translate_text(text: str) -> str:
    """
    Translates the given text to Korean.

    Parameters
    ----------
    text : str
        The text to translate.

    Returns
    -------
    str
        The translated Korean text.
    """
    print("Translating text...")

    model = ChatOllama(model="translategemma:4b")
    messages = [
        ("system", "You are a helpful translator. Translate the user sentence to Korean."),
        ("human", text),
    ]
    resp = model.invoke(messages)
    return resp.text


@tool
def evaluate_user_translation(exercise: str, user_translation: str, tutor_translation: str) -> str:
    """
    Evaluates the user's Korean translation against the correct translation.

    Parameters
    ----------
    exercise : str
        The original text.
    user_translation : str
        The user's Korean translation.
    tutor_translation : str
        Exercise reference translation from a tutor.

    Returns
    -------
    str
        Feedback on the user's translation.
    """
    print("Evaluating user translation...")

    model = ChatOllama(model="translategemma:4b")
    prompt = f"""
    You are a helpful Korean language tutor.
    Evaluate user's translation for the given exercise against the tutor translation.
    Points of evaluation: grammar usage; style; naturalness; vocabulary choice.

    Provide concise feedback highlighting differences and suggestions for improvement.

    Input:
    - exercise: {exercise}
    - user_translation: {user_translation}
    - correct_translation: {tutor_translation}
    """
    messages = [("system", prompt)]
    resp = model.invoke(messages)
    return resp.text


@tool
def user_translation(exercise: str) -> str:
    """
    Invites user to provide Korean translation to the given exercise text.
    Waits until user provides the translation.

    Parameters
    ----------
    exercise : str
        The text to be translated by the user.

    Returns
    -------
    str
        The user's translation of the exercise.
    """
    print("Waiting for user translation...")

    res = input(f"Please provide your Korean translation for the following sentence:\n{exercise}\nYour translation: ")
    return res


@tool
def finish_session(message: str) -> str:
    """
    Finishes the current practice session with a message.

    Parameters
    ----------
    message : str
        The message to display upon finishing the session.
    """
    print("Finishing session...")
    return f"FINISHED: {message}"
