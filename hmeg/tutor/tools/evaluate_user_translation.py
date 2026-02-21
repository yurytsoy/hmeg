from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain.tools import tool


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
