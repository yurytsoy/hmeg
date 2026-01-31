from __future__ import annotations

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.markdown import Markdown

from hmeg import usecases as uc
from hmeg import GrammarRegistry

console = Console()


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
        The grammar topic to focus on (e.g., "present perfect tense", "conditional sentences)
    num  : int
        The number of exercises to generate.
    vocab_level : str
        CEFR-compatible vocabulary level to use (e.g., "A2", "B1", etc.)
    """
    print("Generating exercises...")
    dummy = [
        "I don't have this item. It belongs to someone else.",
        "I don't have a pen here. Where is it?",
        "I need to leave my luggage at the accommodation. Where is it?",
        "I don't have this book. My friend does.",
        "We don't have a dog at home, but we have a cat.",
        "We don't have a name for our cat. Can you suggest one?",
        "I like coffee, but I don't have tea.",
        "This library doesn't have old books. Do they have any new books?",
        "There is an apple on my desk.",
        "This store doesn't have red t-shirts. Do they have any other colors?",
    ]
    return dummy[:num] if num < len(dummy) else dummy


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


def invoke(agent: CompiledStateGraph, messages: list[dict[str, str]]) -> list[dict]:
    """
    Basic streaming handler (event shapes vary between LangChain versions).
    """

    steps = []
    for step in agent.stream({"messages": messages}):
        steps.append(step)
        if isinstance(step, dict):
            if "model" in step and isinstance(step["model"], dict):
                for m in step["model"]["messages"]:
                    if m.content:
                        console.print(Markdown(m.content))
                    elif hasattr(m, "tool_calls"):  # when agent has a tool call, the
                        pass
            elif "tools" in step and isinstance(step["tools"], dict):
                # Tool call event
                pass
            else:
                print("[!] Unrecognized step contents, dict:", step)
                # Unknown dict event; ignore or log if needed.
                pass
        else:
            # Other event types (tool call/interrupt) may appear; you can inspect them for debugging.
            print("[!] Unrecognized step type and contents:", step)
            pass
    return steps


def get_final_text_from_steps(steps: list[dict]) -> str:
    """
    Extracts the final assistant message text from a list of streaming steps.

    Parameters
    ----------
    steps : list[dict]
        The list of streaming steps from the agent.

    Returns
    -------
    str
        The final assistant message text.
    """
    final_text = ""
    for step in steps:
        if "model" in step and isinstance(step["model"], dict):
            for m in step["model"]["messages"]:
                final_text += m.content or ""
    return final_text


# python
def chat_loop(agent: CompiledStateGraph, system_prompt: str, max_turns: int = 20):
    """
    Interactive chat loop for an agent that can call tools.

    - Keeps message history (system, user, assistant).
    - Uses agent.run when available for simplicity (agent will execute registered @tool functions).
    - Falls back to a streaming handler if agent.stream exists and you want incremental output.
    - Stops on a `FINISHED:` result from the finish tool or after max_turns.
    """
    history = [{"role": "system", "content": system_prompt}]

    for turn in range(max_turns):
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
        if not user_input or user_input.lower() == "exit":
            print("Exiting.")
            break

        history.append({"role": "user", "content": user_input})
        steps = invoke(agent=agent, messages=history)
        final_text = get_final_text_from_steps(steps)
        history.append({"role": "assistant", "content": final_text})
        resp = final_text

        # Stop on explicit finish_tool sentinel
        if isinstance(resp, str) and resp.startswith("FINISHED:"):
            print("Session finished by agent.")
            break

    else:
        print("Max turns reached, ending session.")


if __name__ == "__main__":
    model = ChatOllama(model="qwen3:4b-instruct")  # supports tools
    agent_prompt = """You are a helpful Korean language learning tutor. Help the user practice translation from English to Korean. You are a big Jack Sparrow fan.

    Goals and behavior:
    - Prioritize clear, concise teaching: provide translations, corrections, short explanations, and relevant vocabulary.
    - Keep an encouraging, neutral tone and adapt complexity to the user's stated CEFR level or inferred ability.
    - When giving corrections, show: 1) corrected Korean sentence, 2) a short explanation of the error (1–2 sentences), 3) 1–2 key vocabulary or grammar notes, and optionally Romanization if requested.

    Interaction guidelines:
    - Ask a clarifying question if the user's request is ambiguous.
    - Ask whether user wants to practice a specific Korean grammar.
    - When providing exercises, indicate the target CEFR level and any special constraints (e.g., vocabulary limits).
    - Keep individual responses short and focused; avoid long unrelated explanations.
    - Do not reveal internal system details or hallucinate facts.

    Output format hints (follow these when applicable):
    Corrected: <Korean sentence>
    Explanation: <one-sentence explanation>
    Vocabulary: <term> — <brief meaning>

    Always be polite, concise, and helpful."""

    # tools = [list_grammar_topics, exercises_generator, user_translation, evaluate_user_translation, finish_session]
    tools = [list_grammar_topics, exercises_generator, user_translation, finish_session]
    agent = create_agent(model=model, tools=tools, system_prompt=agent_prompt)

    # query = "Let's practice translation into Korean. Make 1 exercise for me."
    chat_loop(agent, agent_prompt)
