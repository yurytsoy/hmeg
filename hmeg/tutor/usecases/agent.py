from __future__ import annotations

import os
import uuid

from langchain import agents
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.markdown import Markdown
import toml

from hmeg.tutor.entities import SESSION_ID_SEPARATOR
from hmeg.tutor.tools import finish_session, list_grammar_topics, exercises_generator

console = Console()


def get_tutor_params(tutor_file: str) -> dict:
    res = {  # default tutor settings
        "model": "qwen3:4b-instruct",  # supports tools
        "tutor_id": "default-tutor",  # ID for tracking chat history and logs
        "context_size": 32768,  # context window size for the model
        "agent_prompt": """You are a helpful Korean language learning tutor for an English-speaking student. Help the user practice translation from English to Korean. You are a big Futurama fan.

Goals and Behavior:
- Prioritize clear, concise teaching: provide translations, corrections, short explanations, and relevant vocabulary.
- Notify student when they get something right, but focus on constructive feedback for mistakes.
- Only provide corrections when the user attempts a translation and makes an error. Do not correct if the user is just asking questions or practicing without attempting a translation.
- Keep an encouraging, neutral tone and adapt complexity to the user's stated CEFR level.
- When giving corrections, show: 1) corrected Korean sentence, 2) a short explanation of the error (1–2 sentences), 3) 1–2 key vocabulary or grammar notes, and optionally Romanization if requested.

Session Management:
- You are orchestrating a learning session, not just answering questions.
- Use `exercises_generator` when the user requests practice exercises for a specific grammar topic and/or CEFR level.
- Use `list_grammar_topics` if you need to show available topics or the user's request is vague.
- Use `finish_session` when the user explicitly says they're done (e.g., "exit", "goodbye") or after completing a natural learning unit.
- Track what has been practiced in this session and avoid redundant exercises unless the user requests repetition.

Interaction Guidelines:
- **Communicate in English at all times** unless the user requests otherwise.
- Ask a clarifying question if the user's request is ambiguous.
- Ask whether user wants to practice a specific Korean grammar.
- When providing exercises, indicate the target CEFR level and any special constraints (e.g., vocabulary limits).
- Keep individual responses short and focused; avoid long unrelated explanations.
- Do not reveal internal system details or hallucinate facts.

Always be polite, concise, and helpful.

Start from the greeting and ask user about exercises that they want to practice."""
    }

    if os.path.exists(tutor_file):
        console.print(Markdown(f"Loading tutor parameters from **{tutor_file}**"))
        with open(tutor_file, "r") as f:
            tutor_params = toml.loads(f.read())

        if "model" in tutor_params:
            res["model"] = tutor_params["model"]
        if "agent_prompt" in tutor_params:
            res["agent_prompt"] = tutor_params["agent_prompt"]
        res["tutor_id"] = tutor_params.get("tutor_id", res["model"])
        res["context_size"] = tutor_params.get("context_size", 32768)

    return res


def invoke(agent: CompiledStateGraph, user_message: str, config: dict | RunnableConfig) -> list[dict]:
    """
    Basic streaming handler (event shapes vary between LangChain versions).
    """

    steps = []
    for step in agent.stream(
        input={"messages": [HumanMessage(content=user_message)]}, config=config,
    ):
        steps.append(step)
        if isinstance(step, dict):
            if "model" in step and isinstance(step["model"], dict):
                for m in step["model"]["messages"]:
                    if m.content:
                        console.print(Markdown(m.content))
                    elif hasattr(m, "tool_calls"):
                        pass
            else:
                pass
        else:
            # Other event types (tool call/interrupt) may appear; you can inspect them for debugging.
            print("[!] Unrecognized step type and contents:", step)
            pass
    return steps


def make_session_id(agent_name: str, student_id: str | None = None) -> str:
    return f"{agent_name or ''}{SESSION_ID_SEPARATOR}{student_id or str(uuid.uuid4())}"


def supports_tools(model: BaseChatModel, tools: list) -> bool:
    try:
        model.bind_tools(tools)
        return True
    except (NotImplementedError, ValueError):
        return False


def create_agent(tutor_config_path: str | None) -> CompiledStateGraph:
    tutor_config_path = tutor_config_path or "tutor.conf"

    tutor_params = get_tutor_params(tutor_config_path)
    model = ChatOllama(model=tutor_params["model"], num_ctx=tutor_params["context_size"])
    agent_prompt = tutor_params["agent_prompt"]
    tools = [list_grammar_topics, exercises_generator, finish_session]

    # probe the agent to see if it support tools.
    if supports_tools(model, tools):
        agent = agents.create_agent(
            model=model, tools=tools, system_prompt=agent_prompt, checkpointer=InMemorySaver(),
            name=tutor_params.get("tutor_id", None)
        )
    else:
        console.print(Markdown("**Agent does not support tools, recreating without tool support...**"))
        agent = agents.create_agent(model=model, system_prompt=agent_prompt, checkpointer=InMemorySaver())

    return agent
