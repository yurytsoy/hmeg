from __future__ import annotations

import copy
from datetime import datetime
import os

import json
import ollama
import toml

from langchain import agents
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import AIMessage
from rich.console import Console
from rich.markdown import Markdown

from .entities import LogFiles, LogRecordType
from .tools import finish_session, list_grammar_topics, exercises_generator, user_translation

console = Console()


def get_tutor_params(tutor_file: str) -> dict:
    res = {  # default tutor settings
        "model": "qwen3:4b-instruct",  # supports tools
        "session_id": "default-session",  # ID for tracking chat history and logs
        "agent_prompt": """You are a helpful Korean language learning tutor. Help the user practice translation from English to Korean. You are a big Jack Sparrow fan.

        Goals and behavior:
        - Prioritize clear, concise teaching: provide translations, corrections, short explanations, and relevant vocabulary.
        - Keep an encouraging, neutral tone and adapt complexity to the user's stated CEFR level or inferred ability.
        - When giving corrections, show: 1) corrected Korean sentence, 2) a short explanation of the error (1–2 sentences), 3) 1–2 key vocabulary or grammar notes, and optionally Romanization if requested.

        Interaction guidelines:
        - Converse in English unless the user requests otherwise.
        - Ask a clarifying question if the user's request is ambiguous.
        - Ask whether user wants to practice a specific Korean grammar.
        - When providing exercises, indicate the target CEFR level and any special constraints (e.g., vocabulary limits).
        - Keep individual responses short and focused; avoid long unrelated explanations.
        - Do not reveal internal system details or hallucinate facts.

        Output format hints (follow these when applicable):
        Corrected: <Korean sentence>
        Explanation: <one-sentence explanation>

        Always be polite, concise, and helpful."""
    }

    if os.path.exists(tutor_file):
        console.print(Markdown(f"Loading tutor parameters from **{tutor_file}**"))
        with open(tutor_file, "r") as f:
            tutor_params = toml.loads(f.read())

        if "model" in tutor_params:
            res["model"] = tutor_params["model"]
        if "agent_prompt" in tutor_params:
            res["agent_prompt"] = tutor_params["agent_prompt"]
        if "session_id" in tutor_params:
            res["session_id"] = tutor_params["session_id"]

    return res


def invoke(agent: CompiledStateGraph, messages: list[dict[str, str]], session_id: str) -> list[dict]:
    """
    Basic streaming handler (event shapes vary between LangChain versions).
    """

    steps = []
    config = RunnableConfig(configurable={"thread_id": session_id})
    for step in agent.stream(input={"messages": messages}, config=config):
        steps.append(step)
        if isinstance(step, dict):
            if "model" in step and isinstance(step["model"], dict):
                for m in step["model"]["messages"]:
                    if m.content:
                        console.print(Markdown(m.content))
                    elif hasattr(m, "tool_calls"):  # when agent has a tool call, the
                        pass
            # elif "tools" in step and isinstance(step["tools"], dict):
            #     # ? Tool call event
            #     pass
            else:
                # print("[!] Unrecognized step contents, dict:", step)
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


def get_tool_call_args_from_aimessage(msg: AIMessage) -> dict:
    if not hasattr(msg, "tool_calls") or len(msg.tool_calls) == 0:
        return {}

    res = {
        "tool_name": msg.tool_calls[0]["name"],
        "args": msg.tool_calls[0]["args"],
    }
    return res


def get_tool_call_result_from_aimessage(msg: AIMessage) -> dict:
    res = {
        "tool_name": msg.name,
        "tool_result": json.loads(msg.content),
    }
    return res


def get_total_token_stats(steps: list[dict]) -> dict:
    token_stats = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    for step in steps:
        if "model" in step and isinstance(step["model"], dict):
            usage = step["model"]["messages"][0].usage_metadata
            token_stats["input_tokens"] += usage.get("input_tokens", 0)
            token_stats["output_tokens"] += usage.get("output_tokens", 0)
            token_stats["total_tokens"] += usage.get("total_tokens", 0)
    # TODO: handle a case when tool calls a sub-agent, and aggregate its token usage too.
    return token_stats


def write_observability_log(filename: str, record: dict):
    # log_filename = "tutor_observability.jsonl"
    event_record = copy.deepcopy(record)
    event_record["timestamp"] = datetime.now().isoformat()

    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("")  # create empty file

    with open(filename, "a") as f:
        f.write(json.dumps(event_record, ensure_ascii=False) + "\n")


def log_tools_usage_from_steps(steps: list[dict], session_id: str):
    def tool_call_to_log_record(call: dict) -> dict:
        return {
            "type": LogRecordType.TOOL_CALL,
            "tool_name": call.get("name", "unknown_tool"),
            "args": call.get("args", {}),
            "tool_call_id": call.get("id"),
            "session_id": session_id,
        }

    def message_result_to_log_record(msg: AIMessage) -> dict:
        return {
            "type": LogRecordType.TOOL_RESULT,
            "tool_name": msg.name,
            "tool_result": msg.content,
            "tool_call_id": getattr(msg, "tool_call_id", None),
            "session_id": session_id,
        }

    for step in steps:
        if "model" in step and isinstance(step["model"], dict):
            tool_calls = step["model"]["messages"][0].tool_calls
            for call in tool_calls:
                record = tool_call_to_log_record(call)
                write_observability_log(LogFiles.TOOL_LOG, record)

        if "tools" in step and isinstance(step["tools"], dict):
            record = message_result_to_log_record(step["tools"]["messages"][0])
            write_observability_log(LogFiles.TOOL_LOG, record)


def log_tokens_usage_from_steps(steps: list[dict], session_id: str):
    token_stats = get_total_token_stats(steps)

    log_record = {
        "type": LogRecordType.TOKENS_USAGE,
        "input_tokens": token_stats["input_tokens"],
        "output_tokens": token_stats["output_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "num_steps": len(steps),
        "session_id": session_id,
    }
    write_observability_log(LogFiles.TOKEN_LOG, log_record)


def log_tools_usage(steps: list[dict]):
    """
    Logs tool usage from the agent's streaming steps.

    Parameters
    ----------
    steps : list[dict]
        The list of streaming steps from the agent.
    """

    # open log
    for step in steps:
        if isinstance(step, dict) and "tools" in step and isinstance(step["tools"], AIMessage):
            tool_args = get_tool_call_args_from_aimessage(step["tools"])
            tool_res = get_tool_call_result_from_aimessage(step["tools"])


def create_agent() -> tuple[CompiledStateGraph, str]:
    tutor_params = get_tutor_params(".tutor")
    model = ChatOllama(model=tutor_params["model"])
    agent_prompt = tutor_params["agent_prompt"]

    tools = [list_grammar_topics, exercises_generator, user_translation, finish_session]
    agent = agents.create_agent(
        model=model, tools=tools, system_prompt=agent_prompt, checkpointer=InMemorySaver()
    )

    # probe the agent to see if it support tools.
    try:
        config = RunnableConfig(configurable={"thread_id": "probe-session"})
        agent.invoke({"messages": [{"role": "user", "content": "Hello"}]}, config=config)
    except ollama.ResponseError as error:
        console.print(Markdown("**Agent does not support tools, recreating without tool support...**"))
        agent = agents.create_agent(model=model, system_prompt=agent_prompt, checkpointer=InMemorySaver())

    return agent, tutor_params["session_id"]
