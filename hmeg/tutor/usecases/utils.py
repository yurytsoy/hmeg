from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph


def get_total_token_stats(steps: list[dict]) -> dict:
    token_stats = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    for step in steps:
        if "model" in step and isinstance(step["model"], dict):
            msgs = step["model"].get("messages", [])
            if not msgs:
                continue
            usage = getattr(msgs[0], "usage_metadata", {})
            token_stats["input_tokens"] += usage.get("input_tokens", 0)
            token_stats["output_tokens"] += usage.get("output_tokens", 0)
            token_stats["total_tokens"] += usage.get("total_tokens", 0)
    # TODO: handle a case when tool calls a sub-agent, and aggregate its token usage too.
    return token_stats


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


def get_agent_text_from_steps(steps: list[dict]) -> str:
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


def extract_messages_from_checkpointer(agent: CompiledStateGraph, config: RunnableConfig) -> list[dict]:
    """
    Extracts messages from the agent's checkpointer.

    Parameters
    ----------
    agent : CompiledStateGraph
        The agent from which to extract messages.
    config : RunnableConfig
        The runnable configuration used for the session.

    Returns
    -------
    list[dict]
        The list of messages extracted from the checkpointer.
    """
    if not agent.checkpointer:
        return []

    last = agent.checkpointer.get(config)
    if last is None:
        return []

    result = []
    messages = last.get("channel_values", {}).get("messages", [])
    for msg in messages:
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage) and (not msg.usage_metadata or not msg.content or not msg.response_metadata):
            continue
        result.append({"role": msg.type, "content": msg.content, "timestamp": (msg.response_metadata or {}).get("created_at"), "id": msg.id, "thread_id": config["configurable"]["thread_id"]})
    return result
