from __future__ import annotations

import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from .usecases import invoke, get_final_text_from_steps, log_tools_usage_from_steps, log_tokens_usage_from_steps


def chat_loop(agent: CompiledStateGraph, student_id: str, max_turns: int = 20):
    """
    Interactive chat loop for an agent that can call tools.

    - Keeps message history (system, user, assistant).
    - Uses agent.run when available for simplicity (agent will execute registered @tool functions).
    - Falls back to a streaming handler if agent.stream exists and you want incremental output.
    - Stops on: empty user message; "exit" command; `FINISHED:` result from the finish tool; after max_turns.
    """

    def make_session_id() -> str:
        return f"{agent.name or ''} : {student_id or str(uuid.uuid4())}"

    history = []
    session_id = make_session_id()
    config = RunnableConfig(configurable={"thread_id": session_id})
    for turn in range(max_turns):
        steps = invoke(agent=agent, messages=history, config=config)
        final_text = get_final_text_from_steps(steps)
        history.append({"role": "assistant", "content": final_text})
        resp = final_text

        log_tools_usage_from_steps(steps, session_id=session_id)
        log_tokens_usage_from_steps(steps, session_id=session_id)

        # Stop on explicit finish_tool sentinel
        if isinstance(resp, str) and resp.startswith("FINISHED:"):
            print("Session finished by agent.")
            break

        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        history.append({"role": "user", "content": user_input})
        if not user_input or user_input.lower() == "exit":
            print("Exiting.")
            break

    else:
        print("Max turns reached, ending session.")

    # TODO: extract messages from a checkpointer. (agent.checkpointer.list(config))


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

    last = agent.checkpointer.list(config)[0].checkpoint
    result = []
    messages = last.get("channel_values", {}).get("messages", [])
    for msg in messages:
        if msg.type == "tool":
            continue
        if msg.type == "ai" and (not msg.usage_metadata or not msg.content):
            continue
        result.append(msg.content)
    return messages
