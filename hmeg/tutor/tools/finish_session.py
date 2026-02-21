from __future__ import annotations

from langchain.tools import tool
from langchain_core.messages import ToolMessage

FINISH_TOKEN = "[FINISHED]"


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
    return FINISH_TOKEN



def is_finish(steps: list[dict]) -> bool:
    """
    Detects whether agent execution steps have call to the finish_session tool.
    """
    for step in steps:
        node = step.get("model") or step.get("tools")
        if not node:
            continue
        msgs = node.get("messages", [])

        for msg in msgs:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                finish_calls = [tool_call for tool_call in msg.tool_calls if tool_call["name"] == "finish_session"]
                if finish_calls:
                    return True
            elif isinstance(msg, ToolMessage) and msg.name == "finish_session":
                return True

    return False
