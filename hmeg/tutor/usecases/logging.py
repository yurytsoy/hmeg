from __future__ import annotations

import copy
from datetime import datetime
import json
import os

from langchain.messages import AIMessage

from hmeg.tutor.entities import LogFiles, LogRecordType
from .utils import get_tool_call_result_from_aimessage, get_tool_call_args_from_aimessage, get_total_token_stats


def write_observability_log(filename: str, record: dict):
    # log_filename = "tutor_observability.jsonl"
    event_record = copy.deepcopy(record)
    event_record["timestamp"] = datetime.now().isoformat()

    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("")  # create empty file

    with open(filename, "a") as f:
        f.write(json.dumps(event_record, ensure_ascii=False) + "\n")


def write_messages_log(filename: str, records: list[dict]):
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("")  # create empty file

    with open(filename, "a") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


# def log_tools_usage(steps: list[dict]):
#     """
#     Logs tool usage from the agent's streaming steps.
#
#     Parameters
#     ----------
#     steps : list[dict]
#         The list of streaming steps from the agent.
#     """
#
#     # open log
#     for step in steps:
#         if isinstance(step, dict) and "tools" in step and isinstance(step["tools"], AIMessage):
#             tool_args = get_tool_call_args_from_aimessage(step["tools"])
#             tool_res = get_tool_call_result_from_aimessage(step["tools"])


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
