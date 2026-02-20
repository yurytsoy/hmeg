from __future__ import annotations

import copy
from datetime import datetime
import json
import os

from langchain_core.messages import AIMessage

from hmeg.tutor.entities import LogFiles, LogRecordType, SESSION_ID_SEPARATOR, TUTOR_DIR
from .utils import get_total_token_stats, get_agent_text_from_steps


def get_log_dir(session_id: str):
    tutor_id, student_id = session_id.split(SESSION_ID_SEPARATOR)
    log_dir = os.path.join(TUTOR_DIR, LogFiles.LOG_DIR, tutor_id, student_id)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir



def write_observability_log(filename: str, record: dict):
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


def log_agent_steps(
    steps: list[dict], session_id: str, log_message: bool=True, log_tokens: bool=True, log_tools: bool=True
):
    if log_message: log_ai_message_from_steps(steps, session_id)
    if log_tokens: log_tokens_usage_from_steps(steps, session_id)
    if log_tools: log_tools_usage_from_steps(steps, session_id)


def log_ai_message_from_steps(steps: list[dict], session_id: str):
    msg = get_agent_text_from_steps(steps)
    record = message_to_Log_record(msg=msg, role="ai", session_id=session_id, msg_id=None)
    log_file = os.path.join(get_log_dir(session_id), LogFiles.MESSAGE_LOG)
    write_messages_log(log_file, [record])


def log_user_message(msg: str, session_id: str):
    record = message_to_Log_record(msg=msg, role="user", session_id=session_id, msg_id=None)
    log_file = os.path.join(get_log_dir(session_id), LogFiles.MESSAGE_LOG)
    write_messages_log(log_file, [record])


def message_to_Log_record(msg: str, role: str, session_id: str, msg_id: str | None=None) -> dict:
    return {
        "role": role,
        "content": msg,
        "timestamp": datetime.now().isoformat(),
        "id": msg_id,
        "thread_id": session_id,
    }


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
    log_file = os.path.join(get_log_dir(session_id), LogFiles.TOKEN_LOG)
    write_observability_log(log_file, log_record)


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

    log_file = os.path.join(get_log_dir(session_id), LogFiles.TOOL_LOG)
    for step in steps:
        if "model" in step and isinstance(step["model"], dict):
            tool_calls = step["model"]["messages"][0].tool_calls
            for call in tool_calls:
                record = tool_call_to_log_record(call)
                write_observability_log(log_file, record)

        if "tools" in step and isinstance(step["tools"], dict):
            record = message_result_to_log_record(step["tools"]["messages"][0])
            write_observability_log(log_file, record)
