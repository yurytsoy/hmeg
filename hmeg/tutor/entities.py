from __future__ import annotations

from dataclasses import dataclass


TUTOR_DIR = ".tutor"
SESSION_ID_SEPARATOR = " : "


@dataclass
class LogRecordType:
    TOKENS_USAGE: str = "tokens_usage"
    TOOL_CALL: str = "tool_call"
    TOOL_RESULT: str = "tool_result"


class LogFiles:
    LOG_DIR: str = "logs"
    TOKEN_LOG: str = "tutor_tokens_usage.jsonl"
    TOOL_LOG: str = "tutor_tool_calls.jsonl"
    MESSAGE_LOG: str = "messages.jsonl"
