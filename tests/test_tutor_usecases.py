from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from hmeg.tutor.entities import LogFiles, LogRecordType, Roles, SESSION_ID_SEPARATOR, TUTOR_DIR
from hmeg.tutor.usecases.utils import (
    extract_messages_from_checkpointer,
    get_agent_text_from_steps,
    get_tool_call_args_from_aimessage,
    get_tool_call_result_from_aimessage,
    get_total_token_stats,
)
from hmeg.tutor.usecases.observability import (
    get_log_dir,
    log_agent_steps,
    log_tokens_usage_from_steps,
    log_tools_usage_from_steps,
    log_user_message,
    message_to_log_record,
    write_messages_log,
    write_observability_log,
)
from hmeg.tutor.usecases.agent import get_tutor_params, make_session_id, supports_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ai_message_with_usage(input_tokens: int, output_tokens: int, total_tokens: int) -> MagicMock:
    msg = MagicMock(spec=AIMessage)
    msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    msg.content = ""
    msg.tool_calls = []
    return msg


def _make_step_with_model_message(msg) -> dict:
    return {"model": {"messages": [msg]}}


# ---------------------------------------------------------------------------
# Tests for hmeg.tutor.usecases.utils
# ---------------------------------------------------------------------------

class TestGetTotalTokenStats(unittest.TestCase):
    def test_empty_steps_returns_zeros(self):
        result = get_total_token_stats([])
        self.assertEqual(result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

    def test_single_step_accumulates_tokens(self):
        msg = _make_ai_message_with_usage(10, 5, 15)
        steps = [_make_step_with_model_message(msg)]
        result = get_total_token_stats(steps)
        self.assertEqual(result["input_tokens"], 10)
        self.assertEqual(result["output_tokens"], 5)
        self.assertEqual(result["total_tokens"], 15)

    def test_multiple_steps_accumulate_tokens(self):
        msg1 = _make_ai_message_with_usage(10, 5, 15)
        msg2 = _make_ai_message_with_usage(20, 8, 28)
        steps = [_make_step_with_model_message(msg1), _make_step_with_model_message(msg2)]
        result = get_total_token_stats(steps)
        self.assertEqual(result["input_tokens"], 30)
        self.assertEqual(result["output_tokens"], 13)
        self.assertEqual(result["total_tokens"], 43)

    def test_step_without_model_key_is_skipped(self):
        steps = [{"tools": {"messages": []}}]
        result = get_total_token_stats(steps)
        self.assertEqual(result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

    def test_step_with_empty_messages_is_skipped(self):
        steps = [{"model": {"messages": []}}]
        result = get_total_token_stats(steps)
        self.assertEqual(result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

    def test_step_model_not_a_dict_is_skipped(self):
        steps = [{"model": "not_a_dict"}]
        result = get_total_token_stats(steps)
        self.assertEqual(result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})


class TestGetToolCallArgsFromAimessage(unittest.TestCase):
    def test_no_tool_calls_returns_empty_dict(self):
        msg = MagicMock(spec=AIMessage)
        msg.tool_calls = []
        result = get_tool_call_args_from_aimessage(msg)
        self.assertEqual(result, {})

    def test_message_without_tool_calls_attr_returns_empty_dict(self):
        msg = MagicMock()
        del msg.tool_calls
        result = get_tool_call_args_from_aimessage(msg)
        self.assertEqual(result, {})

    def test_with_tool_calls_returns_name_and_args(self):
        msg = MagicMock(spec=AIMessage)
        msg.tool_calls = [{"name": "my_tool", "args": {"param": "value"}}]
        result = get_tool_call_args_from_aimessage(msg)
        self.assertEqual(result["tool_name"], "my_tool")
        self.assertEqual(result["args"], {"param": "value"})


class TestGetToolCallResultFromAimessage(unittest.TestCase):
    def test_valid_json_content_is_parsed(self):
        msg = MagicMock(spec=ToolMessage)
        msg.name = "my_tool"
        msg.content = json.dumps(["result1", "result2"])
        result = get_tool_call_result_from_aimessage(msg)
        self.assertEqual(result["tool_name"], "my_tool")
        self.assertEqual(result["tool_result"], ["result1", "result2"])

    def test_json_object_content_is_parsed(self):
        msg = MagicMock(spec=ToolMessage)
        msg.name = "other_tool"
        msg.content = json.dumps({"key": "val"})
        result = get_tool_call_result_from_aimessage(msg)
        self.assertEqual(result["tool_result"], {"key": "val"})


class TestGetAgentTextFromSteps(unittest.TestCase):
    def test_empty_steps_returns_empty_string(self):
        self.assertEqual(get_agent_text_from_steps([]), "")

    def test_extracts_text_from_model_messages(self):
        msg = MagicMock()
        msg.content = "Hello!"
        steps = [{"model": {"messages": [msg]}}]
        result = get_agent_text_from_steps(steps)
        self.assertEqual(result, "Hello!")

    def test_concatenates_content_across_steps(self):
        msg1 = MagicMock()
        msg1.content = "Part one. "
        msg2 = MagicMock()
        msg2.content = "Part two."
        steps = [{"model": {"messages": [msg1]}}, {"model": {"messages": [msg2]}}]
        result = get_agent_text_from_steps(steps)
        self.assertEqual(result, "Part one. Part two.")

    def test_none_content_treated_as_empty_string(self):
        msg = MagicMock()
        msg.content = None
        steps = [{"model": {"messages": [msg]}}]
        result = get_agent_text_from_steps(steps)
        self.assertEqual(result, "")

    def test_non_model_steps_are_ignored(self):
        steps = [{"tools": {"messages": []}}]
        result = get_agent_text_from_steps(steps)
        self.assertEqual(result, "")


class TestExtractMessagesFromCheckpointer(unittest.TestCase):
    def test_no_checkpointer_returns_empty_list(self):
        agent = MagicMock()
        agent.checkpointer = None
        config = {"configurable": {"thread_id": "t1"}}
        result = extract_messages_from_checkpointer(agent, config)
        self.assertEqual(result, [])

    def test_checkpointer_returns_none_gives_empty_list(self):
        agent = MagicMock()
        agent.checkpointer.get.return_value = None
        config = {"configurable": {"thread_id": "t1"}}
        result = extract_messages_from_checkpointer(agent, config)
        self.assertEqual(result, [])

    def test_tool_messages_are_excluded(self):
        tool_msg = ToolMessage(content="tool result", tool_call_id="id1")
        agent = MagicMock()
        agent.checkpointer.get.return_value = {
            "channel_values": {"messages": [tool_msg]}
        }
        config = {"configurable": {"thread_id": "t1"}}
        result = extract_messages_from_checkpointer(agent, config)
        self.assertEqual(result, [])

    def test_ai_messages_without_content_are_excluded(self):
        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.type = "ai"
        ai_msg.content = ""
        ai_msg.usage_metadata = {"input_tokens": 10}
        ai_msg.response_metadata = {"created_at": "2024-01-01"}
        ai_msg.id = "msg1"
        agent = MagicMock()
        agent.checkpointer.get.return_value = {
            "channel_values": {"messages": [ai_msg]}
        }
        config = {"configurable": {"thread_id": "t1"}}
        result = extract_messages_from_checkpointer(agent, config)
        self.assertEqual(result, [])

    def test_human_messages_are_included(self):
        human_msg = HumanMessage(content="Hello!")
        agent = MagicMock()
        agent.checkpointer.get.return_value = {
            "channel_values": {"messages": [human_msg]}
        }
        config = {"configurable": {"thread_id": "t1"}}
        result = extract_messages_from_checkpointer(agent, config)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "Hello!")
        self.assertEqual(result[0]["thread_id"], "t1")


# ---------------------------------------------------------------------------
# Tests for hmeg.tutor.usecases.observability
# ---------------------------------------------------------------------------

class TestGetLogDir(unittest.TestCase):
    def test_creates_directory_with_expected_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("hmeg.tutor.usecases.observability.TUTOR_DIR", tmpdir):
                session_id = f"my-tutor{SESSION_ID_SEPARATOR}student-001"
                log_dir = get_log_dir(session_id)
                expected = os.path.join(tmpdir, LogFiles.LOG_DIR, "my-tutor", "student-001")
                self.assertEqual(log_dir, expected)
                self.assertTrue(os.path.isdir(log_dir))

    def test_invalid_session_id_error_message(self):
        with self.assertRaisesRegex(ValueError, "not enough values to unpack"):
            get_log_dir("no-separator-here")


class TestWriteObservabilityLog(unittest.TestCase):
    def test_creates_file_and_appends_json_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.jsonl")
            record = {"type": "test", "value": 42}
            write_observability_log(log_file, record)

            self.assertTrue(os.path.exists(log_file))
            with open(log_file) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["type"], "test")
            self.assertEqual(parsed["value"], 42)
            self.assertIn("timestamp", parsed)
            datetime.fromisoformat(parsed["timestamp"])  # raises ValueError if not valid ISO format

    def test_appends_multiple_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.jsonl")
            write_observability_log(log_file, {"n": 1})
            write_observability_log(log_file, {"n": 2})

            with open(log_file) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["n"], 1)
            self.assertEqual(json.loads(lines[1])["n"], 2)

    def test_original_record_is_not_mutated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.jsonl")
            record = {"type": "test"}
            write_observability_log(log_file, record)
            self.assertNotIn("timestamp", record)


class TestWriteMessagesLog(unittest.TestCase):
    def test_creates_file_and_writes_all_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "messages.jsonl")
            records = [{"role": "user", "content": "hi"}, {"role": "ai", "content": "hello"}]
            write_messages_log(log_file, records)

            with open(log_file) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["role"], "user")
            self.assertEqual(json.loads(lines[1])["role"], "ai")

    def test_appends_to_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "messages.jsonl")
            write_messages_log(log_file, [{"n": 1}])
            write_messages_log(log_file, [{"n": 2}])

            with open(log_file) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)


class TestMessageToLogRecord(unittest.TestCase):
    def test_returns_expected_keys(self):
        record = message_to_log_record(msg="hello", role=Roles.USER, session_id="t1 : s1")
        self.assertEqual(record["role"], Roles.USER)
        self.assertEqual(record["content"], "hello")
        self.assertEqual(record["thread_id"], "t1 : s1")
        self.assertIn("timestamp", record)
        datetime.fromisoformat(record["timestamp"])  # raises ValueError if not valid ISO format
        self.assertIsNone(record["id"])

    def test_msg_id_is_stored(self):
        record = message_to_log_record(msg="hi", role=Roles.AI, session_id="t : s", msg_id="abc123")
        self.assertEqual(record["id"], "abc123")


class TestLogUserMessage(unittest.TestCase):
    def test_log_user_message_writes_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("hmeg.tutor.usecases.observability.TUTOR_DIR", tmpdir):
                session_id = f"tutor{SESSION_ID_SEPARATOR}student"
                log_user_message(msg="test message", session_id=session_id)

                log_dir = os.path.join(tmpdir, LogFiles.LOG_DIR, "tutor", "student")
                log_file = os.path.join(log_dir, LogFiles.MESSAGE_LOG)
                self.assertTrue(os.path.exists(log_file))

                with open(log_file) as f:
                    lines = f.readlines()
                self.assertEqual(len(lines), 1)
                parsed = json.loads(lines[0])
                self.assertEqual(parsed["content"], "test message")
                self.assertEqual(parsed["role"], Roles.USER)


class TestLogAgentSteps(unittest.TestCase):
    def test_delegates_to_sub_functions(self):
        steps = []
        session_id = f"t{SESSION_ID_SEPARATOR}s"
        with (
            patch("hmeg.tutor.usecases.observability.log_ai_message_from_steps") as mock_msg,
            patch("hmeg.tutor.usecases.observability.log_tokens_usage_from_steps") as mock_tokens,
            patch("hmeg.tutor.usecases.observability.log_tools_usage_from_steps") as mock_tools,
        ):
            log_agent_steps(steps, session_id, log_message=True, log_tokens=True, log_tools=True)
            mock_msg.assert_called_once_with(steps, session_id)
            mock_tokens.assert_called_once_with(steps, session_id)
            mock_tools.assert_called_once_with(steps, session_id)

    def test_flags_control_sub_function_calls(self):
        steps = []
        session_id = f"t{SESSION_ID_SEPARATOR}s"
        with (
            patch("hmeg.tutor.usecases.observability.log_ai_message_from_steps") as mock_msg,
            patch("hmeg.tutor.usecases.observability.log_tokens_usage_from_steps") as mock_tokens,
            patch("hmeg.tutor.usecases.observability.log_tools_usage_from_steps") as mock_tools,
        ):
            log_agent_steps(steps, session_id, log_message=False, log_tokens=False, log_tools=False)
            mock_msg.assert_not_called()
            mock_tokens.assert_not_called()
            mock_tools.assert_not_called()


class TestLogTokensUsageFromSteps(unittest.TestCase):
    def test_writes_token_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("hmeg.tutor.usecases.observability.TUTOR_DIR", tmpdir):
                session_id = f"tutor{SESSION_ID_SEPARATOR}student"
                msg = _make_ai_message_with_usage(5, 3, 8)
                steps = [_make_step_with_model_message(msg)]
                log_tokens_usage_from_steps(steps, session_id)

                log_dir = os.path.join(tmpdir, LogFiles.LOG_DIR, "tutor", "student")
                log_file = os.path.join(log_dir, LogFiles.TOKEN_LOG)
                self.assertTrue(os.path.exists(log_file))

                with open(log_file) as f:
                    parsed = json.loads(f.readline())
                self.assertEqual(parsed["type"], LogRecordType.TOKENS_USAGE)
                self.assertEqual(parsed["input_tokens"], 5)
                self.assertEqual(parsed["output_tokens"], 3)
                self.assertEqual(parsed["total_tokens"], 8)
                self.assertEqual(parsed["session_id"], session_id)


class TestLogToolsUsageFromSteps(unittest.TestCase):
    def test_logs_tool_calls_from_model_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("hmeg.tutor.usecases.observability.TUTOR_DIR", tmpdir):
                session_id = f"tutor{SESSION_ID_SEPARATOR}student"
                msg = MagicMock()
                msg.tool_calls = [{"name": "my_tool", "args": {"x": 1}, "id": "call_1"}]
                steps = [{"model": {"messages": [msg]}}]
                log_tools_usage_from_steps(steps, session_id)

                log_dir = os.path.join(tmpdir, LogFiles.LOG_DIR, "tutor", "student")
                log_file = os.path.join(log_dir, LogFiles.TOOL_LOG)
                self.assertTrue(os.path.exists(log_file))

                with open(log_file) as f:
                    parsed = json.loads(f.readline())
                self.assertEqual(parsed["type"], LogRecordType.TOOL_CALL)
                self.assertEqual(parsed["tool_name"], "my_tool")
                self.assertEqual(parsed["session_id"], session_id)
                self.assertIn("timestamp", parsed)
                datetime.fromisoformat(parsed["timestamp"])  # raises ValueError if not valid ISO format

    def test_logs_tool_result_from_tools_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("hmeg.tutor.usecases.observability.TUTOR_DIR", tmpdir):
                session_id = f"tutor{SESSION_ID_SEPARATOR}student"
                result_msg = MagicMock()
                result_msg.name = "my_tool"
                result_msg.content = "some result"
                result_msg.tool_call_id = "call_1"
                steps = [{"tools": {"messages": [result_msg]}}]
                log_tools_usage_from_steps(steps, session_id)

                log_dir = os.path.join(tmpdir, LogFiles.LOG_DIR, "tutor", "student")
                log_file = os.path.join(log_dir, LogFiles.TOOL_LOG)
                self.assertTrue(os.path.exists(log_file))

                with open(log_file) as f:
                    parsed = json.loads(f.readline())
                self.assertEqual(parsed["type"], LogRecordType.TOOL_RESULT)
                self.assertEqual(parsed["tool_name"], "my_tool")
                self.assertEqual(parsed["tool_result"], "some result")

    def test_empty_messages_list_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("hmeg.tutor.usecases.observability.TUTOR_DIR", tmpdir):
                session_id = f"tutor{SESSION_ID_SEPARATOR}student"
                steps = [{"model": {"messages": []}}, {"tools": {"messages": []}}]
                # Should not raise
                log_tools_usage_from_steps(steps, session_id)


# ---------------------------------------------------------------------------
# Tests for hmeg.tutor.usecases.agent
# ---------------------------------------------------------------------------

class TestGetTutorParams(unittest.TestCase):
    def test_returns_defaults_when_file_does_not_exist(self):
        result = get_tutor_params("/nonexistent/path/tutor.conf")
        self.assertIn("model", result)
        self.assertIn("tutor_id", result)
        self.assertIn("context_size", result)
        self.assertIn("agent_prompt", result)
        self.assertIsInstance(result["model"], str)
        self.assertIsInstance(result["context_size"], int)

    def test_loads_model_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('model = "llama3:8b"\n')
            f.write('tutor_id = "test-tutor"\n')
            tmp_path = f.name

        try:
            result = get_tutor_params(tmp_path)
            self.assertEqual(result["model"], "llama3:8b")
            self.assertEqual(result["tutor_id"], "test-tutor")
        finally:
            os.unlink(tmp_path)

    def test_loads_agent_prompt_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('agent_prompt = "Custom prompt text"\n')
            tmp_path = f.name

        try:
            result = get_tutor_params(tmp_path)
            self.assertEqual(result["agent_prompt"], "Custom prompt text")
        finally:
            os.unlink(tmp_path)

    def test_loads_context_size_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("context_size = 8192\n")
            tmp_path = f.name

        try:
            result = get_tutor_params(tmp_path)
            self.assertEqual(result["context_size"], 8192)
        finally:
            os.unlink(tmp_path)

    def test_defaults_used_for_missing_keys_in_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('model = "custom-model"\n')
            tmp_path = f.name

        try:
            result = get_tutor_params(tmp_path)
            # context_size should fall back to default when not in file
            self.assertEqual(result["context_size"], 32768)
        finally:
            os.unlink(tmp_path)


class TestMakeSessionId(unittest.TestCase):
    def test_uses_provided_agent_and_student(self):
        session_id = make_session_id("my-agent", "student-001")
        self.assertEqual(session_id, f"my-agent{SESSION_ID_SEPARATOR}student-001")

    def test_generates_uuid_when_no_student_id(self):
        import re
        session_id = make_session_id("my-agent")
        parts = session_id.split(SESSION_ID_SEPARATOR)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "my-agent")
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        self.assertRegex(parts[1], uuid_pattern)

    def test_none_agent_name_produces_empty_prefix(self):
        session_id = make_session_id(None, "student-002")
        expected = f"{SESSION_ID_SEPARATOR}student-002"
        self.assertEqual(session_id, expected)

    def test_separator_is_present(self):
        session_id = make_session_id("agent", "student")
        self.assertIn(SESSION_ID_SEPARATOR, session_id)


class TestSupportsTools(unittest.TestCase):
    def test_returns_true_when_bind_tools_succeeds(self):
        model = MagicMock()
        model.bind_tools.return_value = MagicMock()
        tools = [MagicMock()]
        self.assertTrue(supports_tools(model, tools))

    def test_returns_false_on_not_implemented_error(self):
        model = MagicMock()
        model.bind_tools.side_effect = NotImplementedError
        tools = [MagicMock()]
        self.assertFalse(supports_tools(model, tools))

    def test_returns_false_on_value_error(self):
        model = MagicMock()
        model.bind_tools.side_effect = ValueError("tools not supported")
        tools = [MagicMock()]
        self.assertFalse(supports_tools(model, tools))


if __name__ == "__main__":
    unittest.main()
