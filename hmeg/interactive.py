from __future__ import annotations

from hmeg.tutor.main_loop import chat_loop
from hmeg.tutor.usecases import create_agent, extract_messages_from_checkpointer


if __name__ == "__main__":
    agent = create_agent("tutor.conf")
    student_id = "student_001"
    chat_loop(agent, student_id=student_id)
