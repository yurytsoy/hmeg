from __future__ import annotations

from hmeg.tutor.main_loop import chat_loop
from hmeg.tutor.usecases import create_agent


if __name__ == "__main__":
    agent, session_id = create_agent()
    chat_loop(agent, session_id)
