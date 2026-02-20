from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from hmeg.tutor import usecases as tutor_usecases


def chat_loop(agent: CompiledStateGraph, student_id: str, max_turns: int = 20):
    """
    Interactive chat loop for an agent that can call tools.

    - Keeps message history (system, user, assistant).
    - Uses agent.run when available for simplicity (agent will execute registered @tool functions).
    - Falls back to a streaming handler if agent.stream exists and you want incremental output.
    - Stops on: empty user message; "exit" command; `FINISHED:` result from the finish tool; after max_turns.
    """

    session_id = tutor_usecases.make_session_id(agent_name=agent.name, student_id=student_id)
    config = RunnableConfig(configurable={"thread_id": session_id})
    user_input = "[Begin the session]"
    for turn in range(max_turns):
        steps = tutor_usecases.invoke(agent=agent, user_message=user_input, config=config)
        tutor_usecases.log_agent_steps(steps=steps, session_id=session_id)  # log the agent's response, tool calls, and token usage for this turn.

        if tutor_usecases.is_finish(steps):
            print("Session finished by agent.")
            break

        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
        tutor_usecases.log_user_message(msg=user_input, session_id=session_id)

        if not user_input or user_input.lower().strip() == "exit":
            print("Exiting.")
            break

    else:
        print("Max turns reached, ending session.")
