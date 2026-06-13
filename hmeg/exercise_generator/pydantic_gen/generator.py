import logfire

logfire.configure(send_to_logfire="always", console=False)
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

from dataclasses import dataclass
import os
import shutil
import tempfile
from typing import Any

from pydantic_ai import Agent, AgentRunResult, exceptions
from rich.console import Console

from .usecases import make_generator_agent, make_evaluator_agent, get_num_lines, read_all_lines, copy_lines, GeneratorDeps, EvaluatorDeps

console = Console()


def _run_agent(agent: Agent, prompt: str | None = None, deps: type[dataclass] | None = None) -> AgentRunResult[Any] | None:
    agent_resp = None
    try:
        shared_history = []
        agent_resp = agent.run_sync(
            prompt, message_history=shared_history, deps=deps
        )
    except exceptions.UnexpectedModelBehavior as ex:
        print(f"Agent failed! Reason: {ex}\n")

        # 2. Inspect what messages the agent generated before failing
        if not shared_history:
            print("--- NO CONVERSATION HISTORY ---")
        else:
            print("--- FAILED CONVERSATION HISTORY ---")
            for msg in shared_history:
                # Pydantic AI messages have roles: 'user', 'model', or 'tool-return'
                role = getattr(msg, 'role', 'unknown')
                print(f"[{role.upper()}]:")

                # Format or grab text/parts depending on message type
                if hasattr(msg, 'parts'):
                    print(msg.parts)
                print("-" * 20)
    return agent_resp


def generate_exercises(
    topic_name: str, num: int, gen_model: str | None = None, eval_model: str | None = None, vocab_level: str | None = None, out_path: str | None = None, verbose: bool = False, debug: bool = False
) -> list[str]:
    out_path = out_path or "result.txt"  # TODO: include sanitized topic name into the filename
    batch_size = min(num, 10)

    gen_agent = make_generator_agent(model_name=gen_model)
    eval_agent = make_evaluator_agent(model_name=eval_model)
    max_loops = max(int(1.41 * num) // batch_size, num // batch_size + 1)  # include some possibility for the failed attempts
    loop_count = 0
    cur_num_exercises = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        while (cur_num_exercises < num) and (loop_count < max_loops):
            cur_batch_size = min(batch_size, num - cur_num_exercises)

            # ? TODO: add list of nouns and verbs to avoid or use dynamic instructions
            ex_filename = os.path.join(tmp_dir, f"ex_{cur_num_exercises}_{cur_num_exercises + cur_batch_size}.txt")
            gen_res = _run_agent(gen_agent, deps=get_generator_deps(topic_name, cur_batch_size, vocab_level, ex_filename))
            if gen_res is None:
                continue  # try again
            if verbose:
                console.print(f"[{gen_res.timestamp}] Generator usage: {gen_res.usage}")
            if debug:
                shutil.copy(ex_filename, os.path.split(ex_filename)[-1])
            if eval_agent is not None and get_num_lines(ex_filename) > 0:
                # get result from the run agent, eval sentences one by one, and save good lines to the new file
                eval_filename = ex_filename.replace(".txt", "_eval.txt")
                for line in read_all_lines(ex_filename):
                    eval_res = _run_agent(eval_agent, deps=get_evaluator_deps(topic_name, vocab_level, example=line, out_filename=eval_filename))
                    if verbose and eval_res is not None:
                        console.print(f"[{eval_res.timestamp}] Evaluator usage: {eval_res.usage}")
                if debug:
                    shutil.copy(eval_filename, os.path.split(eval_filename)[-1])
            else:
                eval_filename = ex_filename

            num_copied_lines = copy_lines(eval_filename, out_path)
            cur_num_exercises += num_copied_lines
            loop_count += 1

        if (loop_count > max_loops) and (cur_num_exercises < num):
            console.print(f"[red]Failed to generate the required number of exercises after {max_loops} attempts. Current number of exercises: {cur_num_exercises}.[/red]")

    return read_all_lines(out_path)


def get_generator_deps(
    topic_name: str,
    num: int,
    vocab_level: str | None,
    out_filename: str,
) -> GeneratorDeps:
    return GeneratorDeps(
        topic_name=topic_name,
        num_exercises=num,
        vocab_level=vocab_level,
        out_filename=out_filename,
    )


def get_evaluator_deps(
    topic_name: str,
    vocab_level: str | None,
    example: str,
    out_filename: str,
) -> EvaluatorDeps:
    return EvaluatorDeps(
        topic_name=topic_name,
        example=example,
        vocab_level=vocab_level,
        out_filename=out_filename,
    )


def _get_evaluator_prompt(
    topic_name: str,
    vocab_level: str | None,
    input_filename: str,
    out_filename: str,
) -> str:
    vocab_level = vocab_level or "A2"
    prompt = f"Evaluate examples for the provided Korean grammar topic \"{topic_name}\" for the vocabulary matching CEFR level {vocab_level}. The examples are provided in the input file \"{input_filename}\", each line contains exactly 1 example. Write selected examples to the output file \"{out_filename}\"."
    return prompt
