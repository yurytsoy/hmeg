import logfire

logfire.configure(send_to_logfire="always", console=False)
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

import asyncio
from dataclasses import dataclass
import os
import shutil
import tempfile
from typing import Any

from pydantic_ai import Agent, AgentRunResult, exceptions
from rich.console import Console

from .usecases import make_generator_agent, make_evaluator_agent, get_num_lines, read_all_lines, copy_lines, GeneratorDeps, EvaluatorDeps, write_line

OLLAMA_NUM_PARALLEL = 4
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


async def _run_agent_async(sem: asyncio.Semaphore, agent: Agent, prompt: str | None = None, deps: type[dataclass] | None = None):
    async with sem: # Dynamically limits active requests
        response = await asyncio.to_thread(_run_agent, **dict(agent=agent, prompt=prompt, deps=deps))
        return response


async def evaluate_file(eval_agent: Agent, ex_filename: str, topic_name: str, vocab_level: str, ref_examples: list[str], out_path: str, verbose: bool = False) -> int:
    sem = asyncio.Semaphore(OLLAMA_NUM_PARALLEL)

    lines = read_all_lines(ex_filename)
    results = await asyncio.gather(
        *[_run_agent_async(sem=sem, agent=eval_agent, deps=get_evaluator_deps(topic_name, vocab_level, example=line, ref_examples=ref_examples))
          for line in lines]
    )
    if verbose:
        console.print(f"Evaluation results gathered ({len(results)})")
    valid_count = 0
    for eval_res, line in zip(results, lines):
        if eval_res is None:
            continue

        if eval_res.output.is_valid:
            valid_count += 1
            write_line(None, out_path, line)
        if verbose:
            console.print(f"[{eval_res.timestamp}] Evaluator usage: {eval_res.usage}")
    return valid_count


def generate_exercises(
    topic_name: str,
    num: int,
    gen_model: str | None = None,
    eval_model: str | None = None,
    vocab_level: str | None = None,
    examples: list[str] | None = None,
    out_path: str | None = None,
    verbose: bool = False,
    debug: bool = False
) -> list[str]:
    out_path = out_path or "result.txt"  # TODO: include sanitized topic name into the filename
    batch_size = min(num, 10)

    gen_agent = make_generator_agent(model_name=gen_model)
    eval_agent = make_evaluator_agent(model_name=eval_model)
    max_loops = max(int(1.41 * num) // batch_size, num // batch_size + 1)  # include some possibility for the failed attempts
    loop_count = 0
    cur_num_exercises = 0

    tmp_dir = tempfile.mkdtemp()
    try:
        while (cur_num_exercises < num) and (loop_count < max_loops):
            cur_batch_size = min(batch_size, num - cur_num_exercises)

            # ? TODO: add list of nouns and verbs to avoid or use dynamic instructions
            ex_filename = os.path.join(tmp_dir, f"ex_{cur_num_exercises}_{cur_num_exercises + cur_batch_size}.txt")
            gen_deps = get_generator_deps(topic_name, cur_batch_size, vocab_level=vocab_level, out_filename=ex_filename, examples=examples)
            gen_res = _run_agent(gen_agent, deps=gen_deps)
            if gen_res is None:
                continue  # try again
            if verbose:
                console.print(f"[{gen_res.timestamp}] Generator usage: {gen_res.usage}")
            if debug and os.path.exists(ex_filename):
                shutil.copy(ex_filename, os.path.split(ex_filename)[-1])
            if eval_agent is not None and get_num_lines(ex_filename) > 0:
                # get result from the run agent, eval sentences one by one, and save good lines to the new file

                eval_filename = ex_filename.replace(".txt", "_eval.txt")
                asyncio.run(
                    evaluate_file(
                        eval_agent=eval_agent,
                        ex_filename=ex_filename,
                        topic_name=topic_name,
                        vocab_level=vocab_level,
                        ref_examples=examples or [],
                        out_path=eval_filename,
                        verbose=verbose,
                    )
                )
                if debug and os.path.exists(eval_filename):
                    shutil.copy(eval_filename, os.path.split(eval_filename)[-1])
            else:
                eval_filename = ex_filename

            num_copied_lines = copy_lines(eval_filename, out_path)
            cur_num_exercises += num_copied_lines
            loop_count += 1

        if (loop_count > max_loops) and (cur_num_exercises < num):
            console.print(f"[red]Failed to generate the required number of exercises after {max_loops} attempts. Current number of exercises: {cur_num_exercises}.[/red]")
    finally:
        shutil.rmtree(tmp_dir)

    return read_all_lines(out_path)


def get_generator_deps(
    topic_name: str,
    num: int,
    vocab_level: str | None,
    out_filename: str,
    examples: list[str] | None = None,
) -> GeneratorDeps:
    return GeneratorDeps(
        topic_name=topic_name,
        num_exercises=num,
        vocab_level=vocab_level,
        out_filename=out_filename,
        examples=examples or []
    )


def get_evaluator_deps(
    topic_name: str,
    vocab_level: str | None,
    example: str,
    ref_examples: list[str]
) -> EvaluatorDeps:
    return EvaluatorDeps(
        topic_name=topic_name,
        example=example,
        vocab_level=vocab_level,
        ref_examples=ref_examples
    )
