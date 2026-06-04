import os
import shutil
import tempfile

from rich.console import Console

from .usecases import make_generator_agent, make_evaluator_agent, get_num_lines, read_all_lines, copy_lines

DEFAULT_MODEL = "gemma4:e4b"

console = Console()


def generate_exercises(
    topic_name: str, num: int, model: str | None = None, vocab_level: str | None = None, out_path: str | None = None, verbose: bool = False, debug: bool = False
) -> list[str]:
    out_path = out_path or "result.txt"  # TODO: include sanitized topic name into the filename
    batch_size = min(num, 20)

    gen_agent = make_generator_agent(model_name=model or DEFAULT_MODEL)
    eval_agent = make_evaluator_agent(model_name=model or DEFAULT_MODEL)
    max_loops = int(1.41 * num) // batch_size  # include some possibility for the failed attempts
    loop_count = 0
    cur_num_exercises = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        while cur_num_exercises < num:
            cur_batch_size = min(batch_size, num - cur_num_exercises)

            # ? TODO: add list of nouns and verbs to avoid or use dynamic instructions
            ex_filename = os.path.join(tmp_dir, f"ex_{cur_num_exercises}_{cur_num_exercises + cur_batch_size}.txt")
            gen_res = gen_agent.run_sync(get_generator_prompt(topic_name, cur_batch_size, vocab_level, ex_filename))
            if verbose:
                console.print(f"[{gen_res.timestamp}] Generator usage: {gen_res.usage}")
            if debug:
                shutil.copy(ex_filename, os.path.split(ex_filename)[-1])
            if eval_agent is not None:
                # get result from the run agent, eval sentences one by one, and save good lines to the new file
                eval_filename = ex_filename.replace(".txt", "_eval.txt")
                eval_res = eval_agent.run_sync(get_evaluator_prompt(topic_name, vocab_level, ex_filename, eval_filename))
                if verbose:
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
                break

    return read_all_lines(out_path)


def get_generator_prompt(
    topic_name: str,
    num: int,
    vocab_level: str | None,
    out_filename: str,
) -> str:
    vocab_level = vocab_level or "A2"
    prompt = f"Generate {num} exercises for the topic \"{topic_name}\" for the vocabulary matching CEFR level {vocab_level}. Write resulting exercises into the file '{out_filename}', each exercise in a separate line."
    return prompt


def get_evaluator_prompt(
    topic_name: str,
    vocab_level: str | None,
    input_filename: str,
    out_filename: str,
) -> str:
    vocab_level = vocab_level or "A2"
    prompt = f"Evaluate examples for the provided grammar topic \"{topic_name}\" for the vocabulary matching CEFR level {vocab_level}. The examples are provided in the input file \"{input_filename}\", each line contains exactly 1 example. Select sentences that match the grammar and vocabulary level and write them to the output file \"{out_filename}\"."
    return prompt
