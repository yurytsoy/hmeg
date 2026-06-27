from collections.abc import Callable
import os
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelSettings
from pydantic_ai.settings import ThinkingEffort
from pydantic_ai.models.ollama import OllamaModel

from .entities import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_VOCAB_LEVEL


def get_num_lines(filepath: str) -> int:
    if not os.path.exists(filepath):
        return 0

    with open(filepath, "r", encoding="utf-8") as file:
        line_count = sum(1 for line in file)
    return line_count


def read_all_lines(filepath: str) -> list[str]:
    """
    Read all lines from the given file.
    """
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as file:
        res = [line.strip() for line in file]
    return res


def copy_lines(src_filepath: str, dst_filepath: str, only_unique: bool = True) -> int:
    if not os.path.exists(src_filepath):
        return 0

    dst_lines = read_all_lines(dst_filepath)
    lines = read_all_lines(src_filepath)
    with open(dst_filepath, "a", encoding="utf-8") as file:
        for line in lines:
            if only_unique and (line in dst_lines):
                continue
            file.write(line + "\n")
    return len(lines)


def write_line(ctx: RunContext, filepath: str, line: str) -> None:
    """
    Add a line to a file. If file does not exist, then it is created.
    """
    # print(f"Writing {line} to {filepath}")
    with open(filepath, "a", encoding="utf-8") as file:
        file.write(line + "\n" if not line.endswith("\n") else line)


def write_lines(ctx: RunContext, filepath: str, lines: list[str]) -> None:
    """
    Adds multiple lines to a file. If file does not exist, then it is created.
    """
    if not lines:
        return

    with open(filepath, "a", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n" if not line.endswith("\n") else line)


@dataclass
class GeneratorDeps:
    num_exercises: int
    topic_name: str
    vocab_level: str | None
    out_filename: str
    examples: list[str]


class GeneratorOutput(BaseModel):
    result_num: int = Field(description="Number of generated Korean phrases.")


def generator_instructions(ctx: RunContext[GeneratorDeps]) -> str:
    deps = ctx.deps
    vocab_level = deps.vocab_level or DEFAULT_VOCAB_LEVEL
    res = f"Generate {deps.num_exercises} exercises for the Korean grammar topic \"{deps.topic_name}\" using the vocabulary matching CEFR level {vocab_level}. Write resulting exercises into the file '{deps.out_filename}', each exercise in a separate line."

    examples = ""
    if deps.examples:
        examples = "\nExamples of valid exercises:"
        for example in deps.examples:
            examples += f"\n- {example}"
        examples += "\n**IMPORTANT:** **Never** copy examples!"
    res += examples
    return res


def make_generator_agent(model_name: str | None) -> Agent:
    from hmeg.prompt_loader import PromptLoader

    prompt_loader_ = PromptLoader()
    exercise_prompt = prompt_loader_.load("v2/generator/text_kr")
    return make_agent(
        model_name=model_name or exercise_prompt.llm.model,
        system_prompt=exercise_prompt.system_instructions,
        tools=[write_lines],
        deps_type=GeneratorDeps,
        output_type=GeneratorOutput,
        instructions=generator_instructions,
        max_tokens=exercise_prompt.llm.max_tokens,
        top_k=exercise_prompt.llm.top_k,
        top_p=exercise_prompt.llm.top_p,
        temperature=exercise_prompt.llm.temperature,
    )


@dataclass
class EvaluatorDeps:
    topic_name: str
    vocab_level: str | None
    example: str
    ref_examples: list[str]


def evaluator_instructions(ctx: RunContext[EvaluatorDeps]) -> str:
    deps = ctx.deps
    vocab_level = deps.vocab_level or DEFAULT_VOCAB_LEVEL
    res = f"Evaluate sentence: '{deps.example}' against Topic: '{deps.topic_name}' at CEFR: {vocab_level}."

    ref_examples = ""
    if deps.ref_examples:
        ref_examples = "\nReference valid sentences:"
        for ref_ex in deps.ref_examples:
            ref_examples += f"\n- {ref_ex}"
    res += ref_examples
    return res


class EvaluatorOutput(BaseModel):
    is_valid: bool = Field(description="Whether the input sentence is valid or not.")


def make_evaluator_agent(model_name: str | None) -> Agent:
    from hmeg.prompt_loader import PromptLoader

    prompt_loader_ = PromptLoader()
    exercise_prompt = prompt_loader_.load("v2/evaluator/evaluator")
    model_name = model_name or exercise_prompt.llm.model
    return make_agent(
        model_name=model_name,
        system_prompt=exercise_prompt.system_instructions,
        deps_type=EvaluatorDeps,
        output_type=EvaluatorOutput,
        instructions=evaluator_instructions,
        max_tokens=exercise_prompt.llm.max_tokens,
        top_k=exercise_prompt.llm.top_k,
        top_p=exercise_prompt.llm.top_p,
        temperature=exercise_prompt.llm.temperature,
        thinking="low" if ("qwen" in model_name.lower()) else False
    )


def make_agent(
    model_name: str,
    system_prompt: str,
    tools: list[Callable] | None = None,
    deps_type: type[dataclass] | None = None,
    output_type: type[BaseModel] | None = None,
    instructions: Callable | None = None,
    max_tokens: int | None = 4096,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    thinking: ThinkingEffort | bool = "medium"
) -> Agent:
    print(f"Creating an agent based on {model_name} ({max_tokens=}, {top_k=}, {top_p=}, {temperature=}).")
    max_tokens = max_tokens or DEFAULT_MAX_TOKENS
    temperature = temperature or DEFAULT_TEMPERATURE
    top_p = top_p or DEFAULT_TOP_P
    top_k = top_k or DEFAULT_TOP_K
    return Agent(
        OllamaModel(model_name=model_name),
        system_prompt=system_prompt,
        tools=tools or [],
        model_settings=ModelSettings(
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, thinking=thinking,
            extra_body={"options": {"num_ctx": max_tokens}}
        ),
        instructions=instructions,
        deps_type=deps_type,
        output_type=output_type,
        end_strategy='graceful'
    )
