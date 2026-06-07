from collections.abc import Callable
import os

from grpc._cython.cygrpc import BaseCompletionQueue
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelSettings
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.ollama import OllamaModel

from .entities import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K


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


def copy_lines(src_filepath: str, dst_filepath: str) -> int:
    if not os.path.exists(src_filepath):
        return 0

    lines = read_all_lines(src_filepath)
    with open(dst_filepath, "a", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")
    return len(lines)


def write_line(ctx: RunContext, filepath: str, line: str) -> None:
    """
    Add a line to a file. If file does not exist, then it is created.
    """
    # print(f"Writing {line} to {filepath}")
    with open(filepath, "a", encoding="utf-8") as file:
        file.write(line + "\n" if not line.endswith("\n") else line)


def make_orchestrator_agent(
    model_name: str,
    system_prompt: str,
) -> Agent:
    prompt = """
You are an Orchestrator Agent. Your goal is to solve the user's request by breaking it down into steps and using basic tools.
1. Analyze the request.
2. Formulate a step-by-step plan. Each step should be simple enough to be handled by a small language model with 4k tokens context size.
3. Execute the steps on by one.
4. If you absolutely cannot solve it with available tools, explain exactly what basic tools are missing.
    """

    return Agent(
        OllamaModel(model_name=model_name),
        system_prompt=system_prompt,
        tools=[],  # TODO
        model_settings=ModelSettings(temperature=0.0)
    )


def make_generator_agent(model_name: str | None) -> Agent:
    from hmeg.prompt_loader import PromptLoader

    class GeneratorOutput(BaseModel):
        result_num: int = Field(description="Number of generated Korean phrases.")

    prompt_loader_ = PromptLoader()
    exercise_prompt = prompt_loader_.load("v2/generator/text_kr")
    return make_agent(
        model_name=model_name or exercise_prompt.llm.model,
        system_prompt=exercise_prompt.system_instructions,
        tools=[write_line],
        output_type=GeneratorOutput,
        max_tokens=exercise_prompt.llm.max_tokens,
        top_k=exercise_prompt.llm.top_k,
        top_p=exercise_prompt.llm.top_p,
        temperature=exercise_prompt.llm.temperature,
    )


def make_evaluator_agent(model_name: str | None) -> Agent:
    from hmeg.prompt_loader import PromptLoader

    class EvaluatorOutput(BaseModel):
        result: bool = Field(description="Whether provided example passed (True) or not (False).")

    prompt_loader_ = PromptLoader()
    exercise_prompt = prompt_loader_.load("v2/evaluator/evaluator")
    return make_agent(
        model_name=model_name or exercise_prompt.llm.model,
        system_prompt=exercise_prompt.system_instructions,
        tools=[write_line],
        output_type=EvaluatorOutput,
        max_tokens=exercise_prompt.llm.max_tokens,
        top_k=exercise_prompt.llm.top_k,
        top_p=exercise_prompt.llm.top_p,
        temperature=exercise_prompt.llm.temperature,
    )


def make_agent(
    model_name: str,
    system_prompt: str,
    tools: list[Callable] | None = None,
    output_type: type[BaseModel] | None = None,
    max_tokens: int | None = 4096,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
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
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, thinking="medium",
            extra_body={"options": {"num_ctx": max_tokens}}
        ),
        output_type=output_type,
        end_strategy='graceful'
    )
