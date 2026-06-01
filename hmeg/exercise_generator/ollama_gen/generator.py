import json

from hmeg.entities import TranslationExercise


def generate_exercises(
    topic_name: str, num: int, model: str | None = None, vocab_level: str | None = None
) -> list[TranslationExercise]:
    """
    Generate exercises by calling an Ollama LLM prompt and parsing its structured output.

    The function:
    1. Normalizes the provided topic name (supports `"Group / Topic"` style names).
    2. Prepares prompt parameters (including `vocabulary_level`).
    3. Calls `ollama.chat` with the loaded prompt and model.
    4. Parses the completion using `parse_completion` and converts results into
       `TranslationExercise` objects.
    5. If the returned exercises lack English translations, it calls a translation
       prompt to obtain `sentence_en` for each exercise.

    Parameters
    ----------
    topic_name: str
        Topic name (may include a group prefix separated by `" / "`).
    num: int
        Number of exercises to request from the model.
    model: str, default=None
        Ollama model identifier. If `None`, the prompt's default model is used.
    vocab_level: str | None, default=None
        Preferred vocabulary level (e.g. `"A1"`, `"B1"`). If `None`, defaults to `"B1"`.

    Returns
    -------
    list[TranslationExercise]
        A list of `TranslationExercise` instances containing `sentence_kr` and
        `sentence_en` (the latter may be filled via a secondary translation step).

    Raises
    ------
    RuntimeError
        If the model response cannot be parsed into the expected list structure.
    AssertionError
        If the translation step returns a different number of translations than
        the number of generated exercises.

    Notes
    -----
    - The function relies on `PromptLoader` to provide system/user instructions and
      output schema used for structured completion parsing.
    - `vocabulary_level` is JSON-encoded before being passed to the prompt rendering.
    """
    from ollama import chat

    from hmeg.prompt_loader import PromptLoader
    from hmeg.usecases import parse_completion

    prompt_loader_ = PromptLoader()
    exercise_prompt = prompt_loader_.load("v1/generator/text_kr")
    model = model or exercise_prompt.llm.model
    prompt_params = prepare_generation_params(topic_name=topic_name, num=num, vocab_level=vocab_level)
    response = chat(
        model=model,
        format=exercise_prompt.output_schema,
        messages=[
            {"role": "system", "content": exercise_prompt.system_instructions},
            {"role": "user", "content": exercise_prompt.render_user_prompt(**prompt_params)}
        ],
        options={'temperature': exercise_prompt.llm.temperature},
    )

    # parse the result.
    result = parse_completion(response.message.content).get("results")
    if not isinstance(result, list):
        raise RuntimeError(f"Failed to parse the response from the model: {response.message.content}")

    res_exercises = [TranslationExercise(**{"sentence_kr": res_dict["phrase_kr"]}) for res_dict in result]
    if not res_exercises:
        return res_exercises

    if res_exercises[0].sentence_en is None:  # need to translate from Korean to English
        trans_prompt = prompt_loader_.load("v1/translator/translate_kr_en")
        prompt_params = {"sentences_kr": json.dumps([ex.sentence_kr for ex in res_exercises], ensure_ascii=False)}
        response = chat(
            model=trans_prompt.llm.model,
            format=trans_prompt.output_schema,
            messages=[
                {"role": "system", "content": trans_prompt.system_instructions},
                {"role": "user", "content": trans_prompt.render_user_prompt(**prompt_params)}
            ],
            options={'temperature': trans_prompt.llm.temperature},
        )
        result_en = parse_completion(response.message.content).get("results")
        if len(res_exercises) != len(result_en):
            raise RuntimeError(f"Translation count mismatch: expected {len(res_exercises)}, got {len(result_en)}")

        for idx, res_dict in enumerate(result_en):
            res_exercises[idx].sentence_en = res_dict.get("sentence_en")

    return res_exercises


def extract_topic_name(topic_name: str) -> str:
    # if topic name is like "Topic Group / Topic Name", extract "Topic Name"
    sep = " / "
    return topic_name.split(sep)[-1] if sep in topic_name else topic_name


def prepare_generation_params(topic_name: str, num: int, vocab_level: str | None = None) -> dict[str, str | dict | list]:
    res: dict[str, str | dict | list] = {
        "grammar_topic": extract_topic_name(topic_name), "number_of_exercises": num
    }
    res["vocabulary_level"] = json.dumps(vocab_level or "B1", ensure_ascii=False)
    return res

