import random

import json
from nltk.parse.generate import generate
from nltk import CFG
from ollama import chat
import os

from hmeg.entities import ExerciseGenerationEngine, TranslationExercise
from hmeg.grammar_registry import GrammarRegistry
from hmeg.prompt_loader import PromptLoader
from hmeg.usecases import apply_vocabulary, parse_completion
from hmeg.vocabulary import Vocabulary


cur_dir = os.path.split(os.path.realpath(__file__))[0]
DEFAULT_VOCABULARY_FILE = os.path.join(cur_dir, "vocabs/minilex.toml")


class ExerciseGenerator:
    @staticmethod
    def generate_exercises(
        topic_name: str,
        num: int,
        vocab: Vocabulary | None = None,
        engine: str | None = None,
        model: str | None = None,
        vocab_level: str | None = None
    ) -> list[str]:
        """
        Generate a list of exercises for a given grammar topic.

        The function dispatches to the configured generation engine:
        - `ExerciseGenerationEngine.TEMPLATES`: use local template-based generation.
        - `ExerciseGenerationEngine.OLLAMA`: use the Ollama LLM-based generator.

        Parameters
        ----------
        topic_name: str
            The name of the topic to generate exercises for.
        num: int
            Number of exercises to produce.
        vocab: Vocabulary, default=None
            Optional Vocabulary instance used by the template engine. If `None`, a default
            vocabulary file is loaded when using the template engine.
        engine: str, default=None
            Optional engine selector. If `None`, defaults to `ExerciseGenerationEngine.TEMPLATES`.
        model: str, default=None
            Optional model identifier used only by the LLM (OLLAMA) engine.
        vocab_level: str, default=None
            Optional vocabulary level (CEFR) string used only by the LLM (OLLAMA) engine.

        Returns
        -------
        list[str]
            A list of generated exercise sentences (English strings).

        Raises
        ------
        RuntimeError
            If an unknown engine is provided.
        """

        engine = engine or ExerciseGenerationEngine.TEMPLATES
        if engine == ExerciseGenerationEngine.TEMPLATES:
            return ExerciseGenerator.generate_exercises_templates(topic_name, num, vocab)
        elif engine == ExerciseGenerationEngine.OLLAMA:
            ress = ExerciseGenerator.generate_exercises_ollama(topic_name, num, model=model, vocab_level=vocab_level)
            return [res.sentence_en for res in ress]
        raise RuntimeError(f"Unknown exercise generation engine: {engine}")

    @staticmethod
    def generate_exercises_templates(topic_name: str, num: int, vocab: Vocabulary | None = None) -> list[str]:
        """
        Generates list of random translation exercises for the given topic using legacy approach (pre-LLM).
        The generation proceeds in 2 steps:
        1. Generate list of templates wrt selected grammar topic. The result contains
           placeholders for nouns, verbs, ....
        2. Fill-in placeholders according to the given vocabulary.

        See also: `apply_vocabulary`

        Parameters
        ----------
        topic_name: str
            The name of the topic to generate exercises for.
        num: int
            The number of exercises to generate.
        vocab: Vocabulary, default=None
            Vocabulary for words, that can be used for generating exercises.
            If `None` then vocabulary from the `DEFAULT_VOCABULARY_FILE` is used.

        Returns
        -------
        list[str]
            A list of generated exercise sentences (English strings).
        """
        vocab = vocab or Vocabulary.load(DEFAULT_VOCABULARY_FILE)

        if topic_name not in GrammarRegistry.topics:
            raise RuntimeError(f"Requested an unregistered topic: {topic_name}. Please run `python hmeg_cli.py list` to see the existing topics.")

        templates = []
        for exercise_type in GrammarRegistry.topics[topic_name].exercises:
            cur_grammar = CFG.fromstring(exercise_type)
            templates.extend(generate(cur_grammar, n=num))

        res = []
        num_trials = 0
        while len(res) < num:
            cur_idx = random.randint(0, len(templates)-1)
            exercise = apply_vocabulary(" ".join(templates[cur_idx]), vocab)
            exercise = exercise.replace(exercise[0], exercise[0].capitalize(), 1)
            if exercise not in res:
                res.append(exercise)
            num_trials += 1
            if num_trials > num ** 2:
                break
        return res

    @staticmethod
    def generate_exercises_ollama(
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

        def extract_topic_name(topic_name: str) -> str:
            # if topic name is like "Topic Group / Topic Name", extract "Topic Name"
            sep = " / "
            return topic_name.split(sep)[-1] if sep in topic_name else topic_name

        def prepare_generation_params() -> dict[str, str | dict | list]:
            res: dict[str, str | dict | list] = {
                "grammar_topic": extract_topic_name(topic_name), "number_of_exercises": num
            }
            res["vocabulary_level"] = json.dumps(vocab_level or "B1", ensure_ascii=False)
            return res

        prompt_loader_ = PromptLoader()
        exercise_prompt = prompt_loader_.load("v1/generator/text_kr")
        model = model or exercise_prompt.llm.model
        prompt_params = prepare_generation_params()
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
            prompt_params = {"sentences_kr": [ex.sentence_kr for ex in res_exercises]}
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
