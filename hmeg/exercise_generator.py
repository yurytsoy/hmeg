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
        topic_name: str, num: int, vocab: Vocabulary | None = None, engine: str | None = None, model: str | None = None
    ) -> list[str]:
        engine = engine or ExerciseGenerationEngine.TEMPLATES
        if engine == ExerciseGenerationEngine.TEMPLATES:
            return ExerciseGenerator.generate_exercises_templates(topic_name, num, vocab)
        elif engine == ExerciseGenerationEngine.OLLAMA:
            ress = ExerciseGenerator.generate_exercises_ollama(topic_name, num, model=model)
            return [res.sentence_en for res in ress]
        raise RuntimeError(f"Unknown exercise generation engine: {engine}")

    @staticmethod
    def generate_exercises_templates(topic_name: str, num: int, vocab: Vocabulary | None = None):
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
        topic_name: str, num: int, model: str | None = None, vocab_levels: list[str] | None = None
    ) -> list[TranslationExercise]:
        def recommend_local_model() -> str:
            """
            Recommends an Ollama model based on the available GPU memory.
            The models are from the Gemma3 family.
            - gemma3:270m for CPU
            - gemma3:4b for  <12GB GPU
            - gemma3:12b for 12-24GB GPU
            - gemma3:27b for >24GB GPU
            """

            import torch

            if not torch.cuda.is_available():
                return "gemma3:270m"
            else:
                props = torch.cuda.get_device_properties(0)
                total_gbs = props.total_memory // 2**30  # memory in GB
                if total_gbs < 12:
                    return "gemma3:4b"
                elif total_gbs < 24:
                    return "gemma3:12b"
                return "gemma3:27b"

        def extract_topic_name(topic_name: str) -> str:
            # if topic name is like "Topic Group / Topic Name", extract "Topic Name"
            sep = " / "
            return topic_name.split(sep)[-1] if sep in topic_name else topic_name

        def prepare_generation_params() -> dict[str, str | dict | list]:
            res: dict[str, str | dict | list] = {
                "grammar_topic": extract_topic_name(topic_name), "number_of_exercises": num
            }
            res["vocabulary_levels"] = json.dumps(vocab_levels or ["A1", "A2", "B1"], ensure_ascii=False)
            return res

        prompt_loader_ = PromptLoader()
        exercise_prompt = prompt_loader_.load("v1/generator/text_kr")
        model = model or exercise_prompt.llm.model or recommend_local_model()
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
            assert len(res_exercises) == len(result_en)
            for idx, res_dict in enumerate(result_en):
                res_exercises[idx].sentence_en = res_dict.get("sentence_en")

        return res_exercises
