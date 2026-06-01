import random

import json
import os

from hmeg.entities import ExerciseGenerationEngine, TranslationExercise
# from hmeg.grammar_registry import GrammarRegistry
# from hmeg.prompt_loader import PromptLoader
# from hmeg.usecases import apply_vocabulary, parse_completion
from hmeg.vocabulary import Vocabulary

# cur_dir = os.path.split(os.path.realpath(__file__))[0]
# DEFAULT_VOCABULARY_FILE = os.path.join(cur_dir, "vocabs/minilex.toml")


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
            from .templates_gen import generate_exercises
            return generate_exercises(topic_name, num, vocab)

        elif engine == ExerciseGenerationEngine.OLLAMA:
            from .ollama_gen import generate_exercises

            ress = generate_exercises(topic_name, num, model=model, vocab_level=vocab_level)
            return [res.sentence_en for res in ress]
        raise RuntimeError(f"Unknown exercise generation engine: {engine}")
