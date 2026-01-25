from __future__ import annotations

import random

import dotenv
import fire
import numpy as np
import sys
import toml

from hmeg.entities import ExerciseGenerationEngine
from hmeg import usecases as uc, ExerciseGenerator, GrammarChecker, GrammarRegistry, Reranker, Vocabulary

dotenv.load_dotenv()


class Runner:
    def _verify_engine_configuration(self):
        """
        Checks that the selected engine has all required parameters set after the initialization.
        Raises RuntimeError if some required parameter is missing.

        Returns
        -------
        bool
            True if the configuration is valid. False otherwise.
        """

        if self.engine == ExerciseGenerationEngine.OLLAMA:
            if "miniphrase" in self.topic.lower():
                raise RuntimeError("The 'miniphrase' topic is not supported with the Ollama engine.")

            return uc.is_ollama_available(self.model)

        return True

    def _configure_from_file(self, config_path: str | None):
        """
        Load configuration from `config_path` (or default) and initialize attributes.
        Can be called from __init__ or from run(...) when a custom config file is supplied.
        """
        self.config_file = config_path or "hmeg.conf"

        try:
            with open(self.config_file, mode="r") as f:
                run_config = toml.loads(f.read())
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file `{self.config_file}` not found.") from e

        self.engine = run_config.get("engine", ExerciseGenerationEngine.TEMPLATES)

        topics_folder = run_config.get("topics_folder")
        if not topics_folder:
            raise KeyError("`topics_folder` missing in config.")
        uc.register_grammar_topics(topics_folder)

        vocab_file = run_config.get("vocab_file")
        if self.engine == ExerciseGenerationEngine.TEMPLATES and not vocab_file:
            raise KeyError(f"`vocab_file` parameter is required for the \"{ExerciseGenerationEngine.TEMPLATES}\" engine.")
        self.vocab = Vocabulary.load(vocab_file) if vocab_file is not None else None

        self.topic = run_config.get("topic")
        if not self.topic:
            # keep existing topic if already set by constructor args; otherwise error
            if getattr(self, "topic", None) is None:
                raise KeyError("`topic` missing in config and not provided as argument.")

        configured_num = run_config.get("number_exercises", 10)
        try:
            configured_num = int(configured_num)
        except (ValueError, TypeError):
            configured_num = 10
        self.num_exercises = max(5, min(configured_num, 100))

        self.model = run_config.get("model")
        self.vocab_level = run_config.get("vocab_level")
        self.grammar_correction_model = run_config.get("grammar_correction")
        if self.grammar_correction_model is not None:
            Reranker.set_current_model(self.grammar_correction_model)

        if not self._verify_engine_configuration():
            raise RuntimeError("Please check configuration of the exercise generation engine.")

    def __init__(self, config_path: str | None = None, topic: str | None = None, n: int = 0):
        """
        Supported commands:
        * run
        * list

        Parameters
        ----------
        config_path: str, default=None
            Path to the configuration file. If not provided then "hmeg.conf" is used.
        topic: str, default=None
            Name of the topic to generate exercises for. Can override topic from `config`
        n: int, default=0
            Number of exercises. Can override number of exercises defined in `config`.
        """
        self._configure_from_file(config_path)

    def list(self):
        """
        Prints list of registered topics.
        """
        topics = GrammarRegistry.get_registered_topics()
        print("\n".join(topics))

    def run(self, config: str | None = None):
        """
        Runs generation of exercises and prints them on the screen.

        Parameters
        ----------
        config: str, default=None
            Path to the configuration file. If not provided then configuration file used during
            initialization of the Runner instance is used.
        """

        if config is not None:
            self._configure_from_file(config)

        topics = GrammarRegistry.find_topics(self.topic)
        if len(topics) == 0:
            print(f"Requested an unregistered topic: {self.topic}. Please run `python hmeg_cli.py list` to see the existing topics.")
            return
        elif len(topics) == 1:
            print(f"Exercises for topic: {topics[0]}")
        elif len(topics) > 1:
            print(f"Exercises for topics:")
            for topic in topics:
                print(f"\t{topic}")

        exercises = []
        attempts = 0
        num_exercises_per_topic = max(1, self.num_exercises // len(topics))
        while len(exercises) < self.num_exercises:
            cur_topic = np.random.choice(topics)
            cur_topic_num_exercises = min(num_exercises_per_topic, self.num_exercises - len(exercises))
            cur_topic_exercises = ExerciseGenerator.generate_exercises(
                topic_name=cur_topic,
                num=cur_topic_num_exercises,
                vocab=self.vocab,
                vocab_level=self.vocab_level,
                engine=self.engine,
                model=self.model
            )
            for cur_exercise in cur_topic_exercises:
                if cur_exercise not in exercises:
                    exercises.append(cur_exercise)
            attempts += cur_topic_num_exercises
            if attempts > self.num_exercises ** 2:
                break

        if self.engine == ExerciseGenerationEngine.TEMPLATES and self.grammar_correction_model is not None:
            print(f"Using grammar correction model: {self.grammar_correction_model}")
            exercises = GrammarChecker.correct_phrases(exercises, vocab=self.vocab)

        random.shuffle(exercises)
        for idx, exercise in enumerate(exercises):
            print(f"{idx + 1}. {exercise}")


if __name__ == "__main__":
    if len(sys.argv) == 1:  # no arguments
        Runner().run()
    else:
        fire.Fire(Runner)
