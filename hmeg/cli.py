from __future__ import annotations

import random

import dotenv
import fire
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

    def _configure_from_file(self, config_path: str | None, topic: str | None = None, num: int = 0):
        """
        Load configuration from `config_path` (or default) and initialize attributes.
        Can be called from __init__ or from run(...) when a custom config file is supplied.
        """
        def get_topics_folder_from_config():
            raw_topics_folder = run_config.get("topics_folder")
            if raw_topics_folder is None:
                raise KeyError("`topics_folder` missing in config.")

            result = uc.to_abs_path(raw_topics_folder)
            if not result:
                raise KeyError("`topics_folder` is empty or invalid in config.")
            return result

        def get_vocabulary_from_config():
            raw_vocab_file = run_config.get("vocab_file")
            vocab_file = uc.to_abs_path(raw_vocab_file) if raw_vocab_file else raw_vocab_file

            if self.engine == ExerciseGenerationEngine.TEMPLATES:
                if "vocab_file" not in run_config:
                    raise KeyError(
                        f"`vocab_file` parameter is required for the \"{ExerciseGenerationEngine.TEMPLATES}\" engine.")
                if not vocab_file:
                    raise ValueError(
                        f"`vocab_file` in config must not be empty for the \"{ExerciseGenerationEngine.TEMPLATES}\" engine.")
            return Vocabulary.load(vocab_file) if vocab_file else None

        def get_num_exercises():
            num_exercises = num or run_config.get("number_exercises", 10)
            try:
                num_exercises = int(num_exercises)
            except (ValueError, TypeError):
                num_exercises = 10
            return max(5, min(num_exercises, 100))

        self.config_file = config_path or "hmeg.conf"

        try:
            with open(self.config_file, mode="r") as f:
                run_config = toml.loads(f.read())
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file `{self.config_file}` not found.") from e

        self.engine = run_config.get("engine", ExerciseGenerationEngine.TEMPLATES)

        # register descriptions of grammar topics
        topics_folder = get_topics_folder_from_config()
        uc.register_grammar_topics(topics_folder)

        self.topic = topic or run_config.get("topic")
        if not self.topic:
            # keep existing topic if already set by constructor args; otherwise error
            if getattr(self, "topic", None) is None:
                raise KeyError("`topic` missing in config and not provided as argument.")

        self.vocab = get_vocabulary_from_config()
        self.vocab_level = run_config.get("vocab_level")

        self.num_exercises = get_num_exercises()

        self.grammar_correction_model = run_config.get("grammar_correction")
        if self.grammar_correction_model is not None:
            Reranker.set_current_model(self.grammar_correction_model)

        self.model = run_config.get("model")  # for "ollama" engine
        if not self._verify_engine_configuration():
            raise RuntimeError("Please check configuration of the exercise generation engine.")

    def __init__(self, config: str | None = None, topic: str | None = None, num: int = 0):
        """
        Supported commands:
        * run
        * list

        Parameters
        ----------
        config: str, default=None
            Path to the configuration file. If not provided then "hmeg.conf" is used.
        topic: str, default=None
            Name of the topic to generate exercises for. Can override topic from `config`
        num: int, default=0
            Number of exercises. Can override number of exercises defined in `config`.
        """
        if isinstance(topic, (list, tuple)):
            # fire splits values, containing commas. Join with comma+space to restore the original value
            topic = ", ".join(str(p) for p in topic)
        self._configure_from_file(config_path=config, topic=topic, num=num)

    def list(self):
        """
        Prints list of registered grammar topics.
        """
        topics = GrammarRegistry.get_registered_topics()
        print("\n".join(topics))

    def run(self):
        """
        Run generation of exercises using the current configuration and print them on the screen.
        """

        topics = GrammarRegistry.find_topics(self.topic)
        if len(topics) == 0:
            print(f"Requested an unregistered topic: {self.topic}. Please run `hmeg list` to see the existing topics.")
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
            cur_topic = random.choice(topics)
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


def main():
    if len(sys.argv) == 1:  # no arguments
        Runner().run()
    else:
        fire.Fire(Runner)


if __name__ == "__main__":
    main()
