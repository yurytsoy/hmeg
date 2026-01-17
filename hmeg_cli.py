from __future__ import annotations

import random

import dotenv
import fire
import numpy as np
import sys
import toml

from hmeg import usecases as uc, ExerciseGenerator, GrammarChecker, GrammarRegistry, Reranker, Vocabulary

dotenv.load_dotenv()


class Runner:
    def __init__(self, config: str | None = None, topic: str | None = None, n: int = 0):
        """
        Supported commands:
        * run
        * list

        :param config:
            Path to the configuration file. If not provided then "hmeg.conf" is used.
        :param topic:
            Name of the topic to generate exercises for. Can override topic from `config`
        :param n:
            Number of exercises. Can override number of exercises defined in `config`.
        """
        self.config_file = config or "hmeg.conf"

        try:
            with open(self.config_file, mode="r") as f:
                run_config = toml.loads(f.read())
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file `{self.config_file}` not found.") from e

        topics_folder = run_config.get("topics_folder")
        if not topics_folder:
            raise KeyError("`topics_folder` missing in config.")
        uc.register_grammar_topics(topics_folder)

        vocab_file = run_config.get("vocab_file")
        if not vocab_file:
            raise KeyError("`vocab_file` missing in config.")
        self.vocab = Vocabulary.load(vocab_file)

        self.topic = topic or run_config.get("topic")
        if not self.topic:
            raise KeyError("`topic` missing in config and not provided as argument.")

        configured_num = n or run_config.get("number_exercises", 10)
        try:
            configured_num = int(configured_num)
        except (ValueError, TypeError):
            configured_num = 10
        self.num_exercises = max(5, min(configured_num, 100))

        self.grammar_correction_model = run_config.get("grammar_correction")
        if self.grammar_correction_model is not None:
            Reranker.set_current_model(self.grammar_correction_model)

    def list(self):
        """
        Prints list of registered topics.
        """
        topics = GrammarRegistry.get_registered_topics()
        print("\n".join(topics))

    def run(self):
        """
        Runs generation of exercises and prints them on the screen.
        """

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
                topic_name=cur_topic, num=cur_topic_num_exercises, vocab=self.vocab
            )
            for cur_exercise in cur_topic_exercises:
                if cur_exercise not in exercises:
                    exercises.append(cur_exercise)
            attempts += cur_topic_num_exercises
            if attempts > self.num_exercises ** 2:
                break

        if self.grammar_correction_model is not None:
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
