from __future__ import annotations

import random

import fire
import numpy as np
import sys
import toml

from hmeg import utils, ExerciseGenerator, GrammarRegistry, Vocabulary


class Runner:
    def __init__(self, config: str | None = None, topic: str | None = None, n: int = 10):
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

        with open(self.config_file, mode="r") as f:
            run_config = toml.loads(f.read())
        utils.register_grammar_topics(run_config["topics_folder"])
        self.vocab = Vocabulary.load(run_config["vocab_file"])
        self.topic = topic or run_config["topic"]
        self.num_exercises = n or run_config["num_exercises"]

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
        while len(exercises) < self.num_exercises:
            cur_topic = np.random.choice(topics)
            cur_topic_num_exercises = min(self.num_exercises // len(topics), self.num_exercises - len(exercises))
            cur_topic_exercises = ExerciseGenerator.generate_exercises(
                topic_name=cur_topic, num=cur_topic_num_exercises, vocab=self.vocab
            )
            for cur_exercise in cur_topic_exercises:
                if cur_exercise not in exercises:
                    exercises.append(cur_exercise)
            attempts += cur_topic_num_exercises
            if attempts > self.num_exercises ** 2:
                break
        # shuffle exercises
        random.shuffle(exercises)
        for idx, exercise in enumerate(exercises):
            print(f"{idx + 1}. {exercise}")


if __name__ == "__main__":
    if len(sys.argv) == 1:  # no arguments
        Runner().run()
    else:
        fire.Fire(Runner)
