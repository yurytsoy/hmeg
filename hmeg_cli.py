from __future__ import annotations

import fire
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
        for topic, descr in GrammarRegistry.topics.items():
            print(topic)

    def run(self):
        """
        Runs generation of exercises and prints them on the screen.
        """
        exercises = ExerciseGenerator.generate_exercises(
            topic_name=self.topic, num=self.num_exercises, vocab=self.vocab
        )
        print(f"Exercises for topic: {self.topic}")
        for idx, exercise in enumerate(exercises):
            print(f"{idx + 1}. {exercise}")


if __name__ == "__main__":
    if len(sys.argv) == 1:  # no arguments
        Runner().run()
    else:
        fire.Fire(Runner)
