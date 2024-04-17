import random

from nltk.parse.generate import generate
from nltk import CFG
import os

from .registry import GrammarRegistry
from .usecases import apply_vocabulary
from .vocabulary import Vocabulary


cur_dir = os.path.split(os.path.realpath(__file__))[0]
DEFAULT_VOCABULARY_FILE = os.path.join(cur_dir, "vocabs/minilex.toml")


class ExerciseGenerator:
    @staticmethod
    def generate_exercises(topic_name: str, num: int, vocab: Vocabulary | None = None) -> list[str]:
        """
        Generates list of random translation exercises for the given topic.

        The results are represented as templates with placeholders for nouns, verbs, ....

        See also: `apply_vocabulary`
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
