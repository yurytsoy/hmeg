import random

from nltk.parse.generate import generate
from nltk import CFG

from .registry import GrammarRegistry
from .utils import apply_minilex


class ExerciseGenerator:
    @staticmethod
    def generate_exercises(topic_name: str, num: int) -> list[str]:
        """
        Generates list of random translation exercises for the given topic.
        """
        if topic_name not in GrammarRegistry.topics:
            raise RuntimeError(f"Requested an unregistered topic: {topic_name}. Please check the topics' folder.")

        exercise_types = [
            CFG.fromstring(exercise_type)
            for exercise_type in GrammarRegistry.topics[topic_name].exercises
        ]
        res = []
        for _ in range(num):
            cur_type_idx = random.randint(0, len(exercise_types)-1)
            cur_exercise = next(generate(exercise_types[cur_type_idx], n=1))
            res.append(apply_minilex(" ".join(cur_exercise)))
        return res
