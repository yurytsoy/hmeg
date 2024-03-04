import unittest

from hmeg import GrammarRegistry, utils, ExerciseGenerator


class TestExerciseGenerator(unittest.TestCase):
    @classmethod
    def setUp(cls):
        super().setUp(cls)
        utils.register_grammar_topics("hmeg/grammar/")

    def test_generate(self):
        for topic in GrammarRegistry.topics:
            res = ExerciseGenerator.generate_exercises(topic, num=10)
            print(res)

    def test_generate_unregistered_topic(self):
        with self.assertRaises(RuntimeError):
            ExerciseGenerator.generate_exercises("bad topic", num=10)
