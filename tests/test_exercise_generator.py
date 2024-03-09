import unittest

from hmeg import GrammarRegistry, utils, ExerciseGenerator
from hmeg.entities import MinilexPlaceholders


class TestExerciseGenerator(unittest.TestCase):
    @classmethod
    def setUp(cls):
        super().setUp(cls)
        utils.register_grammar_topics("hmeg/grammar/")

    def test_generate(self):
        for topic in GrammarRegistry.topics:
            ress = ExerciseGenerator.generate_exercises(topic, num=10)
            self.assertEqual(len(ress), 10)

            # check that no exercises contain placeholders.
            for placeholder in MinilexPlaceholders.to_list():
                self.assertTrue(all(placeholder not in res for res in ress))

    def test_generate_unregistered_topic(self):
        with self.assertRaises(RuntimeError):
            ExerciseGenerator.generate_exercises("bad topic", num=10)
