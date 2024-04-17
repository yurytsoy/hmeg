import unittest

from hmeg import GrammarRegistry, usecases, ExerciseGenerator
from hmeg.entities import VocabularyPlaceholders


class TestExerciseGenerator(unittest.TestCase):
    @classmethod
    def setUp(cls):
        super().setUp(cls)
        usecases.register_grammar_topics("hmeg/topics/")

    def test_generate_exercises(self):
        for topic in GrammarRegistry.topics:
            exercises = ExerciseGenerator.generate_exercises(topic, num=10)
            self.assertEqual(len(exercises), 10)

            # check that no exercises contain placeholders.
            for placeholder in VocabularyPlaceholders.to_list():
                self.assertTrue(all(placeholder not in res for res in exercises))

    def test_generate_exercises_unregistered_topic(self):
        with self.assertRaises(RuntimeError):
            ExerciseGenerator.generate_exercises("bad topic", num=10)
