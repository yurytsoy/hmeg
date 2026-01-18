import random
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
            exercises = ExerciseGenerator.generate_exercises(topic, num=5)
            self.assertEqual(len(exercises), 5)

            # check that no exercises contain placeholders.
            for placeholder in VocabularyPlaceholders.to_list():
                self.assertTrue(all(placeholder not in res for res in exercises))

    def test_generate_exercises_unregistered_topic(self):
        with self.assertRaises(RuntimeError):
            ExerciseGenerator.generate_exercises("bad topic", num=10)

    @unittest.skip("Run locally with Ollama service available")
    def test_generate_exercises_ollama(self):
        random.seed(42)
        topics = random.choices(list(GrammarRegistry.topics), k=5)

        for topic in topics:
            exercises = ExerciseGenerator.generate_exercises_ollama(topic, num=5, model="ministral-3:14b")
            self.assertEqual(len(exercises), 5)
            print(f"Exercises for topic '{topic}':")
            for ex in exercises:
                print(f"- {ex.source_text} / {ex.translation}")

            # check that no exercises contain placeholders.
            for placeholder in VocabularyPlaceholders.to_list():
                self.assertTrue(all(placeholder not in res.source_text for res in exercises))

    def test_generate_exercises_ollama_with_vocab(self):
        # TODO: implement me
        ...
