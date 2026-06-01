import random
import unittest

from hmeg import GrammarRegistry, usecases, ExerciseGenerator
from hmeg.entities import VocabularyPlaceholders
from hmeg.usecases import is_ollama_available

TEST_OLLAMA_MODEL = "gemma4:e2b"


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

    @unittest.skipIf(
        not is_ollama_available(TEST_OLLAMA_MODEL),
        "Run locally with Ollama service available"
    )
    def test_generate_exercises_ollama(self):
        from hmeg.exercise_generator.ollama_gen import generate_exercises

        random.seed(42)
        topics = random.choices(list(GrammarRegistry.topics), k=5)

        for topic in topics:
            exercises = generate_exercises(topic, num=5, model=TEST_OLLAMA_MODEL)
            self.assertEqual(len(exercises), 5)
            print(f"Exercises for topic '{topic}':")
            for ex in exercises:
                print(f"- {ex.sentence_kr} / {ex.sentence_en}")

            # check that no exercises contain placeholders.
            for placeholder in VocabularyPlaceholders.to_list():
                self.assertTrue(all(placeholder not in res.sentence_en for res in exercises if res.sentence_en is not None))
                self.assertTrue(all(placeholder not in res.sentence_kr for res in exercises if res.sentence_kr is not None))

    @unittest.skipIf(
        not is_ollama_available(TEST_OLLAMA_MODEL),
        "Run locally with Ollama service available"
    )
    def test_generate_exercises_ollama_with_vocabulary_level(self):
        from hmeg.exercise_generator.ollama_gen import generate_exercises

        random.seed(37)
        topic = random.choices(list(GrammarRegistry.topics), k=1)[0]
        print(f"Exercises for topic '{topic}':")

        exercises_a1 = generate_exercises(topic, num=5, model=TEST_OLLAMA_MODEL, vocab_level="A1")
        max_len_kr_a1 = max(len(ex.sentence_kr) for ex in exercises_a1)
        max_len_en_a1 = max(len(ex.sentence_en) for ex in exercises_a1)
        for ex in exercises_a1:
            print(f"- [A1] {ex.sentence_kr} / {ex.sentence_en}")

        exercises_c2 = generate_exercises(topic, num=5, model=TEST_OLLAMA_MODEL, vocab_level="C2")
        max_len_kr_c2 = max(len(ex.sentence_kr) for ex in exercises_c2)
        max_len_en_c2 = max(len(ex.sentence_en) for ex in exercises_c2)
        for ex in exercises_c2:
            print(f"- [C2] {ex.sentence_kr} / {ex.sentence_en}")

        # more advanced vocabulary level normally leads to longer and more complex sentences.
        # but allow for some variance in length due to model randomness.
        self.assertGreaterEqual(max_len_kr_c2 + 8, max_len_kr_a1)
        self.assertGreaterEqual(max_len_en_c2 + 8, max_len_en_a1)
