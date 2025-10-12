import os
import unittest

import random

from hmeg import usecases, GrammarChecker, GrammarRegistry, ExerciseGenerator, Vocabulary, load_minilex
from hmeg.grammar_checker import filter_replacements


@unittest.skipIf(not os.path.exists("lm/en.arpa.bin"), "Download the KenLM model and tokenizer first (eg from: https://huggingface.co/edugp/kenlm)")
class TestGrammarChecker(unittest.TestCase):
    def setUp(self):
        super().setUp()
        usecases.register_grammar_topics("hmeg/topics/")

        self.vocab = load_minilex()

    def test_correct_phrases_futurama(self):
        """
        Just for fun and giggles.
        """
        exercises = [
            "No, I'm ... doesn't",
            "Free Popplers",
            "Free the Popplers",
            "Bonjour",
            "Death by snu-snu!",
            "Let me aks you something",
            "It's a buncha, muncha, cruncha... human",
            "Oh, niggle-snoosh!"
        ]
        res_exercises = GrammarChecker.correct_phrases(exercises, vocab=self.vocab)
        corrections = sum([ex != res_ex for ex, res_ex in zip(exercises, res_exercises)])
        self.assertGreaterEqual(corrections, 1)

    def test_correct_phrases(self):
        random.seed(42)

        corrections = 0
        for topic in list(GrammarRegistry.topics)[:10]:
            exercises = ExerciseGenerator.generate_exercises(topic, num=10)
            res_exercises = GrammarChecker.correct_phrases(exercises, vocab=self.vocab)
            self.assertEqual(len(exercises), len(res_exercises))
            corrections += sum([ex != res_ex for ex, res_ex in zip(exercises, res_exercises)])
        self.assertGreaterEqual(corrections, 10)

    def test_filter_replacements(self):
        with self.subTest("Empty replacements"):
            # return empty result
            res = filter_replacements(original="", replacements=[], vocab=self.vocab)
            self.assertEqual(res, [])

        with self.subTest("Empty original"):
            # no filtering, regardless of replacements
            res = filter_replacements(original="", replacements=["foo", "bar"], vocab=self.vocab)
            self.assertEqual(res, ["foo", "bar"])

        with self.subTest("All caps"):
            res = filter_replacements(original="word", replacements=["APARTMENT", "clean"], vocab=self.vocab)
            self.assertEqual(res, ["clean"])

        with self.subTest("All caps-2"):
            res = filter_replacements(original="WORD", replacements=["APARTMENT", "clean"], vocab=self.vocab)
            self.assertEqual(res, ["APARTMENT"])

        with self.subTest("Word is not from vocabulary"):
            res = filter_replacements(original="foo", replacements=["bar", "apartment", "clean"], vocab=self.vocab)
            self.assertEqual(res, ["apartment", "clean"])

        with self.subTest("Replacements is not from vocabulary"):
            res = filter_replacements(original="word", replacements=["bar", "apartment", "clean"], vocab=self.vocab)
            self.assertEqual(res, ["apartment", "clean"])
