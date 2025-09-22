import os.path
import unittest

import random

from hmeg import usecases, GrammarChecker, GrammarRegistry, ExerciseGenerator, Vocabulary, load_minilex
from hmeg.grammar_checker import filter_replacements, rank_candidates_decoder, rank_candidates_kenlm


class TestGrammarChecker(unittest.TestCase):
    @classmethod
    def setUp(cls):
        super().setUp(cls)
        usecases.register_grammar_topics("hmeg/topics/")

        cls.vocab = load_minilex()

    def test_correct_exercises_futurama(self):
        """
        Just for funs and giggles.
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

    def test_correct_exercises(self):
        random.seed(42)

        corrections = 0
        for topic in list(GrammarRegistry.topics)[:10]:
            exercises = ExerciseGenerator.generate_exercises(topic, num=10)
            res_exercises = GrammarChecker.correct_phrases(exercises, vocab=self.vocab)
            self.assertEqual(len(exercises), len(res_exercises))
            corrections += sum([ex != res_ex for ex, res_ex in zip(exercises, res_exercises)])
        self.assertGreaterEqual(corrections, 1)  # 1 out of 100, not much...

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

    @unittest.skipIf(not os.path.exists("lm/en.arpa.bin"), "Download KenLM model first")
    def test_rank_candidates_kenlm(self):
        with self.subTest("Short sentence-1"):
            res = rank_candidates_kenlm(
                context="Quick brown foks jumped",
                original="foks",
                replacements=["box", "fox", "sox", "crocs"]
            )
            sorted_res = sorted(res, key=lambda k: k[1], reverse=True)
            expected = ('box', -17.535741806030273)  # yeah...
            self.assertEqual(sorted_res[0][0], expected[0])

        with self.subTest("Short sentence-2"):
            res = rank_candidates_kenlm(
                context="Quick brown fox jumpd",
                original="jumpd",
                replacements=["dumped", "bumped", "jumped", "crocs"]
            )
            sorted_res = sorted(res, key=lambda k: k[1], reverse=True)
            expected = ('jumped', -23.997745513916016)
            self.assertEqual(sorted_res[0][0], expected[0])

        with self.subTest("Long phrase"):
            res = rank_candidates_kenlm(
                context="All year long, the grasshopper kept burying acorns for winter while the oktopus mooched off his girlfriend and watched TV.",
                original="oktopus",
                replacements=["aktopus", "octopus", "octocat", "crocs"]
            )
            sorted_res = sorted(res, key=lambda k: k[1], reverse=True)
            expected = ('octopus', -57.33723068237305)
            self.assertEqual(sorted_res[0][0], expected[0])

    def test_rank_candidates_decoder(self):
        with self.subTest("Short sentence-1"):
            res = rank_candidates_decoder(
                context="Quick brown foks jumped",
                original="foks",
                replacements=["box", "fox", "sox", "crocs"]
            )
            sorted_res = sorted(res, key=lambda k: k[1], reverse=True)
            expected = ('crocs', -14.629886627197266)  # yeah...
            self.assertEqual(sorted_res[0][0], expected[0])

        with self.subTest("Short sentence-2"):
            res = rank_candidates_decoder(
                context="Quick brown fox jumpd",
                original="jumpd",
                replacements=["dumped", "bumped", "jumped", "crocs"]
            )
            sorted_res = sorted(res, key=lambda k: k[1], reverse=True)
            expected = ('crocs', -18.443723678588867)  # yeah...
            self.assertEqual(sorted_res[0][0], expected[0])

        with self.subTest("Long phrase"):
            res = rank_candidates_decoder(
                context="All year long, the grasshopper kept burying acorns for winter while the oktopus mooched off his girlfriend and watched TV.",
                original="oktopus",
                replacements=["aktopus", "octopus", "octocat", "crocs"]
            )
            sorted_res = sorted(res, key=lambda k: k[1], reverse=True)
            expected = ('octocat', -32.926570892333984)  # oh, boy
            self.assertEqual(sorted_res[0][0], expected[0])
