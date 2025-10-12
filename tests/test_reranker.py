import os
import unittest

from hmeg.reranker import Reranker


class TestReranker(unittest.TestCase):
    def test_prepare_candidates(self):
        with self.subTest("Replacement at the beginning"):
            res = Reranker.prepare_candidates("test context", "test", ["test2", "test3"], full_context=True)
            self.assertListEqual(res, ["test context", "test2 context", "test3 context"])

            res = Reranker.prepare_candidates("test context", "test", ["test2", "test3"], full_context=False)
            self.assertListEqual(res, ["test", "test2", "test3"])

            res = Reranker.prepare_candidates("test context", "test", ["test2", "test3"])  # default
            self.assertListEqual(res, ["test", "test2", "test3"])

        with self.subTest("Replacement at the end"):
            res = Reranker.prepare_candidates("context test", "test", ["test2", "test3"], full_context=True)
            self.assertListEqual(res, ["context test", "context test2", "context test3"])

            res = Reranker.prepare_candidates("context test", "test", ["test2", "test3"], full_context=False)
            self.assertListEqual(res, ["context test", "context test2", "context test3"])

            res = Reranker.prepare_candidates("context test", "test", ["test2", "test3"])  # default
            self.assertListEqual(res, ["context test", "context test2", "context test3"])

        with self.subTest("Replacement in the middle"):
            res = Reranker.prepare_candidates("context test context", "test", ["test2", "test3"], full_context=True)
            self.assertListEqual(res, ["context test context", "context test2 context", "context test3 context"])

            res = Reranker.prepare_candidates("context test context", "test", ["test2", "test3"], full_context=False)
            self.assertListEqual(res, ["context test", "context test2", "context test3"])

            res = Reranker.prepare_candidates("context test context", "test", ["test2", "test3"])  # default
            self.assertListEqual(res, ["context test", "context test2", "context test3"])

        with self.subTest("Multiple occurrences of the original"):
            with self.assertWarns(UserWarning):
                res = Reranker.prepare_candidates("context test context test context", "test", ["test2", "test3"], full_context=True)
            self.assertListEqual(res, ["context test context test context", "context test2 context test context", "context test3 context test context"])

            with self.assertWarns(UserWarning):
                res = Reranker.prepare_candidates("context test context test context", "test", ["test2", "test3"], full_context=False)
            self.assertListEqual(res, ["context test", "context test2", "context test3"])

            with self.assertWarns(UserWarning):
                res = Reranker.prepare_candidates("context test context test context", "test", ["test2", "test3"])  # default
            self.assertListEqual(res, ["context test", "context test2", "context test3"])

        with self.subTest("No occurrences of original"):
            with self.assertRaises(ValueError):
                Reranker.prepare_candidates("context bad context", "test", ["test2", "test3"])

    @unittest.skipIf(not os.path.exists("lm/en.arpa.bin"), "Download the KenLM model and tokenizer first (eg from: https://huggingface.co/edugp/kenlm)")
    def test_rank_kenlm(self):
        Reranker.set_current_model(Reranker.Models.kenlm_en)

        with self.subTest("Short sentence-1"):
            sorted_res = Reranker.rank(
                context="Quick brown foks jumped",
                original="foks",
                replacements=["box", "fox", "sox", "crocs"]
            )
            expected = ('box', -17.535741806030273)  # yeah...
            self.assertEqual(sorted_res[0], expected)

        with self.subTest("Short sentence-2"):
            sorted_res = Reranker.rank(
                context="Quick brown fox jumpd",
                original="jumpd",
                replacements=["dumped", "bumped", "jumped", "crocs"]
            )
            expected = ('jumped', -23.997745513916016)
            self.assertEqual(sorted_res[0], expected)

        with self.subTest("Long phrase"):
            sorted_res = Reranker.rank(
                context="All year long, the grasshopper kept burying acorns for winter while the oktopus mooched off his girlfriend and watched TV.",
                original="oktopus",
                replacements=["aktopus", "octopus", "octocat", "crocs"]
            )
            expected = ('octopus', -57.33723068237305)
            self.assertEqual(sorted_res[0], expected)

    @unittest.skipIf(not os.path.exists("lm/en.arpa.bin"), "Download the KenLM model and tokenizer first (eg from: https://huggingface.co/edugp/kenlm)")
    def test_rank_kenlm_full_context(self):
        Reranker.set_current_model(Reranker.Models.kenlm_en)

        with self.subTest("Short sentence-1"):
            sorted_res = Reranker.rank(
                context="Quick brown foks jumped",
                original="foks",
                replacements=["box", "fox", "sox", "crocs"],
                full_sentence_score=True
            )
            expected = ('box', -23.4067325592041)  # yeah...
            self.assertEqual(sorted_res[0], expected)

        with self.subTest("Short sentence-2"):
            sorted_res = Reranker.rank(
                context="Quick brown fox jumpd",
                original="jumpd",
                replacements=["dumped", "bumped", "jumped", "crocs"],
                full_sentence_score=True
            )
            expected = ('jumped', -23.997745513916016)
            self.assertEqual(sorted_res[0], expected)

        with self.subTest("Long phrase"):
            sorted_res = Reranker.rank(
                context="All year long, the grasshopper kept burying acorns for winter while the oktopus mooched off his girlfriend and watched TV.",
                original="oktopus",
                replacements=["aktopus", "octopus", "octocat", "crocs"],
                full_sentence_score=True
            )
            expected = ('octopus', -85.282470703125)
            self.assertEqual(sorted_res[0], expected)

    def test_rank_distillgpt2(self):
        Reranker.set_current_model(Reranker.Models.distillgpt2)

        with self.subTest("Short sentence-1"):
            sorted_res = Reranker.rank(
                context="Quick brown foks jumped",
                original="foks",
                replacements=["box", "fox", "sox", "crocs"]
            )
            expected = ('crocs', -9.753257751464844)  # yeah...
            self.assertEqual(sorted_res[0][0], expected[0])
            self.assertAlmostEqual(sorted_res[0][1], expected[1], places=5)

        with self.subTest("Short sentence-2"):
            sorted_res = Reranker.rank(
                context="Quick brown fox jumpd",
                original="jumpd",
                replacements=["dumped", "bumped", "jumped", "crocs"]
            )
            expected = ('crocs', -9.221861839294434)  # yeah...
            self.assertEqual(sorted_res[0][0], expected[0])
            self.assertAlmostEqual(sorted_res[0][1], expected[1], places=5)

        with self.subTest("Long phrase"):
            sorted_res = Reranker.rank(
                context="All year long, the grasshopper kept burying acorns for winter while the oktopus mooched off his girlfriend and watched TV.",
                original="oktopus",
                replacements=["aktopus", "octopus", "octocat", "crocs"]
            )
            expected = ('octopus', -5.121907711029053)
            self.assertEqual(sorted_res[0][0], expected[0])
            self.assertAlmostEqual(sorted_res[0][1], expected[1], places=5)

    def test_rank_distillgpt2_full_context(self):
        Reranker.set_current_model(Reranker.Models.distillgpt2)

        with self.subTest("Short sentence-1"):
            sorted_res = Reranker.rank(
                context="Quick brown foks jumped",
                original="foks",
                replacements=["box", "fox", "sox", "crocs"],
                full_sentence_score=True
            )
            expected = ('crocs', -10.14453125)  # yeah...
            self.assertEqual(sorted_res[0][0], expected[0])
            self.assertAlmostEqual(sorted_res[0][1], expected[1], places=5)

        with self.subTest("Short sentence-2"):
            sorted_res = Reranker.rank(
                context="Quick brown fox jumpd",
                original="jumpd",
                replacements=["dumped", "bumped", "jumped", "crocs"],
                full_sentence_score=True
            )
            expected = ('crocs', -9.221861839294434)  # yeah...
            self.assertEqual(sorted_res[0][0], expected[0])
            self.assertAlmostEqual(sorted_res[0][1], expected[1], places=5)

        with self.subTest("Long phrase"):
            sorted_res = Reranker.rank(
                context="All year long, the grasshopper kept burying acorns for winter while the oktopus mooched off his girlfriend and watched TV.",
                original="oktopus",
                replacements=["aktopus", "octopus", "octocat", "crocs"],
                full_sentence_score=True
            )
            expected = ('octopus', -5.141149044036865)
            self.assertEqual(sorted_res[0][0], expected[0])
            self.assertAlmostEqual(sorted_res[0][1], expected[1], places=4)
