import unittest

from hmeg.reranker import Reranker


class TestReranker(unittest.TestCase):
    def test_rank_kenlm(self):
        with self.subTest("Short sentence-1"):
            sorted_res = Reranker.rank(
                context="Quick brown foks jumped",
                original="foks",
                replacements=["box", "fox", "sox", "crocs"]
            )
            expected = ('box', -17.535741806030273)  # yeah...
            self.assertEqual(sorted_res[0][0], expected[0])

        with self.subTest("Short sentence-2"):
            sorted_res = Reranker.rank(
                context="Quick brown fox jumpd",
                original="jumpd",
                replacements=["dumped", "bumped", "jumped", "crocs"]
            )
            expected = ('jumped', -23.997745513916016)
            self.assertEqual(sorted_res[0][0], expected[0])

        with self.subTest("Long phrase"):
            sorted_res = Reranker.rank(
                context="All year long, the grasshopper kept burying acorns for winter while the oktopus mooched off his girlfriend and watched TV.",
                original="oktopus",
                replacements=["aktopus", "octopus", "octocat", "crocs"]
            )
            expected = ('octopus', -57.33723068237305)
            self.assertEqual(sorted_res[0][0], expected[0])
