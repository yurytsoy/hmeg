from dataclasses import asdict
import pathlib
import random
import unittest

from hmeg import usecases as uc
from hmeg.entities import VocabularyPlaceholders
from hmeg.vocabulary import Vocabulary


class TestUtils(unittest.TestCase):
    @classmethod
    def setUp(cls):
        super().setUp(cls)
        cls.vocab = Vocabulary.load("hmeg/vocabs/minilex.toml")

    def test_apply_vocabulary(self):
        random.seed(42)

        with self.subTest("Empty input"):
            self.assertEqual(uc.apply_vocabulary("", self.vocab), "")

        with self.subTest("No placeholders"):
            s = "abcde123^%&#$}{"
            self.assertEqual(uc.apply_vocabulary(s, self.vocab), s)

        with self.subTest("Multiple placeholders"):
            s = "[{verb}][{verb}][{verb}]"
            res = uc.apply_vocabulary(s, self.vocab)
            with self.assertRaises(ValueError):
                res.index(res[:3], 1)  # no repetitions in the generated string

        with self.subTest("All placeholders"):
            all_placeholders = " ".join(VocabularyPlaceholders.to_list())
            res = uc.apply_vocabulary(all_placeholders, self.vocab)
            for placeholder in VocabularyPlaceholders.to_list():
                self.assertTrue(placeholder not in res)

    def test_get_vocabulary_names(self):
        vocabs = uc.get_vocabulary_names()
        self.assertListEqual(vocabs, ["Minilex", "Nanolex"])

    def test_get_vocabularies_info(self):
        vocabs = uc.get_vocabularies_info()
        expected = [
            {'name': 'Minilex', 'num_adjectives': 90, 'num_adverbs': 69, 'num_nouns': 134, 'num_verbs': 75, 'num_words': 368},
            {'name': 'Nanolex', 'num_adjectives': 28, 'num_adverbs': 3, 'num_nouns': 50, 'num_verbs': 28, 'num_words': 109}
        ]
        self.assertListEqual([asdict(vocab) for vocab in vocabs], expected)

    def test_to_abs_path_returns_same_for_absolute_path(self):
        with self.subTest("Absolute path on input"):
            # tmp_path gives an absolute path on the filesystem
            abs_path = __file__
            self.assertEqual(uc.to_abs_path(abs_path), abs_path)

        with self.subTest("Relative path on input"):
            # tmp_path gives an absolute path on the filesystem
            rel_path = "hmeg/topics"  # default folder for topics
            res = uc.to_abs_path(rel_path)
            files = list(pathlib.Path(res).iterdir())
            self.assertGreaterEqual(len(files), 60)

    def test_to_abs_path_resolves_relative_to_project_parent(self):
        # reproduce the same base_dir calculation used in the function
        base_dir = pathlib.Path(uc.__file__).parent.parent
        rel = "some/relative/path.txt"
        expected = str(base_dir / rel)
        assert uc.to_abs_path(rel) == expected

    def test_to_abs_path_handles_parent_segments(self):
        base_dir = pathlib.Path(uc.__file__).parent.parent
        rel = "../outside_dir/config.toml"
        expected = str(base_dir / rel)
        assert uc.to_abs_path(rel) == expected