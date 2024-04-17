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
