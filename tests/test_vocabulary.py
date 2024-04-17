import unittest

from hmeg import Vocabulary


class TestVocabulary(unittest.TestCase):
    def test_constructor_empty_path(self):
        vocab = Vocabulary()
        self.assertIsNone(vocab.vocab_file)
        self.assertIsNone(vocab.name)
        self.assertEqual(vocab.nouns, [])
        self.assertEqual(vocab.verbs, [])
        self.assertEqual(vocab.adjectives, [])
        self.assertEqual(vocab.adverbs, [])

    def test_constructor(self):
        vocab = Vocabulary("tests/vocabs/test_vocab.toml")
        self.assertEqual(vocab.name, "Test vocabulary")
        self.assertCountEqual(vocab.nouns, ["accident", "address"])
        self.assertCountEqual(vocab.verbs, ["arrive", "ask"])
        self.assertCountEqual(vocab.adjectives, ["angry", "another"])
        self.assertCountEqual(vocab.adverbs, ["angrily"])

    def test_import(self):
        vocab = Vocabulary("tests/vocabs/test_vocab_import.toml")
        self.assertEqual(vocab.name, "Vocabulary with import")
        self.assertCountEqual(vocab.nouns, ["accident", "address", "bag", "beginning"])
        self.assertCountEqual(vocab.verbs, ["arrive", "ask", "become", "begin"])
        self.assertCountEqual(vocab.adjectives, ["angry", "another", "bad", "beautiful"])
        self.assertCountEqual(vocab.adverbs, ["angrily", "badly", "beautifully"])
