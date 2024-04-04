import unittest

from hmeg import GrammarRegistry, utils


class GrammarRegistryTest(unittest.TestCase):
    def test_find_topic(self):
        GrammarRegistry.reset()

        with self.subTest("Empty registry"):
            topics = GrammarRegistry.find_topics("I want to… / -고 싶어요")
            self.assertEqual(topics, [])

        with self.subTest("Empty topic name"):
            utils.register_grammar_topics()
            topics = GrammarRegistry.find_topics("")
            self.assertEqual(topics, [])

            topics = GrammarRegistry.find_topics(None)
            self.assertEqual(topics, [])

        with self.subTest("Non-existing topic"):
            topics = GrammarRegistry.find_topics("Non-existing topic")
            self.assertEqual(topics, [])

        with self.subTest("Exact match"):
            topics = GrammarRegistry.find_topics("I want to… / -고 싶어요")
            self.assertEqual(topics, ["I want to… / -고 싶어요"])

        with self.subTest("Multiple matches"):
            topics = GrammarRegistry.find_topics("면")
            expected = ['While / -(으)면서', 'If, In case / 만약, -(으)면', 'Well then, In that case, If so / 그러면, 그럼']
            self.assertListEqual(topics, expected)

        with self.subTest("Case sensitivity"):
            topics1 = GrammarRegistry.find_topics("Please")
            topics2 = GrammarRegistry.find_topics("please")
            self.assertListEqual(topics1, topics2)
