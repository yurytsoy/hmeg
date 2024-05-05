import unittest

from hmeg import GrammarRegistry, usecases as uc


class GrammarRegistryTest(unittest.TestCase):
    def test_get_registered_levels(self):
        GrammarRegistry.reset()

        with self.subTest("Empty registry"):
            levels_info = GrammarRegistry.get_registered_levels()
            self.assertDictEqual(levels_info, {})

        with self.subTest("Non empty registry"):
            uc.register_grammar_topics()
            levels_info = GrammarRegistry.get_registered_levels()
            expected = {
                'King Sejong Institute Practical Korean': ["3. Intermediate"],
                'TTMIK': ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 9"],
                'HTSK': ["Unit 1", "Unit 2", "Unit 3", "Unit 4", "Unit 5", "Unit 6"]
            }
            self.assertDictEqual(levels_info, expected)

        with self.subTest("With miniphrase"):
            uc.register_grammar_topics(grammar_dir="hmeg/miniphrase")
            levels_info = GrammarRegistry.get_registered_levels()
            expected["The Art and Science of Learning Languages"] = []
            self.assertDictEqual(levels_info, expected)

    def test_find_topic(self):
        GrammarRegistry.reset()

        with self.subTest("Empty registry"):
            topics = GrammarRegistry.find_topics("I want to… / -고 싶어요")
            self.assertEqual(topics, [])

        with self.subTest("Empty topic name"):
            uc.register_grammar_topics()
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
            expected = [
                'The more … the more … / -(으)면 -(으)ㄹ수록',
                'You shouldn’t…, You’re not supposed to… / -(으)면 안 돼요, 하면 안 돼요',
                'While / -(으)면서',
                'If, In case / 만약, -(으)면',
                'Well then, In that case, If so / 그러면, 그럼'
            ]
            self.assertListEqual(topics, expected)

        with self.subTest("Case sensitivity"):
            topics1 = GrammarRegistry.find_topics("Please")
            topics2 = GrammarRegistry.find_topics("please")
            self.assertListEqual(topics1, topics2)
