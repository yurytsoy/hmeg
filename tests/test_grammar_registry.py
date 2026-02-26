import os
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
                'King Sejong Institute Practical Korean': ["1. Beginner", "2. Beginner", "3. Intermediate", "4. Intermediate"],
                'TTMIK': ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 9"],
                'HTSK': ["Unit 1", "Unit 2", "Unit 3", "Unit 4", "Unit 5", "Unit 6"],
                'Korean Grammar in Use': ['Beginning', 'Intermediate'],
                'Oh, my Korean!': ['1A', '1B', '2A', '2B', '3A', '3B', "4"],
            }
            self.assertDictEqual(levels_info, expected)

        with self.subTest("With miniphrase"):
            uc.register_grammar_topics(grammar_dir="hmeg/miniphrase", force=True)
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
                'Either A or B, Or / -거나, -(이)나, 아니면',
                'While / -(으)면서',
                'If, In case / 만약, -(으)면',
                'Well then, In that case, If so / 그러면, 그럼'
            ]
            self.assertCountEqual(topics, expected)

        with self.subTest("Case sensitivity"):
            topics1 = GrammarRegistry.find_topics("Please")
            topics2 = GrammarRegistry.find_topics("please")
            self.assertListEqual(topics1, topics2)


class RegisterGrammarTopicsTest(unittest.TestCase):
    """Tests for the per-directory deduplication logic in register_grammar_topics()."""

    def setUp(self):
        GrammarRegistry.reset()

    def tearDown(self):
        GrammarRegistry.reset()

    def test_loaded_dirs_populated_after_registration(self):
        """Calling register_grammar_topics() adds the resolved directory to loaded_dirs."""
        self.assertEqual(GrammarRegistry.loaded_dirs, set())
        uc.register_grammar_topics()
        self.assertEqual(len(GrammarRegistry.loaded_dirs), 1)

    def test_same_dir_not_reloaded(self):
        """Calling register_grammar_topics() twice for the same dir is a no-op the second time."""
        uc.register_grammar_topics()
        topics_after_first = set(GrammarRegistry.get_registered_topics())
        uc.register_grammar_topics()
        self.assertEqual(set(GrammarRegistry.get_registered_topics()), topics_after_first)

    def test_different_dirs_registered_independently(self):
        """Loading the default dir first does not prevent loading a second directory."""
        uc.register_grammar_topics()
        topics_after_default = set(GrammarRegistry.get_registered_topics())

        miniphrase_dir = os.path.join(os.path.dirname(uc.__file__), "miniphrase")
        uc.register_grammar_topics(grammar_dir=miniphrase_dir)

        topics_after_both = set(GrammarRegistry.get_registered_topics())
        self.assertGreater(len(topics_after_both), len(topics_after_default))
        self.assertEqual(len(GrammarRegistry.loaded_dirs), 2)

    def test_register_miniphrase_after_register_grammar_topics(self):
        """register_miniphrase() must work even if register_grammar_topics() was called first."""
        uc.register_grammar_topics()
        topics_after_default = set(GrammarRegistry.get_registered_topics())

        uc.register_miniphrase()
        topics_after_both = set(GrammarRegistry.get_registered_topics())
        self.assertGreater(len(topics_after_both), len(topics_after_default))

    def test_force_reloads_same_dir(self):
        """force=True causes a directory to be re-read even if already in loaded_dirs."""
        uc.register_grammar_topics()
        self.assertEqual(len(GrammarRegistry.loaded_dirs), 1)
        # force=True should not raise and should keep the dir in loaded_dirs
        uc.register_grammar_topics(force=True)
        self.assertEqual(len(GrammarRegistry.loaded_dirs), 1)

    def test_reset_clears_loaded_dirs(self):
        """GrammarRegistry.reset() must clear loaded_dirs alongside topics."""
        uc.register_grammar_topics()
        self.assertGreater(len(GrammarRegistry.loaded_dirs), 0)
        GrammarRegistry.reset()
        self.assertEqual(GrammarRegistry.loaded_dirs, set())
        self.assertEqual(GrammarRegistry.topics, {})
