import unittest

from hmeg.entities.grammar import GrammarDescription, TopicLevelInfo


class TestTopicLevelInfo(unittest.TestCase):
    def test_creation_with_all_fields(self):
        """Test creating TopicLevelInfo with all fields."""
        info = TopicLevelInfo(resource_name="test_resource", level="A1")
        
        self.assertEqual(info.resource_name, "test_resource")
        self.assertEqual(info.level, "A1")

    def test_creation_with_minimal_fields(self):
        """Test creating TopicLevelInfo with only required fields."""
        info = TopicLevelInfo(resource_name="test_resource")
        
        self.assertEqual(info.resource_name, "test_resource")
        self.assertIsNone(info.level)

    def test_level_as_int(self):
        """Test that level can be an integer."""
        info = TopicLevelInfo(resource_name="test_resource", level=5)
        
        self.assertEqual(info.level, 5)
        self.assertIsInstance(info.level, int)

    def test_level_as_string(self):
        """Test that level can be a string."""
        info = TopicLevelInfo(resource_name="test_resource", level="Advanced")
        
        self.assertEqual(info.level, "Advanced")
        self.assertIsInstance(info.level, str)


class TestGrammarDescription(unittest.TestCase):
    def test_creation_with_all_fields(self):
        """Test creating GrammarDescription with all fields."""
        desc = GrammarDescription(
            name="Present Simple",
            links=["https://example.com/link1", "https://example.com/link2"],
            exercises=["exercise1.txt", "exercise2.txt"],
            levels=[
                TopicLevelInfo(resource_name="book1", level="A1"),
                TopicLevelInfo(resource_name="book2", level="A2")
            ]
        )
        
        self.assertEqual(desc.name, "Present Simple")
        self.assertEqual(len(desc.links), 2)
        self.assertEqual(len(desc.exercises), 2)
        self.assertEqual(len(desc.levels), 2)

    def test_from_dict_minimal(self):
        """Test creating GrammarDescription from dict with minimal fields."""
        data = {
            "name": "Past Tense",
            "links": ["https://example.com"],
            "exercises": ["exercise1.txt"]
        }
        desc = GrammarDescription.from_dict(data)
        
        self.assertEqual(desc.name, "Past Tense")
        self.assertEqual(desc.links, ["https://example.com"])
        self.assertEqual(desc.exercises, ["exercise1.txt"])
        self.assertEqual(desc.levels, [])

    def test_from_dict_with_levels(self):
        """Test creating GrammarDescription from dict with levels."""
        data = {
            "name": "Present Simple",
            "links": ["https://example.com"],
            "exercises": ["exercise1.txt"],
            "levels": [
                {"resource_name": "book1", "level": "A1"},
                {"resource_name": "book2", "level": "A2"},
                {"resource_name": "book3"}  # no level specified
            ]
        }
        desc = GrammarDescription.from_dict(data)
        
        self.assertEqual(desc.name, "Present Simple")
        self.assertEqual(len(desc.levels), 3)
        self.assertIsInstance(desc.levels[0], TopicLevelInfo)
        self.assertEqual(desc.levels[0].resource_name, "book1")
        self.assertEqual(desc.levels[0].level, "A1")
        self.assertEqual(desc.levels[1].resource_name, "book2")
        self.assertEqual(desc.levels[1].level, "A2")
        self.assertEqual(desc.levels[2].resource_name, "book3")
        self.assertIsNone(desc.levels[2].level)

    def test_from_dict_with_integer_levels(self):
        """Test creating GrammarDescription from dict with integer levels."""
        data = {
            "name": "Grammar Topic",
            "links": [],
            "exercises": [],
            "levels": [
                {"resource_name": "book1", "level": 1},
                {"resource_name": "book2", "level": 2}
            ]
        }
        desc = GrammarDescription.from_dict(data)
        
        self.assertEqual(len(desc.levels), 2)
        self.assertEqual(desc.levels[0].level, 1)
        self.assertEqual(desc.levels[1].level, 2)

    def test_empty_lists(self):
        """Test GrammarDescription with empty lists."""
        desc = GrammarDescription(
            name="Test",
            links=[],
            exercises=[],
            levels=[]
        )
        
        self.assertEqual(desc.name, "Test")
        self.assertEqual(desc.links, [])
        self.assertEqual(desc.exercises, [])
        self.assertEqual(desc.levels, [])

    def test_from_dict_preserves_link_order(self):
        """Test that link order is preserved when creating from dict."""
        data = {
            "name": "Test",
            "links": ["link3", "link1", "link2"],
            "exercises": []
        }
        desc = GrammarDescription.from_dict(data)
        
        self.assertEqual(desc.links, ["link3", "link1", "link2"])

    def test_from_dict_preserves_exercise_order(self):
        """Test that exercise order is preserved when creating from dict."""
        data = {
            "name": "Test",
            "links": [],
            "exercises": ["ex3.txt", "ex1.txt", "ex2.txt"]
        }
        desc = GrammarDescription.from_dict(data)
        
        self.assertEqual(desc.exercises, ["ex3.txt", "ex1.txt", "ex2.txt"])

    def test_levels_with_mixed_types(self):
        """Test levels with mixed string and integer types."""
        data = {
            "name": "Mixed Levels",
            "links": [],
            "exercises": [],
            "levels": [
                {"resource_name": "book1", "level": "Beginner"},
                {"resource_name": "book2", "level": 2},
                {"resource_name": "book3", "level": "C1"}
            ]
        }
        desc = GrammarDescription.from_dict(data)
        
        self.assertEqual(len(desc.levels), 3)
        self.assertEqual(desc.levels[0].level, "Beginner")
        self.assertEqual(desc.levels[1].level, 2)
        self.assertEqual(desc.levels[2].level, "C1")


if __name__ == '__main__':
    unittest.main()
