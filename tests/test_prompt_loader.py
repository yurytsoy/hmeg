import os
import tempfile
import unittest
from pathlib import Path

from hmeg.prompt_loader import PromptLoader
from hmeg.entities import Prompt


class TestPromptLoader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_prompt(self, prompt_id: str, content: str):
        """Helper to create a test prompt file."""
        parts = [p for p in Path(prompt_id).parts if p not in ("", ".")]
        path = self.base_dir.joinpath(*parts).with_suffix(".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_init_with_default_base_dir(self):
        """Test PromptLoader initialization with default base directory."""
        loader = PromptLoader()
        expected_base = Path(__file__).parent.parent / "hmeg" / "prompts"
        self.assertEqual(loader.base_dir, expected_base.resolve())

    def test_init_with_custom_base_dir(self):
        """Test PromptLoader initialization with custom base directory."""
        loader = PromptLoader(self.temp_dir)
        self.assertEqual(loader.base_dir, self.base_dir)

    def test_load_prompt_success(self):
        """Test loading a valid prompt file."""
        yaml_content = """
id: test/prompt
system_instructions: Test instructions
user_prompt_template: "Test template {arg1}"
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7
"""
        self._create_test_prompt("test/prompt", yaml_content)
        
        loader = PromptLoader(self.temp_dir)
        prompt = loader.load("test/prompt")
        
        self.assertIsInstance(prompt, Prompt)
        self.assertEqual(prompt.id, "test/prompt")
        self.assertEqual(prompt.system_instructions, "Test instructions")
        self.assertEqual(prompt.user_prompt_template, "Test template {arg1}")
        self.assertEqual(prompt.llm.provider, "openai")
        self.assertEqual(prompt.llm.model, "gpt-4")
        self.assertEqual(prompt.llm.temperature, 0.7)

    def test_load_prompt_with_caching(self):
        """Test that loading the same prompt twice uses cache."""
        yaml_content = """
id: cached/prompt
system_instructions: Cached instructions
user_prompt_template: "Cached template"
llm:
  provider: openai
  model: gpt-4
"""
        self._create_test_prompt("cached/prompt", yaml_content)
        
        loader = PromptLoader(self.temp_dir)
        prompt1 = loader.load("cached/prompt")
        prompt2 = loader.load("cached/prompt")
        
        # Should return the same cached object
        self.assertIs(prompt1, prompt2)
        self.assertEqual(len(loader._cache), 1)

    def test_load_prompt_not_found(self):
        """Test loading a non-existent prompt file raises FileNotFoundError."""
        loader = PromptLoader(self.temp_dir)
        
        with self.assertRaises(FileNotFoundError) as ctx:
            loader.load("nonexistent/prompt")
        
        self.assertIn("Prompt file not found", str(ctx.exception))
        self.assertIn("nonexistent/prompt", str(ctx.exception))

    def test_clear_cache_specific_prompt(self):
        """Test clearing cache for a specific prompt."""
        yaml_content = """
id: test/prompt
system_instructions: Test
user_prompt_template: "Template"
llm:
  provider: openai
  model: gpt-4
"""
        self._create_test_prompt("test/prompt", yaml_content)
        self._create_test_prompt("test/prompt2", yaml_content.replace("test/prompt", "test/prompt2"))
        
        loader = PromptLoader(self.temp_dir)
        loader.load("test/prompt")
        loader.load("test/prompt2")
        
        self.assertEqual(len(loader._cache), 2)
        
        loader.clear_cache("test/prompt")
        self.assertEqual(len(loader._cache), 1)
        self.assertNotIn("test/prompt", loader._cache)
        self.assertIn("test/prompt2", loader._cache)

    def test_clear_cache_all(self):
        """Test clearing entire cache."""
        yaml_content = """
id: test/prompt
system_instructions: Test
user_prompt_template: "Template"
llm:
  provider: openai
  model: gpt-4
"""
        self._create_test_prompt("test/prompt1", yaml_content)
        self._create_test_prompt("test/prompt2", yaml_content)
        
        loader = PromptLoader(self.temp_dir)
        loader.load("test/prompt1")
        loader.load("test/prompt2")
        
        self.assertEqual(len(loader._cache), 2)
        
        loader.clear_cache()
        self.assertEqual(len(loader._cache), 0)

    def test_path_traversal_prevention(self):
        """Test that path traversal attacks are prevented."""
        loader = PromptLoader(self.temp_dir)
        
        with self.assertRaises(ValueError) as ctx:
            loader.load("../../../etc/passwd")
        
        self.assertIn("outside base prompts directory", str(ctx.exception))

    def test_path_traversal_prevention_with_dots(self):
        """Test path traversal prevention with various dot patterns."""
        loader = PromptLoader(self.temp_dir)
        
        dangerous_paths = [
            "../../etc/passwd",
            "./../secrets",
            "valid/../../../etc/passwd",
        ]
        
        for dangerous_path in dangerous_paths:
            with self.subTest(path=dangerous_path):
                with self.assertRaises(ValueError):
                    loader._prompt_path(dangerous_path)

    def test_prompt_path_method(self):
        """Test _prompt_path method generates correct paths."""
        loader = PromptLoader(self.temp_dir)
        
        test_cases = [
            ("simple", self.base_dir / "simple.yaml"),
            ("v1/reranker/openai", self.base_dir / "v1" / "reranker" / "openai.yaml"),
            ("test/nested/deep/prompt", self.base_dir / "test" / "nested" / "deep" / "prompt.yaml"),
        ]
        
        for prompt_id, expected_path in test_cases:
            with self.subTest(prompt_id=prompt_id):
                result = loader._prompt_path(prompt_id)
                self.assertEqual(result, expected_path)

    def test_load_existing_openai_reranker_prompt(self):
        """Test loading the actual OpenAI reranker prompt from the repository."""
        # Use the real prompts directory
        loader = PromptLoader()
        
        try:
            prompt = loader.load("v1/reranker/openai")
            
            self.assertIsInstance(prompt, Prompt)
            self.assertEqual(prompt.id, "v1/reranker/openai")
            self.assertIsNotNone(prompt.system_instructions)
            self.assertIsNotNone(prompt.user_prompt_template)
            self.assertEqual(prompt.llm.provider, "openai")
            self.assertIsNotNone(prompt.llm.model)
        except FileNotFoundError:
            self.skipTest("OpenAI reranker prompt file not found in repository")


if __name__ == '__main__':
    unittest.main()
