import unittest
import yaml

from hmeg.entities.prompt import Prompt, LLMConfig


class TestLLMConfig(unittest.TestCase):
    def test_from_dict_minimal(self):
        """Test creating LLMConfig from dict with minimal fields."""
        data = {
            "provider": "openai",
            "model": "gpt-4"
        }
        config = LLMConfig.from_dict(data)
        
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "gpt-4")
        self.assertIsNone(config.temperature)
        self.assertIsNone(config.max_tokens)
        self.assertEqual(config.verbatim, {})

    def test_from_dict_with_all_fields(self):
        """Test creating LLMConfig from dict with all standard fields."""
        data = {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        config = LLMConfig.from_dict(data)
        
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 1000)

    def test_from_dict_with_extra_fields(self):
        """Test that extra fields are captured in verbatim."""
        data = {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "custom_field": "value"
        }
        config = LLMConfig.from_dict(data)
        
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(len(config.verbatim), 3)
        self.assertEqual(config.verbatim["top_p"], 0.9)
        self.assertEqual(config.verbatim["frequency_penalty"], 0.5)
        self.assertEqual(config.verbatim["custom_field"], "value")

    def test_to_dict_minimal(self):
        """Test converting LLMConfig to dict with minimal fields."""
        config = LLMConfig(provider="openai", model="gpt-4")
        result = config.to_dict()
        
        self.assertEqual(result["provider"], "openai")
        self.assertEqual(result["model"], "gpt-4")
        self.assertNotIn("temperature", result)
        self.assertNotIn("max_tokens", result)

    def test_to_dict_with_all_fields(self):
        """Test converting LLMConfig to dict with all fields."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            verbatim={"top_p": 0.9, "custom": "value"}
        )
        result = config.to_dict()
        
        self.assertEqual(result["provider"], "openai")
        self.assertEqual(result["model"], "gpt-4")
        self.assertEqual(result["temperature"], 0.7)
        self.assertEqual(result["max_tokens"], 1000)
        self.assertEqual(result["top_p"], 0.9)
        self.assertEqual(result["custom"], "value")


class TestPrompt(unittest.TestCase):
    def test_from_dict_minimal(self):
        """Test creating Prompt from dict with minimal fields."""
        data = {
            "id": "test/prompt",
            "llm": {
                "provider": "openai",
                "model": "gpt-4"
            }
        }
        prompt = Prompt.from_dict(data)
        
        self.assertEqual(prompt.id, "test/prompt")
        self.assertEqual(prompt.system_instructions, "")
        self.assertEqual(prompt.user_prompt_template, "")
        self.assertIsNone(prompt.output_schema)
        self.assertEqual(prompt.metadata, {})
        self.assertIsInstance(prompt.llm, LLMConfig)
        self.assertEqual(prompt.llm.model, "gpt-4")

    def test_from_dict_with_all_fields(self):
        """Test creating Prompt from dict with all fields."""
        data = {
            "id": "test/prompt",
            "system_instructions": "You are a helpful assistant",
            "user_prompt_template": "Process this: {input}",
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "output_schema": {"type": "object"},
            "metadata": {"version": "1.0", "author": "test"}
        }
        prompt = Prompt.from_dict(data)
        
        self.assertEqual(prompt.id, "test/prompt")
        self.assertEqual(prompt.system_instructions, "You are a helpful assistant")
        self.assertEqual(prompt.user_prompt_template, "Process this: {input}")
        self.assertEqual(prompt.output_schema, {"type": "object"})
        self.assertEqual(prompt.metadata, {"version": "1.0", "author": "test"})
        self.assertEqual(prompt.llm.temperature, 0.7)

    def test_render_user_prompt(self):
        """Test rendering user prompt template with kwargs."""
        prompt = Prompt(
            id="test/prompt",
            system_instructions="Instructions",
            user_prompt_template="Hello {name}, you have {count} messages",
            llm=LLMConfig(provider="openai", model="gpt-4")
        )
        
        result = prompt.render_user_prompt(name="Alice", count=5)
        self.assertEqual(result, "Hello Alice, you have 5 messages")

    def test_render_user_prompt_complex(self):
        """Test rendering complex user prompt template."""
        prompt = Prompt(
            id="test/prompt",
            system_instructions="Instructions",
            user_prompt_template="Context: {context}\nOriginal: {original}\nReplacements: {replacements}",
            llm=LLMConfig(provider="openai", model="gpt-4")
        )
        
        result = prompt.render_user_prompt(
            context="test sentence",
            original="word",
            replacements=["replacement1", "replacement2"]
        )
        expected = "Context: test sentence\nOriginal: word\nReplacements: ['replacement1', 'replacement2']"
        self.assertEqual(result, expected)

    def test_to_dict(self):
        """Test converting Prompt to dict."""
        prompt = Prompt(
            id="test/prompt",
            system_instructions="Instructions",
            user_prompt_template="Template {arg}",
            llm=LLMConfig(provider="openai", model="gpt-4", temperature=0.7),
            output_schema={"type": "string"},
            metadata={"version": "1.0"}
        )
        
        result = prompt.to_dict()
        
        self.assertEqual(result["id"], "test/prompt")
        self.assertEqual(result["system_instructions"], "Instructions")
        self.assertEqual(result["user_prompt_template"], "Template {arg}")
        self.assertEqual(result["output_schema"], {"type": "string"})
        self.assertEqual(result["metadata"], {"version": "1.0"})
        self.assertIsInstance(result["llm"], dict)
        self.assertEqual(result["llm"]["model"], "gpt-4")
        self.assertEqual(result["llm"]["temperature"], 0.7)

    def test_from_yaml(self):
        """Test creating Prompt from YAML string."""
        yaml_str = """
id: test/prompt
system_instructions: Test instructions
user_prompt_template: "Template {arg}"
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7
output_schema:
  type: object
metadata:
  version: "1.0"
"""
        prompt = Prompt.from_yaml(yaml_str)
        
        self.assertEqual(prompt.id, "test/prompt")
        self.assertEqual(prompt.system_instructions, "Test instructions")
        self.assertEqual(prompt.user_prompt_template, "Template {arg}")
        self.assertEqual(prompt.llm.model, "gpt-4")
        self.assertEqual(prompt.llm.temperature, 0.7)
        self.assertEqual(prompt.output_schema, {"type": "object"})
        self.assertEqual(prompt.metadata, {"version": "1.0"})

    def test_to_yaml(self):
        """Test converting Prompt to YAML string."""
        prompt = Prompt(
            id="test/prompt",
            system_instructions="Instructions",
            user_prompt_template="Template",
            llm=LLMConfig(provider="openai", model="gpt-4", temperature=0.7),
            output_schema={"type": "string"},
            metadata={"version": "1.0"}
        )
        
        yaml_str = prompt.to_yaml()
        parsed = yaml.safe_load(yaml_str)
        
        self.assertEqual(parsed["id"], "test/prompt")
        self.assertEqual(parsed["system_instructions"], "Instructions")
        self.assertEqual(parsed["llm"]["model"], "gpt-4")
        self.assertEqual(parsed["llm"]["temperature"], 0.7)

    def test_roundtrip_yaml(self):
        """Test that converting to YAML and back preserves data."""
        original = Prompt(
            id="test/prompt",
            system_instructions="Test instructions",
            user_prompt_template="Template {arg}",
            llm=LLMConfig(provider="openai", model="gpt-4", temperature=0.7, max_tokens=1000),
            output_schema={"type": "object"},
            metadata={"version": "1.0", "author": "test"}
        )
        
        yaml_str = original.to_yaml()
        restored = Prompt.from_yaml(yaml_str)
        
        self.assertEqual(restored.id, original.id)
        self.assertEqual(restored.system_instructions, original.system_instructions)
        self.assertEqual(restored.user_prompt_template, original.user_prompt_template)
        self.assertEqual(restored.llm.model, original.llm.model)
        self.assertEqual(restored.llm.temperature, original.llm.temperature)
        self.assertEqual(restored.llm.max_tokens, original.llm.max_tokens)
        self.assertEqual(restored.output_schema, original.output_schema)
        self.assertEqual(restored.metadata, original.metadata)

    def test_render_with_missing_kwarg(self):
        """Test that rendering with missing kwarg raises KeyError."""
        prompt = Prompt(
            id="test/prompt",
            system_instructions="Instructions",
            user_prompt_template="Hello {name}",
            llm=LLMConfig(provider="openai", model="gpt-4")
        )
        
        with self.assertRaises(KeyError):
            prompt.render_user_prompt(wrong_key="value")


if __name__ == '__main__':
    unittest.main()
