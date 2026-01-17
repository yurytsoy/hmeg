from dataclasses import dataclass, field
from typing import Any
import yaml


@dataclass
class LLMConfig:
    """
    Configuration for a language model used by a prompt.

    Attributes:
        provider: Provider identifier (e.g. "openai").
        model: Model name or identifier.
        temperature: Sampling temperature (or None to use provider default).
        max_tokens: Optional maximum tokens for generation.
        verbatim: Additional provider-specific fields preserved when serializing.

    Methods:
        from_dict: Construct an LLMConfig from a mapping, preserving unknown keys in `verbatim`.
        to_dict: Serialize the configuration to a mapping suitable for YAML/JSON.
    """

    provider: str
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    verbatim: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LLMConfig":
        return cls(
            provider=d["provider"],
            model=d["model"],
            temperature=d.get("temperature"),
            max_tokens=d.get("max_tokens"),
            verbatim={k: v for k, v in d.items() if k not in {"provider", "model", "temperature", "max_tokens"}},
        )

    def to_dict(self) -> dict[str, Any]:
        base = {
            "provider": self.provider,
            "model": self.model,
        }
        if self.temperature is not None:
            base["temperature"] = self.temperature
        if self.max_tokens is not None:
            base["max_tokens"] = self.max_tokens
        base.update(self.verbatim)
        return base


@dataclass
class Prompt:
    """
    Representation of a prompt bundle loaded from YAML.

    Attributes:
        id: Unique prompt identifier (matches file path under `hmeg/prompts`).
        system_instructions: Instructions injected into the system role.
        user_prompt_template: Template string used to render the user prompt.
        llm: LLMConfig instance describing the target model.
        output_schema: Optional JSON schema describing expected model output.
        metadata: Arbitrary metadata loaded from the prompt file.

    Methods:
        render_user_prompt: Format the `user_prompt_template` with provided keyword arguments.
        from_dict / to_dict: Convert between mapping and Prompt instance for (de)serialization.
        from_yaml / to_yaml: Convenience helpers to load/dump YAML text.
    """

    id: str
    system_instructions: str
    user_prompt_template: str
    llm: LLMConfig
    output_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def render_user_prompt(self, **kwargs: Any) -> str:
        """
        Render the `user_prompt_template` with provided keyword args (e.g. context, original, replacements).
        """
        return self.user_prompt_template.format(**kwargs)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prompt":
        llm_cfg = LLMConfig.from_dict(d.get("llm", {}))
        return cls(
            id=d["id"],
            system_instructions=d.get("system_instructions", ""),
            user_prompt_template=d.get("user_prompt_template", ""),
            llm=llm_cfg,
            output_schema=d.get("output_schema"),
            metadata=d.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "system_instructions": self.system_instructions,
            "user_prompt_template": self.user_prompt_template,
            "llm": self.llm.to_dict(),
            "output_schema": self.output_schema,
            "metadata": self.metadata,
        }

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "Prompt":
        return cls.from_dict(yaml.safe_load(yaml_text))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)
