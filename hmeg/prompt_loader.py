from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from hmeg.entities import Prompt


class PromptLoader:
    """
    Load prompt YAML files by prompt ID from the `hmeg/prompts` directory.

    Example:
        loader = PromptLoader()  # uses package `hmeg/prompts`
        prompt = loader.load("v1/reranker/openai")
    """

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        if base_dir is None:
            # default to the prompts folder next to this file: hmeg/prompts
            base_dir = Path(__file__).parent / "prompts"
        self.base_dir = Path(base_dir).resolve()
        self._cache: Dict[str, Any] = {}

    def _prompt_path(self, prompt_id: str) -> Path:
        # prompt_id like "v1/reranker/openai" -> hmeg/prompts/v1/reranker/openai.yaml
        parts = [p for p in Path(prompt_id).parts if p not in ("", ".")]
        path = self.base_dir.joinpath(*parts).with_suffix(".yaml")
        # prevent path traversal by ensuring resolved path is under base_dir
        resolved = path.resolve()
        try:
            resolved.relative_to(self.base_dir)
        except Exception as exc:
            raise ValueError(f"Prompt id looks outside base prompts directory: {prompt_id}") from exc
        return resolved

    def load(self, prompt_id: str) -> Prompt:
        """
        Load and parse YAML for the given prompt_id.
        Returns the parsed YAML (usually a dict).
        Raises FileNotFoundError if the file does not exist.
        """
        if prompt_id in self._cache:
            return self._cache[prompt_id]

        path = self._prompt_path(prompt_id)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found for id '{prompt_id}': {path}")


        with path.open("r", encoding="utf-8") as f:
            prompt = Prompt.from_yaml(f.read())

        self._cache[prompt_id] = prompt
        return prompt

    def clear_cache(self, prompt_id: Optional[str] = None) -> None:
        """Clear cached entry or entire cache."""
        if prompt_id is None:
            self._cache.clear()
        else:
            self._cache.pop(prompt_id, None)

