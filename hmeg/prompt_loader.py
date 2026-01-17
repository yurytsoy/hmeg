from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from hmeg.entities import Prompt


class PromptLoader:
    """
    Load prompt YAML files by prompt ID from the `hmeg/prompts` directory.

    Example:
        loader = PromptLoader()  # uses package `hmeg/prompts`
        prompt = loader.load("v1/reranker/openai")
    """

    def __init__(self, base_dir: Path | str | None = None) -> None:
        if base_dir is None:
            # default to the prompts folder next to this file: hmeg/prompts
            base_dir = Path(__file__).parent / "prompts"
        self.base_dir = Path(base_dir).resolve()
        self._cache: dict[str, Any] = {}
        self._prompt_schema: dict[str, Any] | None = None

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

    def validate_prompt_schema(self, prompt_dict: dict[str, Any]):
        # load schema YAML from prompts directory
        if not self._prompt_schema:
            schema_path = self.base_dir / "schema.yaml"
            if not schema_path.exists():
                raise FileNotFoundError(f"Prompt schema file not found: {schema_path}")

            # validate using jsonschema
            with schema_path.open("r", encoding="utf-8") as sf:
                self._prompt_schema = yaml.safe_load(sf) or {}

        try:
            def _coerce_dates_to_iso(obj):
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                if isinstance(obj, dict):
                    return {k: _coerce_dates_to_iso(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_coerce_dates_to_iso(v) for v in obj]
                return obj

            validation_dict = _coerce_dates_to_iso(prompt_dict)
            jsonschema.validate(instance=validation_dict, schema=self._prompt_schema)

        except jsonschema.ValidationError as exc:
            raise ValueError(f"Prompt failed schema validation: {exc.message}") from exc

    def load(self, prompt_id: str) -> Prompt:
        """
        Load and parse YAML for the given prompt_id.
        Returns a Prompt instance loaded from the YAML file.
        Raises FileNotFoundError if the file does not exist.
        """
        if prompt_id in self._cache:
            return self._cache[prompt_id]

        path = self._prompt_path(prompt_id)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found for id '{prompt_id}': {path}")

        # read prompt YAML as dict for validation
        with path.open("r", encoding="utf-8") as f:
            prompt_dict = yaml.safe_load(f) or {}

        self.validate_prompt_schema(prompt_dict)

        # construct Prompt from dict and cache it
        prompt = Prompt.from_dict(prompt_dict)
        self._cache[prompt_id] = prompt
        return prompt

    def clear_cache(self, prompt_id: str | None = None) -> None:
        """Clear cached entry or entire cache."""
        if prompt_id is None:
            self._cache.clear()
        else:
            self._cache.pop(prompt_id, None)

