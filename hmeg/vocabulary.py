"""
Class for managing interchangeable vocabulary.
"""

from __future__ import annotations

import random
import tomllib


class Vocabulary:
    vocab_file: str  # file with the current vocabulary
    adjectives: list[str]
    adverbs: list[str]
    nouns: list[str]
    verbs: list[str]
    weekdays: list[str] = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]
    seasons = [
        "Spring", "Summer", "Autumn", "Winter"
    ]

    def __init__(self, vocab_file: str):
        self.vocab_file = vocab_file
        self._load()

    @staticmethod
    def load(vocab_file: str) -> Vocabulary:
        return Vocabulary(vocab_file)

    def _load(self):
        with open(self.vocab_file, "r") as f:
            vocab_dict = tomllib.loads(f.read())
            self.adjectives = vocab_dict.get("adjectives") or []
            self.adverbs = vocab_dict.get("adverbs") or []
            self.nouns = vocab_dict.get("nouns") or []
            self.verbs = vocab_dict.get("verbs") or []

    def random_verb(self) -> str:
        return random.choice(self.verbs)

    def random_noun(self) -> str:
        return random.choice(self.nouns)

    def random_adjective(self) -> str:
        return random.choice(self.adjectives)

    def random_adverb(self) -> str:
        return random.choice(self.adverbs)

    def random_weekday(self) -> str:
        return random.choice(self.weekdays)

    def random_season(self) -> str:
        return random.choice(self.seasons)
