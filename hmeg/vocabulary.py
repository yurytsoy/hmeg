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
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    cities = [
        "Beijing", "Berlin", "Busan", "Daejeon", "London", "New York", "Paris", "Seoul", "Tokyo"
    ]
    countries = [
        "Australia", "Brazil", "Canada", "China", "England", "France", "Germany", "India", "Japan",
        "Korea", "Philippines", "Russia", "Thailand", "Vietnam", "USA"
    ]
    nationalities = [
        "Australian", "Brazilian", "Canadian", "Chinese", "English", "French", "German", "Indian", "Japanese",
        "Korean", "Filipino", "Russian", "Thai", "Vietnamese", "American"
    ]
    person_nouns = [
        "anybody", "boy", "brother", "child", "daughter", "everybody",
        "family", "father", "friend", "girl", "husband", "lady", "man", "mother",
        "nobody", "people", "person", "relative", "sister", "somebody", "son",
        "teacher", "wife", "woman"
    ]
    places = [
        "airport", "apartment", "country", "flat", "home", "house", "road", "room",
        "school", "shop", "side", "street", "town", "work",
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

    def random_noun_non_person(self) -> str:
        res = random.choice(self.nouns)
        while res in Vocabulary.person_nouns:
            res = random.choice(self.nouns)
        return res

    def random_adjective(self) -> str:
        return random.choice(self.adjectives)

    def random_adverb(self) -> str:
        return random.choice(self.adverbs)

    def random_weekday(self) -> str:
        return random.choice(self.weekdays)

    def random_season(self) -> str:
        return random.choice(self.seasons)

    def random_month(self) -> str:
        return random.choice(self.months)

    def random_number(self, max) -> str:
        return str(random.randint(0, max))

    def random_place(self) -> str:
        return random.choice(self.places)

    def random_city(self) -> str:
        return random.choice(self.cities)

    def random_country(self) -> str:
        return random.choice(self.countries)

    def random_nationality(self) -> str:
        return random.choice(self.nationalities)
