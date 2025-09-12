"""
Class for managing interchangeable vocabulary.
"""

from __future__ import annotations

import inflect
import numpy as np
import os
import random
import toml

from .entities import VOWELS
from .verb_conjugator import VerbConjugator


p = inflect.engine()

# dictionary with fixes for conjugation of the mlconjug3
verbs_past = {
    'borrow': 'borrowed',
    'look for': 'looked for',
}


class Vocabulary:
    vocab_file: str | None  # file with the current vocabulary
    name: str | None
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
        "airport", "apartment", "country", "flat", "home", "hospital", "house", "road", "room",
        "school", "shop", "side", "street", "town", "work", "bathroom"
    ]

    def __init__(self, vocab_file: str | None = None):
        self.vocab_file = vocab_file
        self.name = None
        self.adjectives = []
        self.adverbs = []
        self.nouns = []
        self.verbs = []
        self._load()

    @staticmethod
    def load(vocab_file: str) -> Vocabulary:
        return Vocabulary(vocab_file)

    def _load(self):
        if self.vocab_file is None:
            return

        with open(self.vocab_file, "r") as f:
            vocab_dict = toml.loads(f.read())

        if "import" in vocab_dict:
            import_dir = os.path.split(self.vocab_file)[0]
            import_vocab = Vocabulary.load(os.path.join(import_dir, vocab_dict["import"]))
        else:
            import_vocab = Vocabulary()

        self.name = vocab_dict.get("name")
        self.adjectives = sorted(set(import_vocab.adjectives + (vocab_dict.get("adjectives") or [])))
        self.adverbs = sorted(set(import_vocab.adverbs + (vocab_dict.get("adverbs") or [])))
        self.nouns = sorted(set(import_vocab.nouns + (vocab_dict.get("nouns") or [])))
        self.verbs = sorted(set(import_vocab.verbs + (vocab_dict.get("verbs") or [])))
        self.set_ = set(self.adjectives).union(self.adverbs).union(self.nouns).union(self.verbs)

    def __contains__(self, word: str) -> bool:
        return word.lower() in self.set_

    def random_verb(self) -> str:
        return random.choice(self.verbs)

    def random_verb_singular_3rd(self) -> str:
        return VerbConjugator.present_singular(self.random_verb())

    def random_verb_past(self) -> str:
        return VerbConjugator.past_simple(self.random_verb())

    def random_verb_progressive(self) -> str:
        return VerbConjugator.continuous(self.random_verb())

    def random_noun(self) -> str:
        return random.choice(self.nouns)

    def random_anoun(self) -> str:
        noun = self.random_noun()
        article = "an" if noun[0] in VOWELS else "a"
        return f"{article} {noun}"

    def random_noun_plural(self) -> str:
        noun = random.choice(self.nouns)
        return p.plural_noun(noun)

    def random_noun_non_person(self) -> str:
        res = random.choice(self.nouns)
        while res in Vocabulary.person_nouns:
            res = random.choice(self.nouns)
        return res

    def random_anoun_non_person(self) -> str:
        noun = self.random_noun_non_person()
        article = "an" if noun[0] in VOWELS else "a"
        return f"{article} {noun}"

    def random_person(self) -> str:
        res = random.choice(self.person_nouns)
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

    def random_number_large(self) -> str:
        # ranges:
        # 1 -- 5k -- 5M -- regular prices
        # 2 -- 3M -- 50M -- car price
        # 3 -- 20M -- 5B -- housing range (loan, purchase)
        # 4 -- 50M -- 10B -- small business related
        # 5 -- 1B -- 100B -- medium business related
        category = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.2, 0.15, 0.1, 0.05])
        mins = {
            1: 5000,
            2: 3 * 10**6,
            3: 20 * 10 ** 6,
            4: 50 * 10**6,
            5: 10**9,
        }
        maxes = {
            1: 5 * 10**6,
            2: 50 * 10**6,
            3: 5 * 10**9,
            4: 10 * 10**9,
            5: 100 * 10**9
        }
        res = random.randint(mins[category], maxes[category])
        return f"{res:,}"

    def random_place(self) -> str:
        return random.choice(self.places)

    def random_city(self) -> str:
        return random.choice(self.cities)

    def random_country(self) -> str:
        return random.choice(self.countries)

    def random_nationality(self) -> str:
        return random.choice(self.nationalities)


def load_minilex() -> Vocabulary:
    cur_dir = os.path.split(__file__)[0]
    return Vocabulary.load(os.path.join(cur_dir, "vocabs", "minilex.toml"))


def load_nanolex() -> Vocabulary:
    cur_dir = os.path.split(__file__)[0]
    return Vocabulary.load(os.path.join(cur_dir, "vocabs", "nanolex.toml"))
