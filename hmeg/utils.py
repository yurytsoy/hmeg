from functools import partial
import os
import re
import toml

from .entities import GrammarDescription, VocabularyPlaceholders, TopicLevelInfo
from .registry import GrammarRegistry
from .vocabulary import Vocabulary


def register_miniphrase():
    cur_dir = os.path.split(__file__)[0]
    miniphrase_dir = os.path.join(cur_dir, "miniphrase")
    register_grammar_topics(miniphrase_dir)


def register_grammar_topics(grammar_dir: str | None = None):
    """
    Read and register descriptions of grammar exercises.
    """

    cur_dir = os.path.split(__file__)[0]
    default_grammar_dir = os.path.join(cur_dir, "topics")
    grammar_dir = grammar_dir or default_grammar_dir

    # iterate over files in `grammar_dir`, load descriptions of topics and exercises and register them.
    for file in os.listdir(grammar_dir):
        if not file.endswith(".toml"):
            continue
        with open(os.path.join(grammar_dir, file), "r") as f:
            grammar_descr_dict = toml.loads(f.read())
            grammar_descr = GrammarDescription(
                name=grammar_descr_dict["name"],
                links=grammar_descr_dict["links"],
                exercises=grammar_descr_dict["exercises"],
                levels=[TopicLevelInfo(**level_descr) for level_descr in grammar_descr_dict.get("levels", [])],
            )
            GrammarRegistry.register_grammar_topic(grammar_descr)


def apply_vocabulary(s: str, vocab: Vocabulary) -> str:
    """
    Takes input string and replaces placeholders with respective minilex entities.
    """

    vocab_function = {
        VocabularyPlaceholders.Verb: vocab.random_verb,
        VocabularyPlaceholders.VerbSingular3rd: vocab.random_verb_singular_3rd,
        VocabularyPlaceholders.VerbPast: vocab.random_verb_past,
        VocabularyPlaceholders.VerbProgressive: vocab.random_verb_progressive,
        VocabularyPlaceholders.Noun: vocab.random_noun,
        VocabularyPlaceholders.ANoun: vocab.random_anoun,
        VocabularyPlaceholders.NounPlural: vocab.random_noun_plural,
        VocabularyPlaceholders.NounNonPerson: vocab.random_noun_non_person,
        VocabularyPlaceholders.ANounNonPerson: vocab.random_anoun_non_person,
        VocabularyPlaceholders.Number100: partial(vocab.random_number, max=100),
        VocabularyPlaceholders.Number1000: partial(vocab.random_number, max=1000),
        VocabularyPlaceholders.Number100k: partial(vocab.random_number, max=100_000),
        VocabularyPlaceholders.Weekday: partial(vocab.random_weekday),
        VocabularyPlaceholders.Season: partial(vocab.random_season),
        VocabularyPlaceholders.Month: partial(vocab.random_month),
        VocabularyPlaceholders.Adjective: partial(vocab.random_adjective),
        VocabularyPlaceholders.Adverb: partial(vocab.random_adverb),
        VocabularyPlaceholders.Country: partial(vocab.random_country),
        VocabularyPlaceholders.Place: partial(vocab.random_place),
        VocabularyPlaceholders.City: partial(vocab.random_city),
        VocabularyPlaceholders.Nationality: partial(vocab.random_nationality),
    }

    placeholder_patterns = re.compile("|".join(list(vocab_function)))
    for match in placeholder_patterns.findall(s):
        s = placeholder_patterns.sub(
            vocab_function[match](), string=s, count=1
        )

    return s
