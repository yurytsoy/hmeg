from functools import partial
import os
import pandas as pd
import re
import toml

from .entities import GrammarDescription, VocabularyPlaceholders, VocabularyInfo
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
            grammar_descr = GrammarDescription.from_dict(grammar_descr_dict)
            GrammarRegistry.register_grammar_topic(grammar_descr)


def get_vocabulary_names() -> list[str]:
    """
    Returns list with names of the built-in vocabularies.
    """
    vocabs_dir = os.path.join(os.path.dirname(__file__), "vocabs")
    res = []
    for file in os.listdir(vocabs_dir):
        with open(os.path.join(vocabs_dir, file), "r") as f:
            vocab_info = toml.loads(f.read())
        res.append(vocab_info["name"])
    return res


def get_vocabularies_info() -> list[VocabularyInfo]:
    """
    Returns list with names of the built-in vocabularies.
    """
    vocabs_dir = os.path.join(os.path.dirname(__file__), "vocabs")
    res = []
    for file in os.listdir(vocabs_dir):
        vocab = Vocabulary(os.path.join(vocabs_dir, file))
        res.append(
            VocabularyInfo(
                name=vocab.name,
                num_words=len(vocab.nouns) + len(vocab.verbs) + len(vocab.adjectives) + len(vocab.adverbs),
                num_nouns=len(vocab.nouns),
                num_verbs=len(vocab.verbs),
                num_adjectives=len(vocab.adjectives),
                num_adverbs=len(vocab.adverbs),
            )
        )
    return res


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
        VocabularyPlaceholders.Person: vocab.random_person,
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


def split_vocabulary_top_n_words(input_vocab_file: str, top_n: int):
    """
    Splits input vocabulary into two non-overlapping vocabularies.
    The first contains top-N words having the highest usage frequency.
    And the second contains remaining words.

    Requires file with frequencies of words from here:
    * http://norvig.com/ngrams/count_1w100k.txt

    Args
    ----
    input_vocab_file: str
    top_n: int
        Keep top_n most frequent words.
    """

    input_vocab = Vocabulary(input_vocab_file)
    freq_df = pd.read_table("count_1w100k.txt", sep="\t", header=None)
    freq_info = freq_df.set_index(0).to_dict()[1]

    input_words = set(input_vocab.nouns + input_vocab.adjectives + input_vocab.adverbs + input_vocab.verbs)
    input_words_freq = [freq_info.get(word.upper(), 0) for word in input_words]
    filtered_words = pd.Series(index=input_words, data=input_words_freq).sort_values(ascending=False)[:top_n]

    res_vocab = Vocabulary()
    res_vocab.nouns = [word for word in input_vocab.nouns if word in filtered_words.index]
    res_vocab.adverbs = [word for word in input_vocab.adverbs if word in filtered_words.index]
    res_vocab.adjectives = [word for word in input_vocab.adjectives if word in filtered_words.index]
    res_vocab.verbs = [word for word in input_vocab.verbs if word in filtered_words.index]

    out_path = "out_vocab_incl.toml"
    vocab_dict = {
        "name": f"Included top-{top_n} words",
        "nouns": [word for word in input_vocab.nouns if word in filtered_words.index],
        "adverbs": [word for word in input_vocab.adverbs if word in filtered_words.index],
        "adjectives": [word for word in input_vocab.adjectives if word in filtered_words.index],
        "verbs": [word for word in input_vocab.verbs if word in filtered_words.index],
    }
    with open(out_path, "w") as f:
        toml.dump(vocab_dict, f)

    out_path = "out_vocab_excl.toml"
    vocab_dict = {
        "name": "Excluded words",
        "nouns": [word for word in input_vocab.nouns if word not in filtered_words.index],
        "adverbs": [word for word in input_vocab.adverbs if word not in filtered_words.index],
        "adjectives": [word for word in input_vocab.adjectives if word not in filtered_words.index],
        "verbs": [word for word in input_vocab.verbs if word not in filtered_words.index],
    }
    with open(out_path, "w") as f:
        toml.dump(vocab_dict, f)
