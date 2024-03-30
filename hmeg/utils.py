from mlconjug3 import Conjugator
import os
import toml

from .entities import GrammarDescription, VocabularyPlaceholders, VOWELS
from .inflections import verbs_singular_3rd
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
            )
            GrammarRegistry.register_grammar_topic(grammar_descr)


def apply_vocabulary(s: str, vocab: Vocabulary) -> str:
    """
    Takes input string and replaces placeholders with respective minilex entities.
    """

    while VocabularyPlaceholders.Verb in s:
        s = s.replace(VocabularyPlaceholders.Verb, vocab.random_verb(), 1)

    while VocabularyPlaceholders.Noun in s:
        s = s.replace(VocabularyPlaceholders.Noun, vocab.random_noun(), 1)

    while VocabularyPlaceholders.NounPlural in s:
        s = s.replace(VocabularyPlaceholders.NounPlural, vocab.random_noun_plural(), 1)

    while VocabularyPlaceholders.NounNonPerson in s:
        s = s.replace(VocabularyPlaceholders.NounNonPerson, vocab.random_noun_non_person(), 1)

    while VocabularyPlaceholders.Number100 in s:
        s = s.replace(VocabularyPlaceholders.Number100, vocab.random_number(max=100), 1)

    while VocabularyPlaceholders.Number1000 in s:
        s = s.replace(VocabularyPlaceholders.Number1000, vocab.random_number(max=1000), 1)

    while VocabularyPlaceholders.Number100k in s:
        s = s.replace(VocabularyPlaceholders.Number100k, vocab.random_number(max=100_000), 1)

    while VocabularyPlaceholders.ANoun in s:
        noun = vocab.random_noun()
        suffix = "an" if noun[0] in VOWELS else "a"
        s = s.replace(VocabularyPlaceholders.ANoun, f"{suffix} {noun}", 1)

    while VocabularyPlaceholders.ANounNonPerson in s:
        noun = vocab.random_noun_non_person()
        suffix = "an" if noun[0] in VOWELS else "a"
        s = s.replace(VocabularyPlaceholders.ANounNonPerson, f"{suffix} {noun}", 1)

    while VocabularyPlaceholders.Weekday in s:
        s = s.replace(VocabularyPlaceholders.Weekday, vocab.random_weekday(), 1)

    while VocabularyPlaceholders.Season in s:
        s = s.replace(VocabularyPlaceholders.Season, vocab.random_season(), 1)

    while VocabularyPlaceholders.Month in s:
        s = s.replace(VocabularyPlaceholders.Month, vocab.random_month(), 1)

    while VocabularyPlaceholders.Adjective in s:
        s = s.replace(VocabularyPlaceholders.Adjective, vocab.random_adjective(), 1)

    while VocabularyPlaceholders.Adverb in s:
        s = s.replace(VocabularyPlaceholders.Adverb, vocab.random_adverb(), 1)

    while VocabularyPlaceholders.Country in s:
        s = s.replace(VocabularyPlaceholders.Country, vocab.random_country(), 1)

    while VocabularyPlaceholders.Place in s:
        s = s.replace(VocabularyPlaceholders.Place, vocab.random_place(), 1)

    while VocabularyPlaceholders.City in s:
        s = s.replace(VocabularyPlaceholders.City, vocab.random_city(), 1)

    while VocabularyPlaceholders.Nationality in s:
        s = s.replace(VocabularyPlaceholders.Nationality, vocab.random_nationality(), 1)

    conj = Conjugator(language="en")
    while VocabularyPlaceholders.VerbSingular3rd in s:
        cur_verb = conj.conjugate(vocab.random_verb())
        conj_verb = verbs_singular_3rd.get(
            cur_verb.name,
            cur_verb["indicative"]["indicative present"]["he/she/it"]
        )
        s = s.replace(VocabularyPlaceholders.VerbSingular3rd, conj_verb, 1)

    while VocabularyPlaceholders.VerbPast in s:
        cur_verb = conj.conjugate(vocab.random_verb())
        conj_verb = cur_verb["indicative"]["indicative past tense"]["I"]
        s = s.replace(VocabularyPlaceholders.VerbPast, conj_verb, 1)

    while VocabularyPlaceholders.VerbProgressive in s:
        cur_verb = conj.conjugate(vocab.random_verb())
        conj_verb = cur_verb["indicative"]["indicative present continuous"]["I"]
        s = s.replace(VocabularyPlaceholders.VerbProgressive, conj_verb, 1)

    return s
