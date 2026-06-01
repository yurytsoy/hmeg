import os
from pathlib import Path

from hmeg.vocabulary import Vocabulary

base_dir = Path(__file__).parent.parent.parent
DEFAULT_VOCABULARY_FILE = os.path.join(base_dir, "vocabs/minilex.toml")


def generate_exercises(topic_name: str, num: int, vocab: Vocabulary | None = None) -> list[str]:
    """
    Generates list of random translation exercises for the given topic using legacy approach (pre-LLM).
    The generation proceeds in 2 steps:
    1. Generate list of templates wrt selected grammar topic. The result contains
       placeholders for nouns, verbs, ....
    2. Fill-in placeholders according to the given vocabulary.

    See also: `apply_vocabulary`

    Parameters
    ----------
    topic_name: str
        The name of the topic to generate exercises for.
    num: int
        The number of exercises to generate.
    vocab: Vocabulary, default=None
        Vocabulary for words, that can be used for generating exercises.
        If `None` then vocabulary from the `DEFAULT_VOCABULARY_FILE` is used.

    Returns
    -------
    list[str]
        A list of generated exercise sentences (English strings).
    """
    import random

    from nltk import CFG
    from nltk.parse.generate import generate

    from hmeg.grammar_registry import GrammarRegistry
    from hmeg.usecases import apply_vocabulary

    vocab = vocab or Vocabulary.load(DEFAULT_VOCABULARY_FILE)

    if topic_name not in GrammarRegistry.topics:
        raise RuntimeError(
            f"Requested an unregistered topic: {topic_name}. Please run `python hmeg_cli.py list` to see the existing topics.")

    templates = []
    for exercise_type in GrammarRegistry.topics[topic_name].exercises:
        cur_grammar = CFG.fromstring(exercise_type)
        templates.extend(generate(cur_grammar, n=num))

    res = []
    num_trials = 0
    while len(res) < num:
        cur_idx = random.randint(0, len(templates) - 1)
        exercise = apply_vocabulary(" ".join(templates[cur_idx]), vocab)
        exercise = exercise.replace(exercise[0], exercise[0].capitalize(), 1)
        if exercise not in res:
            res.append(exercise)
        num_trials += 1
        if num_trials > num ** 2:
            break
    return res

