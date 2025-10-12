from __future__ import annotations

import atexit
import language_tool_python as ltp
import requests
import spacy

from .reranker import Reranker
from .usecases import is_port_in_use
from .vocabulary import Vocabulary


# In the cases when suggested replacement contains "doesn't", "haven't" etc, it is broken in 2 tokens,
#   one of which is "n't", which: (1) does not belong to the vocabulary as it is not a word;
#   (2) not required for lemmatization anyways, and thus can be ignored.
IGNORE_LEMMATIZATION_TOKENS = ["n't", "n\u2019t", "\u2019t"]


####################################################
# Init NLP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


####################################################
# Init local Language Tool server
def get_language_tool():
    """
    Get or create `LanguageTool` instance.
    """

    def is_lt_server_running(port: int):
        LT_URL = f"http://localhost:{port}/v2/check"

        try:
            args = "text=foo&language=en"
            response = requests.post(f"{LT_URL}?{args}", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    # Note: the code below inherently assumes that LT is trying to use ports starting from `LanguageTool._MIN_PORT`
    ports = [ltp.LanguageTool._MIN_PORT + k for k in range(10)]
    for port in ports:
        if is_port_in_use(port):
            if is_lt_server_running(port):
                return ltp.LanguageTool('en-US', remote_server=f"localhost:{port}")
            # Port is in use but not running LanguageTool, try next port
            continue

        # Port is free, try to start a new LanguageTool server here.
        return ltp.LanguageTool('en-US', host='localhost')
    raise RuntimeError(f"All ports are in use by other services (tested ports: {ports})")


language_tool = get_language_tool()


####################################################
# Cleanup
def on_exit():
    if "language_tool" in globals():
        language_tool.close()

atexit.register(on_exit)


####################################################

class GrammarChecker:
    @staticmethod
    def correct_phrases(phrases: list[str], vocab: Vocabulary) -> list[str]:
        """
        Checks grammar and spelling in the provided list of phrases.
        The correction is performed wrt to the provided vocabulary, so that only the vocabulary words can appear in
            the corrected phrase.

        Returns a list of fixed phrases.
        """

        res = []
        for phrase in phrases:
            matches = language_tool.check(phrase)
            matches = fix_and_rank_matches(matches, vocab)
            res.append(ltp.utils.correct(phrase, matches))

        return res


def fix_and_rank_matches(matches: list[ltp.Match], vocab: Vocabulary, reranker_model: str | None = None) -> list[ltp.Match]:
    """
    Filters out suggested replacements that are not in the provided vocabulary.

    Filters replacements for each match so only words present in `vocab` remain, then ranks the
    remaining replacements using the specified reranker model. The function mutates the input
    `matches` in-place (each match's `replacements` is updated).

    Parameters
    ----------
    matches : list[ltp.Match]
        List of matches detected by the language tool.
    vocab : Vocabulary
        Vocabulary object used to filter valid replacements.
    reranker_model : str | None, optional
        Name of the reranker model to use for ranking replacements. Defaults to None
        (uses `Reranker.model_name_`).

    Returns
    -------
    list[ltp.Match]
        The list of matches with their replacements filtered to only include vocabulary words,
        and ranked according to the reranker model.

    Notes
    -----
    - Ranks the remaining replacements using the specified reranker model.
    """

    reranker_model = reranker_model or Reranker.model_name_
    Reranker.set_current_model(reranker_model)

    for match in matches:
        cur_replacements = filter_replacements(match.matchedText, match.replacements, vocab)
        ranked_replacements = Reranker.rank(context=match.context, original=match.matchedText, replacements=cur_replacements)
        match.replacements = [replacement for replacement, score in ranked_replacements]

    return matches


def filter_replacements(original: str, replacements: list[str], vocab: Vocabulary) -> list[str]:
    """
    Heuristic filtering of suggested replacements.

    Applies a small set of heuristics to remove unlikely replacements before ranking:
    - If `original` is not all upper-case, drop suggestions that are all upper-case; otherwise keep only
      all upper-case suggestions.
    - Tokenize each candidate with the module-level `nlp` pipeline and form a list of token lemmas,
      ignoring tokens listed in the module-level `IGNORE_LEMMATIZATION_TOKENS` (for example, `n't`).
    - Keep a replacement only if every remaining token lemma is present in `vocab`.

    Parameters
    ----------
    original : str
        The original part of the context that is considered for replacement.
    replacements : list[str]
        Candidate replacement strings.
    vocab : Vocabulary
        Vocabulary object that restricts acceptable replacements.

    Returns
    -------
    list[str]
        A new list containing replacements that passed the heuristics. The input `replacements`
        list is not modified.

    Notes
    -----
    - Tokenization and lemmatization are performed with the module-level `nlp` pipeline.
    - Tokens whose text appears in `IGNORE_LEMMATIZATION_TOKENS` are excluded from vocabulary checks.
    - The function is conservative: if no replacements remain after filtering, an empty list is returned.
    """

    if not replacements:
        return []

    if not original:
        return replacements

    is_all_caps = original.upper() == original
    if not is_all_caps:
        # keep only non-all-caps from the replacements
        replacements = [item for item in replacements if item.upper() != item]
    else:
        # keep only all-caps from the replacements
        replacements = [item for item in replacements if item.upper() == item]

    if not replacements:
        return []

    # only keep replacements that match the vocabulary.
    docs = list(nlp.pipe(replacements))
    res = []
    for item, doc in zip(replacements, docs):
        lemmas = [tok.lemma_ for tok in doc if tok.text.lower() not in IGNORE_LEMMATIZATION_TOKENS]
        if all(lemma in vocab for lemma in lemmas):
            res.append(item)
    return res
