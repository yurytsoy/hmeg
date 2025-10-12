from __future__ import annotations

import atexit
import language_tool_python as ltp
import requests
import spacy

from .reranker import Reranker
from .usecases import is_port_in_use
from .vocabulary import Vocabulary


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

    ports = [ltp.LanguageTool._MIN_PORT + k for k in range(10)]
    for port in ports:
        if is_port_in_use(port) and is_lt_server_running(port):
            return ltp.LanguageTool('en-US', remote_server=f"localhost:{port}")
        else:
            return ltp.LanguageTool('en-US', host='localhost')
    raise RuntimeError(f"All ports are in use by other services (tested ports: {ports})")


language_tool = get_language_tool()


####################################################
# Cleanup
def on_exit():
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
    Processes found matches by:
    * Filtering out replacements outside of the vocabulary.
    * Ranking replacements according to the current reranker.
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
    Heuristic removal of suggested replacements.
    # * if original text is not all caps, then remove all-caps suggestions
    # * remove suggestions, which are not part of the vocabulary.
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
        if all(token.lemma_ in vocab for token in doc):
            res.append(item)
    return res
