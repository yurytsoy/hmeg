from __future__ import annotations

import atexit
import language_tool_python as ltp
import requests
import spacy

from .reranker import Reranker
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
    def is_lt_server_running():
        LT_URL = f"http://localhost:{LT_PORT}/v2/check"

        try:
            args = "text=foo&language=en"
            response = requests.post(f"{LT_URL}?{args}", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    LT_PORT = 8081  # fixme: what if LT is running on a different port?

    if is_lt_server_running():
        return ltp.LanguageTool('en-US', remote_server=f"localhost:{LT_PORT}")
    else:
        return ltp.LanguageTool('en-US', host='localhost')


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

    # only keep replacements from the vocabulary
    replacements = [item for item in replacements if nlp(item)[0].lemma_ in vocab]
    return replacements
