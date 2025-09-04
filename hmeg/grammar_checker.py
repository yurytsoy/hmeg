from __future__ import annotations

import language_tool_python as ltp
import spacy

from .vocabulary import Vocabulary


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


tool = ltp.LanguageTool('en-US')


class GrammarChecker:
    @staticmethod
    def correct_phrases(phrases: list[str], vocab: Vocabulary) -> list[str]:
        """
        Checks grammar in the generated list of exercises.
        Returns a list of fixed exercises.
        """

        res = []
        for phrase in phrases:
            matches = tool.check(phrase)
            matches = fix_matches(matches, vocab)
            res.append(ltp.utils.correct(phrase, matches))
        return res


def fix_matches(matches: list[ltp.Match], vocab: Vocabulary) -> list[ltp.Match]:
    """
    Heuristic removal of suggested replacements.

    # * if original text is not all caps, then remove all-caps suggestions
    # * remove suggestions, which are not part of the vocabulary.
    # * TODO: part of speech consistency check
    # * TODO: sort by the Levenshtein distance
    #   * ? remove suggestions, which are too far-away from the original text.
    # * TODO: ? keep only top-N replacements
    # * TODO: ? prefer shorter replacements, because those are simpler
    """

    for match in matches:
        match.replacements = filter_replacements(match.matchedText, match.replacements, vocab)
    return matches


def filter_replacements(original: str, replacements: list[str], vocab: Vocabulary) -> list[str]:
    """

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

    # TODO: sort replacements based on the Levenshtein distance

    return replacements
