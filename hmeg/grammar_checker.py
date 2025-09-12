from __future__ import annotations

import language_tool_python as ltp
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        Checks grammar and spelling in the provided list of phrases.
        The correction is performed wrt to the provided vocabulary, so that only the vocabulary words can appear in
            the corrected phrase.

        Returns a list of fixed phrases.
        """

        res = []
        for phrase in phrases:
            matches = tool.check(phrase)
            matches = fix_matches(matches, vocab)
            res.append(ltp.utils.correct(phrase, matches))
        return res


def fix_matches(matches: list[ltp.Match], vocab: Vocabulary) -> list[ltp.Match]:
    """
    Process suggested matches with the vocabulary and return a list of fixed matches.
    """

    for match in matches:
        match.replacements = filter_replacements(match.matchedText, match.replacements, vocab)
    return matches


def rank_replacements(match: ltp.Match, method: str = "minilm") -> list[tuple[str, float]]:
    """
    Parameters
    ----------
    match : ltp.Match
        Information about a replacement.
    method : str
        Method used to rank replacements. The following methods are available:
        "minilm" -- nreimers/MiniLM-L6-H384-uncased, bidirectional language model, 22M parameters.
             Works better for the replacements at any  middle and at the end. More computationally expensive.
        "kenlm" -- KenLM, fast n-gram based model, works better for replacements in the middle
             and at the end of a phrase.

    Returns
    -------
    list[tuple[str, float]]
        List of replacements along with their scores, ordered by score from top to bottom.
    """

    candidates = []
    for replacement in match.replacements:
        candidates.append(match.context.replace(match.matchedText, replacement))

    ...


def rank_candidates_decoder(context: str, original: str, replacements: list[str]) -> list[tuple[str, float]]:
    """
    Uses decoder-based ranking of candidate replacements:
    * original + replacements are run through the GPT-like model
    * log-likelihood for tokens corresponding to the original and replacements are computed.
    * the best original and replacement are ordered by the log-likelihood.
    """

    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    subctx = context[:context.index(original)]  # context before the replacement
    repls = [original] + replacements  # combine all replacements, putting the original first.
    batch = [subctx + repl for repl in repls]

    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(batch, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**tokens)

    # shift for causal LM
    shift_logits = outputs.logits[:, :-1]
    shift_labels = tokens.input_ids[:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    subctx_offset = len(tokenizer(subctx.rstrip()).input_ids)
    shift_mask = tokens.attention_mask[:, 1:].clone()
    shift_mask[:, :subctx_offset-1] = 0
    likelihoods = token_log_probs.sum(-1) / shift_mask.sum(-1)  # mean log-likelihoods

    return list(zip(repls, likelihoods.tolist()))


def rank_candidates_minilm(candidates: list[str]) -> list[tuple[str, float]]:
    """
    Ranks candidates in terms of the embedding distance to the original sentence.
    """
    ...
    #
    # model_name = "nreimers/MiniLM-L6-H384-uncased"
    # pipe = pipeline("feature-extraction", model=model_name)
    # outs = pipe(candidates)
    # TODO


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

    # TODO: sort replacements based on the Levenshtein distance

    return replacements
