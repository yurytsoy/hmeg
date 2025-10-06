from __future__ import annotations

import kenlm
import os
import sentencepiece as spm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

from hmeg.usecases import find_sublist_index


class Reranker:
    class Models:
        kenlm_en = "kenlm/en"
        distillgpt2 = "distilbert/distilgpt2"

    model_name_: str = Models.kenlm_en
    models_ : dict[str, object] = dict()
    tokenizers_ : dict[str, object] = dict()

    def __init__(self, model_name: str | None = None) -> None:
        Reranker.set_current_model(model_name or Reranker.Models.kenlm_en)

    @staticmethod
    def set_current_model(model_name: str):
        if model_name not in Reranker.models_:

            if model_name == Reranker.Models.kenlm_en:
                if not os.path.exists("lm/en.arpa.bin"):
                    raise RuntimeError("The KenLM model is not found. Download the KenLM model and tokenizer first (eg from: https://huggingface.co/edugp/kenlm)")

                Reranker.models_[Reranker.Models.kenlm_en] = kenlm.LanguageModel("lm/en.arpa.bin")
                Reranker.tokenizers_[Reranker.Models.kenlm_en] = spm.SentencePieceProcessor(model_file="lm/en.sp.model")

            elif model_name == Reranker.Models.distillgpt2:
                Reranker.models_[Reranker.Models.distillgpt2] = AutoModelForCausalLM.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.pad_token = tokenizer.eos_token
                Reranker.tokenizers_[Reranker.Models.distillgpt2] = tokenizer

            else:
                raise NotImplementedError(f"Unknown model name {model_name}")

        Reranker.model_name_ = model_name

    @staticmethod
    def unload_model(model_name: str) -> bool:
        if model_name in Reranker.models_:
            del Reranker.models_[model_name]
            del Reranker.tokenizers_[model_name]
            return True
        return False

    @staticmethod
    def rank(context: str, original: str, replacements: list[str], full_sentence_score: bool = False) -> list[tuple[str, float]]:
        if Reranker.model_name_ not in Reranker.models_:  # load on demand
            Reranker.set_current_model(Reranker.model_name_)

        method = {
            Reranker.Models.kenlm_en: Reranker.rank_kenlm_en,
            Reranker.Models.distillgpt2: Reranker.rank_distillgpt2,
        }
        kwargs = dict(
            context=context,
            original=original,
            replacements=replacements,
            full_sentence_score=full_sentence_score
        )
        res = method[Reranker.model_name_](**kwargs)
        sorted_res = sorted(res, key=lambda k: k[1], reverse=True)
        return sorted_res

    @staticmethod
    def prepare_candidates(context: str, original: str, replacements: list[str], full_context: bool = False) -> list[str]:
        """
        Prepare candidate phrases for ranking.

        Note:
        * if `original` is encountered several times in the `context`, then only the **1st** occurrence
          will be replaced to generate candidates.

        Parameters
        ----------
        context: str
            Context for evaluation of ranking.
        original: str
            Original part of the context.
        replacements: list[str]
            Replacements for the `original`, that will be evaluated for ranking.
        full_context: bool, default=False
            Indicates whether candidates should use full context (True) or only part of the context before
            the `original` (False).
        """

        count = context.count(original)
        if count == 0:
            raise ValueError(f"Original '{original}' not in context '{context}'")

        if count > 1:
            warnings.warn(f"Original '{original}' is present in context multiple times. The first occurrence will be used.", UserWarning, stacklevel=2)

        subctx = None
        if not full_context:
            subctx = context.split(original)[0]
            res = [subctx + original]
        else:
            res = [context]

        for replacement in replacements:
            if full_context:
                res.append(context.replace(original, replacement, 1))
            else:
                res.append(subctx + replacement)
        return res

    @staticmethod
    def rank_kenlm_en(context: str, original: str, replacements: list[str], full_sentence_score: bool = False) -> list[tuple[str, float]]:
        tokenizer: spm.SentencePieceProcessor = Reranker.tokenizers_[Reranker.Models.kenlm_en]
        model: kenlm.LanguageModel = Reranker.models_[Reranker.Models.kenlm_en]

        res = []
        all_replacements = [original] + replacements
        candidates = Reranker.prepare_candidates(context, original, replacements, full_context=full_sentence_score)
        for idx, candidate in enumerate(candidates):
            tokens = tokenizer.encode(candidate, out_type=str)
            cur_score = model.score(" ".join(tokens), bos=True, eos=True)
            res.append((all_replacements[idx], cur_score))

        return res

    @staticmethod
    def rank_distillgpt2(context: str, original: str, replacements: list[str], full_sentence_score: bool = False) -> list[tuple[str, float]]:
        model = Reranker.models_[Reranker.Models.distillgpt2]
        tokenizer = Reranker.tokenizers_[Reranker.Models.distillgpt2]

        candidates = Reranker.prepare_candidates(context, original, replacements, full_context=full_sentence_score)
        tokens = tokenizer(candidates, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**tokens)

        # shift for causal LM
        shift_logits = outputs.logits[:, :-1]
        shift_labels = tokens.input_ids[:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        shift_mask = tokens.attention_mask[:, 1:].clone()
        likelihoods = (shift_mask * token_log_probs).sum(-1) / shift_mask.sum(-1)

        all_replacements = [original] + replacements
        return list(zip(all_replacements, likelihoods.tolist()))
