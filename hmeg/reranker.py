from __future__ import annotations

import kenlm
import sentencepiece as spm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
                Reranker.models_[Reranker.Models.kenlm_en] = kenlm.LanguageModel("lm/en.arpa.bin")
                Reranker.tokenizers_[Reranker.Models.kenlm_en] = spm.SentencePieceProcessor(model_file="lm/en.sp.model")

            elif model_name == Reranker.Models.distillgpt2:
                Reranker.models_[Reranker.Models.distillgpt2] = AutoModelForCausalLM.from_pretrained(model_name)
                Reranker.tokenizers_[Reranker.Models.distillgpt2] = AutoTokenizer.from_pretrained(model_name)

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
    def rank(context: str, original: str, replacements: list[str]) -> list[tuple[str, float]]:
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
        )
        res = method[Reranker.model_name_](**kwargs)
        sorted_res = sorted(res, key=lambda k: k[1], reverse=True)
        return sorted_res

    @staticmethod
    def rank_kenlm_en(context: str, original: str, replacements: list[str]) -> list[tuple[str, float]]:
        tokenizer: spm.SentencePieceProcessor = Reranker.tokenizers_[Reranker.Models.kenlm_en]
        model: kenlm.LanguageModel = Reranker.models_[Reranker.Models.kenlm_en]

        tokens = tokenizer.encode(context, out_type=str)

        # find context before the replacements
        og_tokens = tokenizer.encode(original, out_type=str)
        subctx_offset = find_sublist_index(tokens, og_tokens)

        # Run KenLM ranker
        res = []
        original_score = model.score(" ".join(tokens[:subctx_offset] + og_tokens), bos=True, eos=True)
        res.append((original, original_score))

        for replacement in replacements:
            r_tokens = tokenizer.encode(replacement, out_type=str)
            cur_score = model.score(" ".join(tokens[:subctx_offset] + r_tokens), bos=True, eos=True)
            res.append((replacement, cur_score))

        return res

    @staticmethod
    def rank_distillgpt2(context: str, original: str, replacements: list[str]) -> list[tuple[str, float]]:
        model = Reranker.models_[Reranker.Models.distillgpt2]
        tokenizer = Reranker.tokenizers_[Reranker.Models.distillgpt2]

        # fixme: what if original appears multiple times in the context?
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

        # todo: double-check masking
        subctx_offset = len(tokenizer(subctx.rstrip()).input_ids)
        shift_mask = tokens.attention_mask[:, 1:].clone()
        shift_mask[:, :subctx_offset - 1] = 0
        likelihoods = token_log_probs.sum(-1) / shift_mask.sum(-1)  # mean log-likelihoods

        return list(zip(repls, likelihoods.tolist()))
