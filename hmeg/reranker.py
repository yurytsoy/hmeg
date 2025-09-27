from __future__ import annotations

import kenlm
import sentencepiece as spm


class RerankerModels:
    kenlm_en = "kenlm/en"


class Reranker:
    model_name_: str = RerankerModels.kenlm_en
    models_ : dict[str, object] = dict()
    tokenizers_ : dict[str, object] = dict()

    def __init__(self, model_name: str | None = None) -> None:
        Reranker.set_current_model(model_name or RerankerModels.kenlm_en)

    @staticmethod
    def set_current_model(model_name: str):
        if model_name not in Reranker.models_:
            if model_name == RerankerModels.kenlm_en:
                Reranker.models_[RerankerModels.kenlm_en] = kenlm.LanguageModel("lm/en.arpa.bin")
                Reranker.tokenizers_[RerankerModels.kenlm_en] = spm.SentencePieceProcessor(model_file="lm/en.sp.model")
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
        if not Reranker.models_:
            Reranker.set_current_model(Reranker.model_name_)

        method = {
            RerankerModels.kenlm_en: Reranker.rank_kenlm_en,
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
        tokenizer: spm.SentencePieceProcessor = Reranker.tokenizers_[RerankerModels.kenlm_en]
        model: kenlm.LanguageModel = Reranker.models_[RerankerModels.kenlm_en]

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



def find_sublist_index(full: list[str], sublist: list[str]) -> int:
    """
    Takes on input two lists and returns index in the first list from which the second list can be found.
    If second list is not part/sublist of the first list, then return -1.
    """
    if not sublist or len(sublist) > len(full):
        return -1
    for k in range(len(full) - len(sublist) + 1):
        if full[k:k+len(sublist)] == sublist:
            return k
    return -1
