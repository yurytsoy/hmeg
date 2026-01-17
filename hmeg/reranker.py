from __future__ import annotations

import kenlm
from openai import OpenAI
import orjson
import os
import sentencepiece as spm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

from hmeg.prompt_loader import PromptLoader


class Reranker:
    """
    Class for GEC which can use various underlying models. The class behaves as a singleton object and is intended
    to have 1 "active" preloaded model, which is selected using `Reranker.set_current_model` method.

    Ranking can be performed using either the entire sentence or only correction span via `Reranker.rank` method.

    The list of supported models is given in the `Reranker.Models` subclass.
    """

    class Models:
        kenlm_en = "kenlm/en"
        distillgpt2 = "distilgpt2"
        openai = "openai"

    model_name_: str = Models.kenlm_en
    models_: dict[str, object] = dict()
    tokenizers_: dict[str, object] = dict()
    prompt_loader_: PromptLoader | None = None

    def __init__(self, model_name: str | None = None):
        Reranker.set_current_model(model_name or Reranker.Models.kenlm_en)

    @staticmethod
    def set_current_model(model_name: str):
        """
        Select model which will be used for reranking.

        If selected model can use CUDA device, then it will be automatically sent onto the device.

        Unloads models that were previously loaded / selected.

        Parameters
        ----------
        model_name : str
            Model name available in the `Reranker.Models`.
        """
        if model_name not in Reranker.models_:

            if model_name == Reranker.Models.kenlm_en:
                if not os.path.exists("lm/en.arpa.bin"):
                    raise RuntimeError("The KenLM model is not found. Download the KenLM model and tokenizer first (eg from: https://huggingface.co/edugp/kenlm)")

                Reranker.models_[Reranker.Models.kenlm_en] = kenlm.LanguageModel("lm/en.arpa.bin")
                Reranker.tokenizers_[Reranker.Models.kenlm_en] = spm.SentencePieceProcessor(model_file="lm/en.sp.model")

            elif model_name == Reranker.Models.distillgpt2:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = AutoModelForCausalLM.from_pretrained(model_name)
                model.to(device)
                Reranker.models_[model_name] = model

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.pad_token = tokenizer.eos_token
                Reranker.tokenizers_[model_name] = tokenizer

            elif model_name == Reranker.Models.openai:
                ...

            else:
                raise NotImplementedError(f"Unknown model name {model_name}")

        Reranker.model_name_ = model_name
        Reranker.unload_unused_models()

    @staticmethod
    def unload_unused_models():
        """
        Unloads all models and their tokenizers except for the model in `model_name_`.
        """
        cur_models = list(Reranker.models_)
        for model_name in cur_models:
            if model_name == Reranker.model_name_:
                continue
            Reranker.unload_model(model_name)

    @staticmethod
    def unload_model(model_name: str) -> bool:
        """
        Unload selected model and its tokenizer.

        Returns
        -------
        bool
            `True` if the model was unloaded, `False` otherwise (eg if model was not previously loaded).
        """
        if model_name in Reranker.models_:
            del Reranker.models_[model_name]
            del Reranker.tokenizers_[model_name]
            return True
        return False

    @staticmethod
    def rank(context: str, original: str, replacements: list[str], full_sentence_score: bool = False) -> list[tuple[str, float]]:
        """
        Rank `replacements` of the `original` text in the given `context` using log-likelihood score.

        Ranking is performed using model `Reranker.model_name_` (see `Reranker.set_current_model`).

        Parameters
        ----------
        context : str
            Context, inside which the original text is defined, and where the replacement happens.
        original : str
            Original part of the context that needs to be ranked agains the `replacements`.
        replacements : list[str]
            Suggested replacements for `original`.
        full_sentence_score : bool, default=False
            Indicates whether scoring should be performed on the full sentence (True) or only on the
            correction span (False), that includes context before the original and the replacement. The former is
            more accurate, but more computationally expensive.

        Return
        ------
        list[tuple[str, float]]
            List of replacements and their log-likelihood scores. The first item always corresponds to `original`.
            The scores for replacements are listed in the same order as items in the `replacements`.
        """
        if Reranker.model_name_ not in Reranker.models_:  # load on demand
            Reranker.set_current_model(Reranker.model_name_)

        method = {
            Reranker.Models.kenlm_en: Reranker.rank_kenlm_en,
            Reranker.Models.distillgpt2: Reranker.rank_distillgpt2,
            Reranker.Models.openai: Reranker.rank_openai,
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
            raise ValueError(f"Original text '{original}' not found in context '{context}'. Ensure the original text exists exactly as specified in the context.")

        if count > 1:
            warnings.warn(f"Original '{original}' is present in context multiple times. Only the first occurrence will be replaced when generating candidates.", UserWarning, stacklevel=2)

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
        device = next(model.parameters()).device

        candidates = Reranker.prepare_candidates(context, original, replacements, full_context=full_sentence_score)
        tokenizer = Reranker.tokenizers_[Reranker.Models.distillgpt2]
        tokens = tokenizer(candidates, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**tokens)

        # shift for causal LM
        shift_logits = outputs.logits[:, :-1]
        shift_labels = tokens.input_ids[:, 1:]
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        shift_mask = tokens.attention_mask[:, 1:]
        likelihoods = (shift_mask * token_log_probs).sum(-1) / shift_mask.sum(-1)

        all_replacements = [original] + replacements
        return list(zip(all_replacements, likelihoods.tolist()))

    @staticmethod
    def rank_openai(context: str, original: str, replacements: list[str], full_sentence_score: bool = False) -> list[tuple[str, float]]:
        """
        Rank candidate replacements using an OpenAI chat completion-based reranker.

        This method:
        - Loads a prompt template via `PromptLoader` (prompt id `v1/reranker/openai`),
        - Renders the user prompt with the provided `context`, `original`, `replacements` and
          `full_sentence_score` flag,
        - Calls the OpenAI Chat Completions API using the prompt's LLM configuration and system
          instructions, and
        - Parses an ordered JSON `{"results": [...]}` object from the model response and converts
          it into a list of `(replacement, score)` pairs where scores are negative integers
          (higher is better; first item corresponds to the original).

        Parameters
        ----------
        context : str
            Full sentence or surrounding text in which the `original` substring appears.
        original : str
            The original substring in `context` to be replaced/ranked.
        replacements : list[str]
            Candidate replacement strings to be ranked. The first returned item corresponds to
            `original`; subsequent items correspond to `replacements` in the same order.
        full_sentence_score : bool, optional
            If True, candidates are scored using the full `context`; otherwise a truncated
            context before `original` is used (default False).

        Returns
        -------
        list[tuple[str, float]]
            List of `(replacement, score)` tuples ordered by the model's ranking. Scores are
            negative integers produced as `-(rank_index + 1)` where a less-negative value
            indicates a better (higher) rank.

        Raises
        ------
        ValueError
            If the OpenAI response does not contain a valid JSON object with the expected
            `results` list, or if JSON parsing fails.
        Exception
            Any network/API errors raised by the OpenAI client or other unexpected failures
            (these are not swallowed by this function).

        Notes
        -----
        - This function performs network I/O and may incur API costs and latency.
        - The prompt id `v1/reranker/openai` and the prompt's `llm` configuration dictate
          the specific model and generation parameters used.
        - The function ensures that a `PromptLoader` is available (creates one if needed).
        """

        def parse_completion(output_text: str) -> list[str]:
            """
            Extract ordered replacements from OpenAI response.

            Parameters
            ----------
            output_text: str
                OpenAI response.

            Returns
            -------
            list[str]
                Decoded replacements.
            """

            json_start = output_text.find("{")
            json_end = output_text.rfind("}")

            if json_start == -1 or json_end == -1 or json_end < json_start:
                raise ValueError(f"Could not find valid JSON object in OpenAI response: {output_text!r}")

            json_str = output_text[json_start: json_end + 1]

            try:
                res = orjson.loads(json_str)
            except orjson.JSONDecodeError as e:
                raise ValueError(f"Failed to decode JSON from OpenAI response: {e}") from e

            if not isinstance(res, dict) or "results" not in res:
                raise ValueError(f"OpenAI response JSON does not contain expected 'results' field: {res!r}")

            results = res["results"]

            if not isinstance(results, list):
                raise ValueError(f"'results' field in OpenAI response JSON is not a list: {results!r}")

            return results

        if Reranker.prompt_loader_ is None:
            Reranker.prompt_loader_ = PromptLoader()

        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable "
                "before calling Reranker.rank_openai."
            )

        prompt_id = "v1/reranker/openai"
        prompt = Reranker.prompt_loader_.load(prompt_id)
        user_msg = prompt.render_user_prompt(
            context=context, original=original, replacements=replacements, full_sentence_score=full_sentence_score
        )

        client = OpenAI()
        response = client.chat.completions.create(
            model=prompt.llm.model,
            messages=[
                {"role": "system", "content": prompt.system_instructions},
                {"role": "user", "content": user_msg},
            ],
        )
        results = parse_completion(response.choices[0].message.content or "")
        return [(item, -(idx + 1)) for idx, item in enumerate(results)]
