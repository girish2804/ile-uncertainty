"""Salient Knowledge Token Identification.

Adapted from the attribution scoring formulation in SDLG (Aichberger et al., ICLR 2025:
"Improving Uncertainty Estimation through Semantically Diverse Language Generation").

This module acts purely as an architectural filter to identify answer-bearing positions
in greedy output sequences, preventing syntax and stop words from diluting factual
uncertainty signals during subsequent layer-wise ensembling.
"""

import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# Dynamic path resolution to ensure root and src modules are importable
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..'))
for path in [script_dir, src_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Backward-compatibility stubs for external references
latest_grads = [None]
latest_embeddings = [None]


def forward_hook(module, input, output):
    """Hook stub retained for backward compatibility."""
    latest_embeddings[0] = output.detach()
    output.register_hook(lambda grad: latest_grads.__setitem__(0, grad))


def get_word_indices(text_ids: torch.Tensor, tokenizer) -> List[torch.Tensor]:
    """Segment token IDs into continuous word-level groupings.

    Args:
        text_ids: 1D tensor of token IDs representing the answer sequence.
        tokenizer: Hugging Face tokenizer instance.

    Returns:
        List of tensors, where each element contains token indices belonging to one word.
    """
    all_word_indices = []
    word_indices = [0]
    lookback = 1
    for t in range(1, len(text_ids)):
        if len(tokenizer.decode(text_ids[t]).strip()) == 0:
            word_indices.append(t)
            lookback += 1
        elif len(tokenizer.decode(text_ids[t - lookback : t + 1]).split()) == 1:
            word_indices.append(t)
            lookback = 1
        else:
            all_word_indices.append(torch.tensor(word_indices))
            word_indices = [t]
            lookback = 1
    all_word_indices.append(torch.tensor(word_indices))
    return all_word_indices


def rank_tensor(t: torch.Tensor, descending: bool = True) -> torch.Tensor:
    """Rank values in tensor with tie handling via unique consecutive values.

    Args:
        t: Input 1D tensor or list of scalar values.
        descending: Whether higher values receive rank 1.

    Returns:
        Tensor of 1-based integer ranks matching original input shape.
    """
    t = torch.tensor(t)
    sorted_vals, sorted_idx = torch.sort(t, descending=descending)
    unique_vals, inverse_indices = torch.unique_consecutive(sorted_vals, return_inverse=True)
    ranks = inverse_indices + 1
    corrected_ranks = torch.empty_like(sorted_idx)
    corrected_ranks[sorted_idx] = ranks
    return corrected_ranks


def compute_token_score_ranking(
    deberta_model,
    deberta_tokenizer,
    device_deberta,
    question: str,
    initial_generation_ids: torch.Tensor,
    initial_generation_text: str,
    additional_generated_text: List[str],
    generation_logits: torch.Tensor,
    deberta_embeddings: torch.Tensor,
    tokenizer,
    args,
) -> Tuple[torch.Tensor, Dict]:
    """Compute token substitution ranking using NLI attribution scores.

    Identifies high-salience answer-bearing positions via attribution scoring
    adapted from SDLG (Aichberger et al., ICLR 2025). Freezes DeBERTa parameters and
    evaluates input embedding gradients via direct autograd to identify tokens whose
    substitution isolates semantic divergence.

    Args:
        deberta_model: NLI classifier used to evaluate premise-hypothesis entailment.
        deberta_tokenizer: Tokenizer corresponding to the NLI classifier.
        device_deberta: Device placement for DeBERTa operations.
        question: Prompt query string.
        initial_generation_ids: Greedy decoded token IDs.
        initial_generation_text: Greedy decoded response string.
        additional_generated_text: Previously generated hypothesis strings.
        generation_logits: Next-token logits from base language model.
        deberta_embeddings: Pre-extracted word embedding table of DeBERTa.
        tokenizer: Tokenizer of the primary generative language model.
        args: Experiment configuration parameters.

    Returns:
        sorted_indices: Permutation tensor ordering candidate replacements by combined score.
        token_info: Dictionary mapping candidate (word_idx, token_idx, vocab_idx) tuples to
            (attribution_score, substitution_score, importance_score).
    """
    invalid_ids = getattr(args, 'invalid_ids', None) or []

    # Freeze DeBERTa parameters to eliminate full backward pass over model weights
    for p in deberta_model.parameters():
        p.requires_grad = False

    ce_loss_fn = torch.nn.CrossEntropyLoss()
    encoded_question = deberta_tokenizer.encode(question, padding=True, return_tensors='pt').squeeze()[:-1]
    encoded_answer = deberta_tokenizer.encode(' ' + initial_generation_text, padding=True, return_tensors='pt').squeeze()[1:-1]
    all_word_indices_deberta = get_word_indices(text_ids=encoded_answer, tokenizer=deberta_tokenizer)

    # Filter special tokens to match generative LLM tokens
    special_ids = set(tokenizer.all_special_ids)
    clean_llm_ids = torch.tensor([tid for tid in initial_generation_ids.tolist() if tid not in special_ids], device=initial_generation_ids.device)
    all_word_indices_llm = get_word_indices(text_ids=clean_llm_ids, tokenizer=tokenizer)

    words = initial_generation_text.split()
    if len(all_word_indices_deberta) != len(all_word_indices_llm) or len(all_word_indices_deberta) != len(words):
        print(f"Segmentation mismatch: words={len(words)}, deberta={len(all_word_indices_deberta)}, llm={len(all_word_indices_llm)}")
        return False, {}

    qa_initial = question + ' ' + initial_generation_text
    input_sequence = qa_initial + ' [SEP] ' + qa_initial
    model_input = [input_sequence]

    for additional_a in additional_generated_text:
        input_sequence = qa_initial + ' [SEP] ' + question + ' ' + additional_a
        model_input.append(input_sequence)

    encoded_input = deberta_tokenizer(model_input, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device_deberta)

    # Direct input embedding autograd
    embedding_layer = deberta_model.get_input_embeddings() if hasattr(deberta_model, 'get_input_embeddings') else deberta_model.deberta.embeddings.word_embeddings
    inputs_embeds = embedding_layer(encoded_input["input_ids"]).detach()
    inputs_embeds.requires_grad_(True)

    input_kwargs = {k: v for k, v in encoded_input.items() if k != "input_ids"}
    prediction_out = deberta_model(inputs_embeds=inputs_embeds, **input_kwargs)
    prediction = prediction_out['logits'] if (isinstance(prediction_out, dict) or hasattr(prediction_out, '__getitem__')) else prediction_out.logits

    target = torch.tensor([0] + [0] * len(additional_generated_text), device=device_deberta)
    loss = ce_loss_fn(prediction, target)
    embeds_grads = torch.autograd.grad(loss, inputs_embeds)[0]

    assert encoded_input["input_ids"].shape[1] == embeds_grads.shape[1] == inputs_embeds.shape[1]

    for i in range(len(additional_generated_text) + 1):
        if encoded_question.tolist() != encoded_input["input_ids"][i, :len(encoded_question)].tolist():
            return False, {}
    for word, word_token_indices in zip(words, all_word_indices_deberta):
        if word.strip() != deberta_tokenizer.decode(encoded_answer[word_token_indices]).strip():
            return False, {}

    token_info = {}
    with torch.no_grad():
        qa1_gradients = embeds_grads[:, len(encoded_question) : len(encoded_question) + len(encoded_answer), :]
        qa1_embeddings = inputs_embeds[:, len(encoded_question) : len(encoded_question) + len(encoded_answer), :]
        qa1_attributions = qa1_gradients * qa1_embeddings
        token_attributions = torch.abs(qa1_attributions)

        # 1. Word-level attribution magnitude
        all_word_gradient_magnitudes = []
        for i in range(len(additional_generated_text) + 1):
            word_attributions = torch.vstack([token_attributions[i, word_token_indices, :].mean(dim=0) for word_token_indices in all_word_indices_deberta])
            all_word_gradient_magnitudes.append(torch.norm(word_attributions, dim=-1).tolist())

        word_gradient_magnitude = torch.tensor(all_word_gradient_magnitudes).mean(dim=0)
        deberta_gradients = qa1_gradients

        # 2 & 3. Substitution and importance scores
        for initial_gen_word_idx, word_token_indices_deberta in enumerate(all_word_indices_deberta):
            deberta_token_idx = word_token_indices_deberta[0]
            initial_voc_token_idx_deberta = encoded_answer[deberta_token_idx]

            word_token_indices_llm = word_token_indices_deberta if initial_gen_word_idx >= len(all_word_indices_llm) else all_word_indices_llm[initial_gen_word_idx]
            initial_gen_token_idx_llm = word_token_indices_llm[0]
            initial_voc_token_idx_llm = initial_generation_ids[initial_gen_token_idx_llm]

            if args.token_prob_threshold is None:
                other_voc_token_indices_llm = torch.tensor(range(len(generation_logits[initial_gen_token_idx_llm])))
            else:
                other_voc_token_indices_llm = torch.where(generation_logits[initial_gen_token_idx_llm] > args.token_prob_threshold)[0]

            candidate_embs = []
            valid_indices_llm = []
            for token_idx_llm in other_voc_token_indices_llm:
                token_idx_llm_val = token_idx_llm.item()
                new_token_text = tokenizer.decode([token_idx_llm_val])
                if len(new_token_text.strip()) == 0:
                    continue
                encoded_new_token = deberta_tokenizer.encode(new_token_text, add_special_tokens=False)
                if len(encoded_new_token) > 0:
                    emb = deberta_embeddings[encoded_new_token].mean(dim=0)
                    candidate_embs.append(emb)
                    valid_indices_llm.append(token_idx_llm_val)

            if len(candidate_embs) == 0:
                continue

            candidate_embs = torch.stack(candidate_embs).to(device_deberta)
            delta_embeddings = deberta_embeddings[initial_voc_token_idx_deberta].unsqueeze(0) - candidate_embs

            all_substitution_scores = []
            for i in range(len(additional_generated_text) + 1):
                all_substitution_scores.append(torch.nn.functional.cosine_similarity(delta_embeddings, deberta_gradients[i, deberta_token_idx].unsqueeze(0)).tolist())

            all_substitution_scores = torch.tensor(all_substitution_scores).mean(dim=0)

            for new_token_idx_llm, substitution_score in zip(valid_indices_llm, all_substitution_scores):
                attribution_score = word_gradient_magnitude[initial_gen_word_idx]
                importance_score = generation_logits[initial_gen_token_idx_llm][new_token_idx_llm]

                if new_token_idx_llm != initial_voc_token_idx_llm and new_token_idx_llm not in invalid_ids:
                    token_info[(initial_gen_word_idx, initial_gen_token_idx_llm.item(), new_token_idx_llm)] = (
                        attribution_score.item(),
                        substitution_score.item(),
                        importance_score.item(),
                    )

    ranking_attribution_score = rank_tensor([v[0] for v in token_info.values()], descending=True)
    ranking_substitution_score = rank_tensor([v[1] for v in token_info.values()], descending=True)
    ranking_importance_score = rank_tensor([v[2] for v in token_info.values()], descending=True)

    sorted_indices = torch.argsort(
        args.alphas[0] * ranking_attribution_score
        + args.alphas[1] * ranking_substitution_score
        + args.alphas[2] * ranking_importance_score,
        descending=False,
    )

    return sorted_indices, token_info
