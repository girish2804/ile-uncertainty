"""Intermediate Layer Ensembling (ILE) Multi-Range Variant.

Extends Intermediate Layer Ensembling to cumulative candidate layer ranges (from [L-2, L-1]
down to [0, ..., L-1]), producing the multi-range List[List[Dict]] structure consumed by
compute_semantic_entropy_var.

Salient knowledge token identification is adapted from the attribution scoring formulation
of SDLG (Aichberger et al., ICLR 2025: "Improving Uncertainty Estimation through Semantically
Diverse Language Generation").
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

try:
    from .utils import clean_generation, compute_likelihood, generate_text
    from .disagree import (
        calculate_all_disagreement_metrics,
        compute_all_layers_disagreement_vectorized,
        get_relative_top_filter,
    )
    from .token_attribution import (
        compute_token_score_ranking,
        forward_hook,
        get_word_indices,
        rank_tensor,
    )
    from .layer_ensemble import get_model_layer_count
except ImportError:
    from utils import clean_generation, compute_likelihood, generate_text
    from disagree import (
        calculate_all_disagreement_metrics,
        compute_all_layers_disagreement_vectorized,
        get_relative_top_filter,
    )
    from token_attribution import (
        compute_token_score_ranking,
        forward_hook,
        get_word_indices,
        rank_tensor,
    )
    from layer_ensemble import get_model_layer_count


def generate_layer_ensemble_var_sequences(
    results_dict: Dict,
    deberta_model,
    deberta_tokenizer,
    device_deberta,
    deberta_embeddings: torch.Tensor,
    model,
    tokenizer,
    device_llm,
    input_ids: torch.Tensor,
    prompt: str,
    question: str,
    initial_generation: Dict,
    initial_likelihood: Dict,
    key: str,
    args,
) -> Dict:
    """Generate diverse sequences and calculate multi-range epistemic uncertainty.

    Intermediate hidden state logits are grouped into cumulative layer candidate ranges
    (from range [L-2, L-1] down to [0, ..., L-1]), producing a List[List[Dict]] structure
    consumed by compute_semantic_entropy_var.

    Args:
        results_dict: Storage container for experiment outputs.
        deberta_model: NLI classifier for attribution scoring.
        deberta_tokenizer: NLI tokenizer.
        device_deberta: Device placement for DeBERTa.
        deberta_embeddings: Precomputed DeBERTa embeddings table.
        model: Base generative language model.
        tokenizer: Generative model tokenizer.
        device_llm: Primary device placement for generative model.
        input_ids: Encoded prompt token IDs.
        prompt: Raw prompt text.
        question: Query substring.
        initial_generation: Greedy output dictionary.
        initial_likelihood: Likelihood and logit tensors.
        key: Sample index key in results_dict.
        args: Experiment configuration parameters.

    Returns:
        Updated results_dict containing generations, likelihoods, and multi-range epistem_entropies.
    """
    invalid_ids = getattr(args, 'invalid_ids', None) or []
    initial_generation_text = initial_generation['generation_text'][0]
    initial_generation_ids = initial_generation['generation_ids'][0]

    assert len(initial_likelihood["generation_logits"]) == 1
    generation_logits = initial_likelihood["generation_logits"][0].to(dtype=torch.float32)
    generation_logits = torch.nn.functional.softmax(generation_logits, dim=-1) + 1e-9

    assert generation_logits.shape[0] == initial_generation_ids.shape[0]

    single_word = False
    if initial_generation_ids.shape[0] == 0 or len(initial_generation_text.split()) == 0:
        return results_dict

    if len(initial_generation_text.split()) == 1:
        single_word = True
        token_info = {}
        if args.token_prob_threshold is None:
            other_voc_token_indices = torch.tensor(range(len(generation_logits[0])))
        else:
            other_voc_token_indices = torch.where(generation_logits[0] > args.token_prob_threshold)[0]

        for new_token_idx in other_voc_token_indices:
            if new_token_idx != initial_generation_ids[0] and new_token_idx not in invalid_ids:
                importance_score = generation_logits[0][new_token_idx]
                token_info[(0, 0, new_token_idx.item())] = (0, 0, importance_score.item())

        sorted_indices = torch.argsort(rank_tensor([v[2] for v in token_info.values()], descending=True), descending=False)

    additional_generated_text = []
    num_added_gens = 0

    if not single_word:
        sorted_indices, token_info = compute_token_score_ranking(
            deberta_model,
            deberta_tokenizer,
            device_deberta,
            question,
            initial_generation_ids,
            initial_generation_text,
            additional_generated_text,
            generation_logits,
            deberta_embeddings,
            tokenizer,
            args,
        )
        if not isinstance(sorted_indices, torch.Tensor):
            results_dict[key]['generations'] = []
            results_dict[key]['likelihoods'] = []
            return results_dict

    token_info_list = list(token_info.keys())
    if getattr(args, 'save_only_important_tokens', False):
        results_dict[key]['important_tokens'] = []
        num_added_tokens = 0
        for s in sorted_indices:
            initial_gen_word_idx, initial_gen_token_idx, new_token_idx = token_info_list[s]
            scores = token_info[(initial_gen_word_idx, initial_gen_token_idx, new_token_idx)]
            new_token_text = tokenizer.decode([new_token_idx])
            results_dict[key]['important_tokens'].append({
                'word_idx': initial_gen_word_idx,
                'token_idx': new_token_idx,
                'initial_gen_token_idx': initial_gen_token_idx,
                'token_text': new_token_text,
                'importance_score': scores[2],
                'substitution_score': scores[1],
                'attribution_score': scores[0],
            })
            num_added_tokens += 1
            if num_added_tokens >= args.num_total_generations:
                break
        return results_dict

    with torch.no_grad():
        for i, s in enumerate(sorted_indices):
            initial_gen_word_idx, initial_gen_token_idx, new_token_idx = token_info_list[s]

            if initial_gen_token_idx > 0:
                new_input_ids = initial_generation_ids[:initial_gen_token_idx]
                token_to_replace_id = initial_generation_ids[initial_gen_token_idx]
                all_input_ids = torch.hstack([input_ids.to(device_llm), new_input_ids.unsqueeze(0).to(device_llm)])
            else:
                initial_gen_token_idx = 0
                all_input_ids = input_ids.to(device_llm)
                token_to_replace_id = initial_generation_ids[0]

            new_token_text = tokenizer.decode([new_token_idx])
            importance_score = generation_logits[initial_gen_token_idx][new_token_idx].item()

            # Candidate ranges multi-depth epistemic uncertainty calculation
            lm_head = model.get_output_embeddings()
            model_outputs = model(all_input_ids, return_dict=True, output_hidden_states=True)

            final_logits = model_outputs.logits[:, -1, :].float()
            total_layers = get_model_layer_count(model)

            candidate_ranges = []
            for start_layer in range(total_layers - 2, -1, -1):
                candidate_ranges.append(list(range(start_layer, total_layers)))

            epistem_entropy = []
            for candidate_premature_layers in candidate_ranges:
                candidate_premature_logits = {}
                for candidate_premature_layer in candidate_premature_layers:
                    candidate_premature_logits[candidate_premature_layer] = lm_head(
                        model_outputs.hidden_states[candidate_premature_layer][:, -1, :]
                    ).to(final_logits.device)

                entropy_result = calculate_all_disagreement_metrics(
                    final_logits,
                    candidate_premature_logits,
                    calculate_metrics=['entropy_mixture'],
                    store_logits=False,
                )
                epistem_entropy.append(entropy_result)

            if new_token_idx in invalid_ids or new_token_idx == token_to_replace_id.item():
                continue

            eos_ref = getattr(args, 'eos_token_ids', None) or tokenizer.eos_token_id
            is_eos_token = (new_token_idx in eos_ref) if isinstance(eos_ref, (list, tuple, set)) else (new_token_idx == eos_ref)

            if not is_eos_token:
                final_input_ids = torch.hstack([all_input_ids, torch.tensor(new_token_idx).unsqueeze(0).unsqueeze(0).to(device_llm)])
                alternative_generation = generate_text(
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=final_input_ids,
                    len_prompt=len(prompt),
                    decoding_method="ensemble",
                    device=device_llm,
                )
            else:
                if initial_gen_word_idx == 0:
                    continue
                generation_to_add = torch.hstack([new_input_ids[0], torch.tensor(new_token_idx)])
                generation_text = tokenizer.decode(generation_to_add, skip_special_tokens=True).strip()
                cleaned_generation_text = clean_generation(generation_text)
                alternative_generation = {
                    'generation_ids': [generation_to_add],
                    'generation_text': [generation_text],
                    'cleaned_generation_ids': [
                        generation_to_add
                        if generation_text == cleaned_generation_text
                        else tokenizer.encode(cleaned_generation_text, add_special_tokens=False, return_tensors='pt')[0]
                    ],
                    'cleaned_generation_text': [cleaned_generation_text],
                    'logits': None,
                }

            if len(alternative_generation['generation_text'][0].strip()) == 0:
                continue

            alternative_likelihoods = compute_likelihood(
                prompt,
                alternative_generation,
                model,
                device_llm,
                compute_cleaned=False,
                store_logits=False,
            )

            alternative_generation['word_idx'] = initial_gen_word_idx
            alternative_generation['token_idx'] = new_token_idx
            alternative_generation['initial_gen_token_idx'] = initial_gen_token_idx
            alternative_generation['token_text'] = new_token_text
            alternative_generation['token_likelihood'] = importance_score
            alternative_generation['num_computed_gen'] = i + 1
            alternative_generation['initial_generation_ids'] = initial_generation_ids
            alternative_generation['initial_generation_text'] = initial_generation_text

            results_dict[key]['epistem_entropies'].append(epistem_entropy)
            results_dict[key]['generations'].append(alternative_generation)
            results_dict[key]['likelihoods'].append(alternative_likelihoods)
            num_added_gens += 1

            additional_generated_text.append(alternative_generation['generation_text'][0])

            if num_added_gens >= args.num_total_generations:
                return results_dict

    return results_dict


# Backward-compatibility alias
generate_semantically_diverse_output_sequences = generate_layer_ensemble_var_sequences
