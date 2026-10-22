import os
import sys
import random
import pickle
import gzip
import json
import numpy as np
import torch
import math
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
try:
    from seq_lvl_imp import sentence_importance
except Exception:
    sentence_importance = None

# nli_model_sent = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')

# Compatibility Shim for Phi-3 / DeepSeek MoE with modern transformers DynamicCache ('seen_tokens', 'get_max_length' errors)
try:
    from transformers.cache_utils import DynamicCache, Cache
    if not hasattr(DynamicCache, "seen_tokens"):
        def _get_seen_tokens(self, layer_idx=0):
            try:
                return self.get_seq_length(layer_idx)
            except Exception:
                try:
                    return self.get_seq_length()
                except Exception:
                    return len(self.key_cache[0][0]) if (hasattr(self, 'key_cache') and self.key_cache) else 0
        DynamicCache.seen_tokens = property(_get_seen_tokens)
        DynamicCache._seen_tokens = property(_get_seen_tokens)

    _orig_cache_getitem = getattr(DynamicCache, "__getitem__", None)
    def _safe_cache_getitem(self, item):
        if item in ("seen_tokens", "_seen_tokens"):
            try:
                return self.get_seq_length()
            except Exception:
                return 0
        if _orig_cache_getitem is not None:
            return _orig_cache_getitem(self, item)
        if isinstance(item, int):
            return (self.key_cache[item], self.value_cache[item])
        raise KeyError(item)
    DynamicCache.__getitem__ = _safe_cache_getitem

    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, seq_len=None, layer_idx=0):
            try:
                return self.get_seq_length(layer_idx)
            except Exception:
                return seq_len if seq_len is not None else 0
        DynamicCache.get_usable_length = _get_usable_length

    if not hasattr(DynamicCache, "get_max_length"):
        def _get_max_length(self):
            return getattr(self, "max_cache_length", None)
        DynamicCache.get_max_length = _get_max_length

    if not hasattr(DynamicCache, "get_max_cache_shape"):
        def _get_max_cache_shape(self):
            return getattr(self, "max_cache_length", None)
        DynamicCache.get_max_cache_shape = _get_max_cache_shape

    for attr, fn in [
        ("get_max_length", lambda self: getattr(self, "max_cache_length", None)),
        ("get_max_cache_shape", lambda self: getattr(self, "max_cache_length", None)),
        ("get_usable_length", lambda self, seq_len=None, layer_idx=0: self.get_seq_length(layer_idx) if hasattr(self, "get_seq_length") else seq_len)
    ]:
        if 'Cache' in locals() and not hasattr(Cache, attr):
            setattr(Cache, attr, fn)
except Exception:
    pass

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_models_and_tokenizers(model_type_llm=None, 
                              device_llm=None, 
                              model_type_deberta=None, 
                              device_deberta=None, 
                              get_tokenizer_only_llm=False, 
                              get_tokenizer_only_deberta=False, 
                              use_flash_attention=True):

    if model_type_llm is not None:

        if not get_tokenizer_only_llm:
            assert device_llm is not None , "device_llm must be specified"
        if not get_tokenizer_only_deberta:
            assert device_deberta is not None, "device_deberta must be specified"

        if model_type_llm in ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b', 'opt-66b']:
            opt_path = os.path.join("facebook", model_type_llm)
            tokenizer = AutoTokenizer.from_pretrained(opt_path)
            if use_flash_attention:
                model = AutoModelForCausalLM.from_pretrained(opt_path,
                                                    torch_dtype=torch.bfloat16, 
                                                    attn_implementation="flash_attention_2",
                                                    use_cache=True) if not get_tokenizer_only_llm else None
            else:
                model = AutoModelForCausalLM.from_pretrained(opt_path, torch_dtype=torch.bfloat16) if not get_tokenizer_only_llm else None
        else:
            # General loader supporting DeepSeek MoE, Qwen MoE, Llama, Gemma, and other Hugging Face models
            is_phi = "phi-3" in str(model_type_llm).lower() or "phi3" in str(model_type_llm).lower()
            trust_code = False if is_phi else True
            tokenizer = AutoTokenizer.from_pretrained(model_type_llm, trust_remote_code=trust_code)
            if not get_tokenizer_only_llm:
                model = AutoModelForCausalLM.from_pretrained(
                    model_type_llm,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=trust_code,
                    device_map="auto" if device_llm is None else None
                )
                if device_llm is not None and not hasattr(model, 'hf_device_map'):
                    model = model.to(device_llm)
        
        if not get_tokenizer_only_llm:
            if model_type_llm in ['opt-30b']:
                device_map = {
                    'model.decoder.embed_tokens': 0,
                    'model.decoder.embed_positions': 0,
                    'model.decoder.layers.0': 0,
                    'model.decoder.layers.1': 0,
                    'model.decoder.layers.2': 0,
                    'model.decoder.layers.3': 0,
                    'model.decoder.layers.4': 0,
                    'model.decoder.layers.5': 0,
                    'model.decoder.layers.6': 0,
                    'model.decoder.layers.7': 0,
                    'model.decoder.layers.8': 0,
                    'model.decoder.layers.9': 0,
                    'model.decoder.layers.10': 0,
                    'model.decoder.layers.11': 0,
                    'model.decoder.layers.12': 0,
                    'model.decoder.layers.13': 0,
                    'model.decoder.layers.14': 0,
                    'model.decoder.layers.15': 0,
                    'model.decoder.layers.16': 1,
                    'model.decoder.layers.17': 1,
                    'model.decoder.layers.18': 1,
                    'model.decoder.layers.19': 1,
                    'model.decoder.layers.20': 1,
                    'model.decoder.layers.21': 1,
                    'model.decoder.layers.22': 1,
                    'model.decoder.layers.23': 1,
                    'model.decoder.layers.24': 1,
                    'model.decoder.layers.25': 1,
                    'model.decoder.layers.26': 1,
                    'model.decoder.layers.27': 1,
                    'model.decoder.layers.28': 1,
                    'model.decoder.layers.29': 1,
                    'model.decoder.layers.30': 1,
                    'model.decoder.layers.31': 1,
                    'model.decoder.layers.32': 1,
                    'model.decoder.layers.33': 1,
                    'model.decoder.layers.34': 1,
                    'model.decoder.layers.35': 1,
                    'model.decoder.layers.36': 1,
                    'model.decoder.layers.37': 1,
                    'model.decoder.layers.38': 1,
                    'model.decoder.layers.39': 1,
                    'model.decoder.layers.40': 1,
                    'model.decoder.layers.41': 1,
                    'model.decoder.layers.42': 1,
                    'model.decoder.layers.43': 1,
                    'model.decoder.layers.44': 1,
                    'model.decoder.layers.45': 1,
                    'model.decoder.layers.46': 0,
                    'model.decoder.layers.47': 0,
                    'model.decoder.layers.48': 0,
                    'model.decoder.final_layer_norm': 0,
                    'lm_head': 0
                }
                dispatch_model(model, device_map=device_map)
            elif model_type_llm in ['opt-66b']:
                device_map = {
                    'model.decoder.embed_tokens': 0,
                    'model.decoder.embed_positions': 0,
                    'model.decoder.layers.0': 0,
                    'model.decoder.layers.1': 0,
                    'model.decoder.layers.2': 0,
                    'model.decoder.layers.3': 0,
                    'model.decoder.layers.4': 0,
                    'model.decoder.layers.5': 0,
                    'model.decoder.layers.6': 0,
                    'model.decoder.layers.7': 0,
                    'model.decoder.layers.8': 0,
                    'model.decoder.layers.9': 0,
                    'model.decoder.layers.10': 1,
                    'model.decoder.layers.11': 1,
                    'model.decoder.layers.12': 1,
                    'model.decoder.layers.13': 1,
                    'model.decoder.layers.14': 1,
                    'model.decoder.layers.15': 1,
                    'model.decoder.layers.16': 1,
                    'model.decoder.layers.17': 1,
                    'model.decoder.layers.18': 1,
                    'model.decoder.layers.19': 1,
                    'model.decoder.layers.20': 1,
                    'model.decoder.layers.21': 1,
                    'model.decoder.layers.22': 1,
                    'model.decoder.layers.23': 1,
                    'model.decoder.layers.24': 1,
                    'model.decoder.layers.25': 1,
                    'model.decoder.layers.26': 1,
                    'model.decoder.layers.27': 2,
                    'model.decoder.layers.28': 2,
                    'model.decoder.layers.29': 2,
                    'model.decoder.layers.30': 2,
                    'model.decoder.layers.31': 2,
                    'model.decoder.layers.32': 2,
                    'model.decoder.layers.33': 2,
                    'model.decoder.layers.34': 2,
                    'model.decoder.layers.35': 2,
                    'model.decoder.layers.36': 2,
                    'model.decoder.layers.37': 2,
                    'model.decoder.layers.38': 2,
                    'model.decoder.layers.39': 2,
                    'model.decoder.layers.40': 2,
                    'model.decoder.layers.41': 2,
                    'model.decoder.layers.42': 2,
                    'model.decoder.layers.43': 2,
                    'model.decoder.layers.44': 3,
                    'model.decoder.layers.45': 3,
                    'model.decoder.layers.46': 3,
                    'model.decoder.layers.47': 3,
                    'model.decoder.layers.48': 3,
                    'model.decoder.layers.49': 3,
                    'model.decoder.layers.50': 3,
                    'model.decoder.layers.51': 3,
                    'model.decoder.layers.52': 3,
                    'model.decoder.layers.53': 3,
                    'model.decoder.layers.54': 3,
                    'model.decoder.layers.55': 3,
                    'model.decoder.layers.56': 3,
                    'model.decoder.layers.57': 3,
                    'model.decoder.layers.58': 3,
                    'model.decoder.layers.59': 3,
                    'model.decoder.layers.60': 3,
                    'model.decoder.layers.61': 0,
                    'model.decoder.layers.62': 0,
                    'model.decoder.layers.63': 0,
                    'model.decoder.layers.64': 0,
                    'model.decoder.final_layer_norm': 0,
                    'lm_head': 0
                }
                dispatch_model(model, device_map=device_map)
            else:
                model = model.to(device_llm)
    else:
        model, tokenizer = None, None

    if model_type_deberta is not None:

        if not get_tokenizer_only_deberta:
            assert device_deberta is not None and not get_tokenizer_only_deberta, "device_deberta must be specified"

        if model_type_deberta in ["deberta-base-mnli", "deberta-large-mnli", "deberta-xlarge-mnli", "deberta-v2-xlarge-mnli", "deberta-v2-xxlarge-mnli"]:
            deberta_tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{model_type_deberta}")
            deberta_model = AutoModelForSequenceClassification.from_pretrained(f"microsoft/{model_type_deberta}").to(device_deberta) if not get_tokenizer_only_deberta else None

    else:
        deberta_model, deberta_tokenizer = None, None

    return model, tokenizer, deberta_model, deberta_tokenizer


@torch.no_grad()
def remove_invalid_ids(generation, 
                       invalid_ids):
    for invalid in invalid_ids:
        if invalid in generation:
            generation = generation[:torch.where(generation == invalid)[0][0]]
    return generation


@torch.no_grad()
def clean_generation(generation):
    if not isinstance(generation, str) or not generation.strip():
        return ""

    generation = generation.strip()
    
    # Strip leading conversational / template prefixes without truncating
    prefixes = ['Answer:', 'answer:', 'ANSWER:', 'Answers:', 'answers:', 'A:', 'A;', 'Question:', 'question:', 'QUESTION:', 'Q:', 'Q;']
    for p in prefixes:
        if generation.startswith(p):
            generation = generation[len(p):].strip()

    # Truncate only if a second question/turn marker appears later in text
    first_line = generation.split('\n')[0].strip()
    for stop_word in prefixes:
        stop_idx = first_line.find(stop_word)
        if stop_idx > 0:  # strictly after position 0
            first_line = first_line[:stop_idx].strip()

    return first_line if first_line else generation.strip()


@torch.no_grad()
def generate_text(args, 
                  model, 
                  tokenizer, 
                  input_ids, 
                  len_prompt, 
                  decoding_method, 
                  device):
    
    # Standardize input to device
    input_ids = input_ids.to(device).reshape(1, -1) if args.dataset == 'trivia_qa' else input_ids.to(device)
    
    attention_mask = torch.ones_like(input_ids)

    # Dynamically resolve model-specific EOS token ID
    effective_eos = getattr(args, 'eos_token_ids', None)
    if effective_eos is None or effective_eos == 4:
        effective_eos = getattr(model.config, 'eos_token_id', tokenizer.eos_token_id)
        if effective_eos is None:
            effective_eos = tokenizer.eos_token_id

    # Check if Gemma model is active
    model_type_str = str(getattr(model.config, 'model_type', '')).lower()
    is_gemma_model = "gemma" in model_type_str or "gemma" in getattr(args, 'model_type_llm', '').lower()

    if is_gemma_model:
        # Isolated Gemma-specific generation parameters
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        gen_kwargs = {
            "attention_mask": attention_mask,
            "pad_token_id": pad_id,
            "eos_token_id": effective_eos,
            "max_new_tokens": args.max_length_of_generated_sequence,
        }
    else:
        # Standard generation parameters conserved for all models
        gen_kwargs = {
            "attention_mask": attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": effective_eos,
            "max_new_tokens": args.max_length_of_generated_sequence,
        }

    if getattr(args, 'invalid_ids', None) and len(args.invalid_ids) > 0:
        vocab_sz = getattr(model.config, 'vocab_size', None)
        if vocab_sz is None and hasattr(model, 'get_input_embeddings'):
            emb = model.get_input_embeddings()
            if emb is not None:
                vocab_sz = emb.weight.shape[0]
        if vocab_sz is None and tokenizer is not None:
            vocab_sz = len(tokenizer)

        valid_invalid_ids = [
            tid for tid in args.invalid_ids
            if isinstance(tid, int) and (vocab_sz is None or (0 <= tid < vocab_sz))
        ]
        if valid_invalid_ids:
            gen_kwargs["bad_words_ids"] = [[token_id] for token_id in valid_invalid_ids]
    
    if decoding_method == 'dola':
        generation_ids = model.generate(input_ids,
                                        num_return_sequences=args.num_return_sequences_most_likely,
                                        do_sample=False,
                                        dola_layers='high',
                                        repetition_penalty=1.2,
                                        temperature=args.temperature_most_likely,
                                        top_p=args.top_p_most_likely,
                                        trust_remote_code=True,
                                        **gen_kwargs)

    elif decoding_method == "most_likely":
        generation_ids = model.generate(input_ids,
                                        num_beams=args.num_beams_most_likely,
                                        num_return_sequences=args.num_return_sequences_most_likely,
                                        do_sample=args.do_sample_most_likely,
                                        temperature=args.temperature_most_likely,
                                        top_p=args.top_p_most_likely,
                                        **gen_kwargs)

    elif decoding_method == 'baseline':
        generation_ids = model.generate(input_ids,
                                        num_beams=args.num_beams_baseline,
                                        num_beam_groups=args.num_beam_groups_baseline,
                                        diversity_penalty=args.diversity_penalty_baseline,
                                        num_return_sequences=args.num_return_sequences_baseline,
                                        do_sample=args.do_sample_baseline,
                                        temperature=args.temperature_baseline,
                                        top_p=args.top_p_baseline,
                                        **gen_kwargs)

    elif decoding_method in ('sdlg', 'ensemble', 'layer_ensemble'):
        beams = getattr(args, 'num_beams_ensemble', getattr(args, 'num_beams_sdlg', 1))
        return_seqs = getattr(args, 'num_return_sequences_ensemble', getattr(args, 'num_return_sequences_sdlg', 1))
        do_sample = getattr(args, 'do_sample_ensemble', getattr(args, 'do_sample_sdlg', False))
        temp = getattr(args, 'temperature_ensemble', getattr(args, 'temperature_sdlg', 1.0))
        top_p = getattr(args, 'top_p_ensemble', getattr(args, 'top_p_sdlg', 1.0))
        generation_ids = model.generate(input_ids,
                                        num_beams=beams * return_seqs,
                                        num_return_sequences=return_seqs,
                                        do_sample=do_sample,
                                        temperature=temp,
                                        top_p=top_p,
                                        **gen_kwargs)

    generation_ids = generation_ids.detach().to('cpu')

    generation_ids_list, generation_text_list, cleaned_generation_ids_list, cleaned_generation_text_list = list(), list(), list(), list()
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    for i in range(len(generation_ids)):
        
        generation_to_add = generation_ids[i][len_prompt:]
        if pad_token_id is not None:
            generation_to_add = generation_to_add[generation_to_add != pad_token_id] # remove pad_token_ids

        # generation_to_add = remove_invalid_ids(generation_to_add, args.invalid_ids)
        generation_ids_list.append(generation_to_add)
        generation_text = tokenizer.decode(generation_to_add, skip_special_tokens=True).strip()

        generation_text_list.append(generation_text)

        cleaned_generation_text = clean_generation(generation_text)
        cleaned_generation_text_list.append(cleaned_generation_text)
        cleaned_generation_ids_list.append(generation_to_add if cleaned_generation_text == generation_text else \
            tokenizer.encode(cleaned_generation_text, add_special_tokens=False, return_tensors='pt')[0])

    return {
        'generation_ids': generation_ids_list,
        'generation_text': generation_text_list,

        'cleaned_generation_ids': cleaned_generation_ids_list,
        'cleaned_generation_text': cleaned_generation_text_list,
    }



@torch.no_grad()
def prepare_generated_text(generation_ids, 
                           generation_text, 
                           cleaned_generation_ids, 
                           cleaned_generation_text, 
                           **kwargs):
    list_generation_dicts = []


    for i in range(len(generation_ids)):
        generation_dict = {
            'generation_ids': [generation_ids[i]],
            'generation_text': [generation_text[i]],
            'cleaned_generation_ids': [cleaned_generation_ids[i]],
            'cleaned_generation_text': [cleaned_generation_text[i]],
        }
            
        # add other kwargs (word_idx, new_token, token_likelihood)
        for kwarg in kwargs.keys():
            generation_dict[kwarg] = kwargs[kwarg]
        
        list_generation_dicts.append(generation_dict)

    return list_generation_dicts


@torch.no_grad()
def compute_correctness(args, 
                        reference_answers, 
                        incorrect_answers, 
                        most_likely_generation_text, 
                        exact_match_metric=None, 
                        rouge=None, 
                        bleurt=None):
    correctness_dict = {}

    if exact_match_metric is not None:
        exact_match = 0.0
        for answer in reference_answers:
            results = exact_match_metric.compute(predictions=[most_likely_generation_text],
                                                references=[answer],
                                                ignore_case=True,
                                                ignore_punctuation=True)
            exact_match = max(results['exact_match'], exact_match)
        
        correctness_dict['exact_match'] = exact_match
            
    if rouge is not None:
        rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
        for answer in reference_answers:
            rouge_results = rouge.compute(predictions=[most_likely_generation_text], 
                                        references=[answer])
            rouge1 = max(rouge_results['rouge1'].item(), rouge1)
            rouge2 = max(rouge_results['rouge2'].item(), rouge2)
            rougeL = max(rouge_results['rougeL'].item(), rougeL)

        incorrect_rouge1, incorrect_rouge2, incorrect_rougeL = 0.0, 0.0, 0.0
        for incorrect_answer in incorrect_answers:
            rouge_results = rouge.compute(predictions=[most_likely_generation_text], 
                                        references=[incorrect_answer])
            incorrect_rouge1 = max(rouge_results['rouge1'].item(), incorrect_rouge1)
            incorrect_rouge2 = max(rouge_results['rouge2'].item(), incorrect_rouge2)
            incorrect_rougeL = max(rouge_results['rougeL'].item(), incorrect_rougeL)

        if len(incorrect_answers) != 0:
            correctness_dict['rouge1-diff'] = rouge1 - incorrect_rouge1
            correctness_dict['rouge2-diff'] = rouge2 - incorrect_rouge2
            correctness_dict['rougeL-diff'] = rougeL - incorrect_rougeL
        correctness_dict['rouge1'] = rouge1
        correctness_dict['rouge2'] = rouge2
        correctness_dict['rougeL'] = rougeL

    if bleurt is not None:
        scores_true = max(bleurt.compute(predictions=[most_likely_generation_text] * len(reference_answers), references=reference_answers)['scores'])

        correctness_dict['bleurt'] = scores_true
        if len(incorrect_answers) != 0:
            scores_false = max(bleurt.compute(predictions=[most_likely_generation_text] * len(incorrect_answers), 
                                              references=incorrect_answers)['scores'])
            correctness_dict['bleurt-diff'] = scores_true - scores_false

    return correctness_dict


@torch.no_grad()
def compute_likelihood(prompt, 
                       generation, 
                       model, 
                       device, 
                       compute_cleaned=False, 
                       store_logits=True):
        
    # Note: This computation of NLL follows the impementation of Kuhn et al. (2023)
    list_average_neg_log_likelihoods, list_neg_log_likelihood = [], []
    list_cleaned_average_neg_log_likelihood, list_cleaned_neg_log_likelihood = [], []
    list_generation_logits, list_cleaned_generation_logits = [], []

    # iterate over all generations -> "generation_ids" is list of generations
    for i in range(len(generation['generation_ids'])): 

        generation_ids = generation['generation_ids'][i]

        generation_input = torch.hstack([prompt, generation_ids]).to(device)

        target_ids = generation_input.clone()
        target_ids[:len(prompt)] = -100
        model_output = model(torch.reshape(generation_input, (1, -1)), labels=target_ids)
        average_neg_log_likelihood = model_output['loss'].item()
        neg_log_likelihood = average_neg_log_likelihood * (len(generation_ids))

        list_average_neg_log_likelihoods.append(average_neg_log_likelihood)
        list_neg_log_likelihood.append(neg_log_likelihood)

        # compute logits
        if store_logits:
            generation_logits = model_output["logits"][0, len(prompt)-1:-1, :].detach().half().to('cpu') 
            # shift by 1 since token probs at last token of prompt already belong to first token of generation
            list_generation_logits.append(generation_logits)
            assert generation_logits.shape[0] == generation_ids.shape[0]

        if compute_cleaned:

            cleaned_generation_ids = generation['cleaned_generation_ids'][i]

            if torch.equal(cleaned_generation_ids, generation_ids) or \
                generation['cleaned_generation_text'][i] == generation['generation_text'][i]:
                cleaned_average_neg_log_likelihood = average_neg_log_likelihood
                cleaned_neg_log_likelihood = neg_log_likelihood
                if store_logits:
                    cleaned_generation_logits = generation_logits
            elif generation['cleaned_generation_text'][i] == '':
                # Note: setting nll to ngative infinity (zero likelihood) if cleaned generation is empty
                cleaned_average_neg_log_likelihood = float('-inf')
                cleaned_neg_log_likelihood = float('-inf')
                if store_logits:
                    cleaned_generation_logits = []
            else:
                # Note: computation of NNL follows tutorial: https://huggingface.co/docs/transformers/perplexity
                generation_input = torch.hstack([prompt, cleaned_generation_ids]).to(device)
                target_ids = generation_input.clone()
                target_ids[:len(prompt)] = -100
                model_output = model(torch.reshape(generation_input, (1, -1)), labels=target_ids)
                cleaned_average_neg_log_likelihood = model_output['loss'].item()
                cleaned_neg_log_likelihood = cleaned_average_neg_log_likelihood * (len(cleaned_generation_ids))

                # compute logits
                if store_logits:
                    cleaned_generation_logits = model_output["logits"][0, len(prompt)-1:-1, :].detach().half().to('cpu')
            
            if store_logits:
                list_cleaned_generation_logits.append(cleaned_generation_logits)
            list_cleaned_average_neg_log_likelihood.append(cleaned_average_neg_log_likelihood)
            list_cleaned_neg_log_likelihood.append(cleaned_neg_log_likelihood)

    return {
        'average_neg_log_likelihood': list_average_neg_log_likelihoods,
        'neg_log_likelihood': list_neg_log_likelihood,
        'generation_logits': list_generation_logits,
        
        'cleaned_average_neg_log_likelihood': list_cleaned_average_neg_log_likelihood,
        'cleaned_neg_log_likelihood': list_cleaned_neg_log_likelihood,
        'cleaned_generation_logits': list_cleaned_generation_logits,
    }


@torch.no_grad()
def prepare_likelihood(average_neg_log_likelihood, 
                       neg_log_likelihood,
                       generation_logits=None, 
                       cleaned_average_neg_log_likelihood=None, 
                       cleaned_neg_log_likelihood=None, 
                       cleaned_generation_logits=None, 
                       compute_cleaned=False,
                       store_logits=True):
    list_likelihood_dicts = []
    if isinstance(average_neg_log_likelihood, list):
        for i in range(len(average_neg_log_likelihood)):
            
            likelihood_dict = {
                'average_neg_log_likelihood': [average_neg_log_likelihood[i]],
                'neg_log_likelihood': [neg_log_likelihood[i]],
                'generation_logits': [generation_logits[i]] if (store_logits and generation_logits is not None and len(generation_logits) > i) else [],
                'cleaned_average_neg_log_likelihood': [cleaned_average_neg_log_likelihood[i]] if compute_cleaned else [],
                'cleaned_neg_log_likelihood': [cleaned_neg_log_likelihood[i]] if compute_cleaned else [],
                'cleaned_generation_logits': [cleaned_generation_logits[i]] if (compute_cleaned and store_logits and cleaned_generation_logits is not None and len(cleaned_generation_logits) > i) else [],
                
            }

            list_likelihood_dicts.append(likelihood_dict)
    else:
        assert not isinstance(neg_log_likelihood, list) and \
               not isinstance(cleaned_neg_log_likelihood, list) and \
               not isinstance(cleaned_average_neg_log_likelihood, list)
        
        likelihood_dict = {
            'average_neg_log_likelihood': [average_neg_log_likelihood],
            'neg_log_likelihood': [neg_log_likelihood],
            'generation_logits': [generation_logits] if (store_logits and generation_logits is not None) else [],
            'cleaned_average_neg_log_likelihood': [cleaned_average_neg_log_likelihood] if compute_cleaned else [],
            'cleaned_neg_log_likelihood': [cleaned_neg_log_likelihood] if compute_cleaned else [],
            'cleaned_generation_logits': [cleaned_generation_logits] if (compute_cleaned and store_logits and cleaned_generation_logits is not None) else [],
        }

        list_likelihood_dicts.append(likelihood_dict)

    return list_likelihood_dicts

def prepare_results_new(num_samples, 
                    run_key, 
                    metric=None, 
                    start_sample_id=0, 
                    base_path='results',
                    offset=0,
                    batch_size=50):  # Process in smaller batches
    """
    Memory-efficient implementation that loads and processes results in batches.
    """
    list_results_dict = []
    list_correct_labels = []
    dataset_size = 0
    
    # Process in smaller batches to control memory usage
    for batch_start in tqdm(range(offset + start_sample_id, offset + num_samples, batch_size)):
        batch_end = min(batch_start + batch_size, offset + num_samples)
        batch_results = []
        batch_labels = []
        
        for i in range(batch_start, batch_end):
            try:
                # Load one file at a time (support gzip compression)
                import gzip
                file_path_gz = os.path.join(base_path, f'results_dict_{i}.pkl.gz')
                file_path_normal = os.path.join(base_path, f'results_dict_{i}.pkl')
                # Skip if neither compressed nor normal pickle exists
                if not os.path.exists(file_path_gz) and not os.path.exists(file_path_normal):
                    continue
                    
                if os.path.exists(file_path_gz):
                    with gzip.open(file_path_gz, 'rb') as f:
                        results_dict = pickle.load(f)
                else:
                    with open(file_path_normal, 'rb') as f:
                        results_dict = pickle.load(f)
                    
                # Determine target run key safely
                target_key = run_key
                if target_key not in results_dict:
                    for fallback_k in ['ours', 'ensemble', 'layer_ensemble', 'sdlg', 'baseline']:
                        if fallback_k in results_dict:
                            target_key = fallback_k
                            break
                            
                if target_key not in results_dict:
                    continue
                    
                gens = results_dict[target_key].get('generations', [])
                if not gens or len(gens) == 0:
                    continue
                if isinstance(gens[0], dict) and 'generation_ids' in gens[0]:
                    g_ids = gens[0]['generation_ids']
                    if len(g_ids) == 0 or (isinstance(g_ids[0], (list, torch.Tensor)) and len(g_ids[0]) == 0):
                        continue
                
                # Process generations and likelihoods safely
                prepared_generation = []
                prepared_likelihoods = []
                liks = results_dict[target_key].get('likelihoods', [])
                for generations, likelihoods in zip(gens, liks):
                    prepared_generation += prepare_generated_text(**generations)
                    prepared_likelihoods += prepare_likelihood(**likelihoods)
                
                results_dict[target_key]['generations'] = prepared_generation
                results_dict[target_key]['likelihoods'] = prepared_likelihoods
                
                # Extract metric safely with fallback
                if metric is not None:
                    c_dict = results_dict.get("correctness_dict", {})
                    if metric in c_dict:
                        batch_labels.append(c_dict[metric])
                    elif 'bleurt' in c_dict:
                        batch_labels.append(c_dict['bleurt'])
                    elif 'rougeL' in c_dict:
                        batch_labels.append(c_dict['rougeL'])
                    elif 'exact_match' in c_dict:
                        batch_labels.append(c_dict['exact_match'])
                    elif len(c_dict) > 0:
                        batch_labels.append(list(c_dict.values())[0])
                    else:
                        batch_labels.append(0.0)
                
                # Add to batch results
                batch_results.append(results_dict)
                dataset_size += 1
                
            except Exception as e:
                # Log first 3 errors to console for visibility
                if dataset_size == 0 and i < batch_start + 3:
                    print(f"[DEBUG prepare_results] Error processing sample #{i}: {e}")
                continue
        
        # Add batch results to the main list
        list_results_dict.extend(batch_results)
        list_correct_labels.extend(batch_labels)
        
        # Explicitly trigger garbage collection after each batch
        import gc
        gc.collect()
    
    return list_results_dict, list_correct_labels, dataset_size


# updated function, added the questions list
def prepare_results(num_samples, 
                    run_key, 
                    metric=None, 
                    start_sample_id=0, 
                    base_path='results',
                    sample_indices=None):

    list_results_dict = []
    dataset_size = 0
    list_correct_labels = []
    
    indices_to_iterate = sample_indices if sample_indices is not None else list(range(start_sample_id, num_samples))
    
    for i in tqdm(indices_to_iterate, total=len(indices_to_iterate)):
        try:
            import gzip
            file_path_gz = os.path.join(base_path, f'results_dict_{i}.pkl.gz')
            file_path_normal = os.path.join(base_path, f'results_dict_{i}.pkl')
            
            if not os.path.exists(file_path_gz) and not os.path.exists(file_path_normal):
                continue
                
            if os.path.exists(file_path_gz):
                with gzip.open(file_path_gz, 'rb') as f:
                    results_dict = pickle.load(f)
            else:
                with open(file_path_normal, 'rb') as f:
                    results_dict = pickle.load(f)

            # Determine target run key safely
            target_key = run_key
            if target_key not in results_dict:
                for fallback_k in ['ours', 'ensemble', 'layer_ensemble', 'sdlg', 'baseline']:
                    if fallback_k in results_dict:
                        target_key = fallback_k
                        break
                        
            if target_key not in results_dict:
                continue

            gens = results_dict[target_key].get('generations', [])
            if not gens or len(gens) == 0:
                continue
            if isinstance(gens[0], dict) and 'generation_ids' in gens[0]:
                g_ids = gens[0]['generation_ids']
                if len(g_ids) == 0 or (isinstance(g_ids[0], (list, torch.Tensor)) and len(g_ids[0]) == 0):
                    continue

            prepared_generation = []
            prepared_likelihoods = []
            liks = results_dict[target_key].get('likelihoods', [])
            for generations, likelihoods in zip(gens, liks):
                prepared_generation += prepare_generated_text(**generations)
                prepared_likelihoods += prepare_likelihood(**likelihoods)
            results_dict[target_key]['generations'] = prepared_generation
            results_dict[target_key]['likelihoods'] = prepared_likelihoods

            # Extract metric safely with fallback
            if metric is not None:
                c_dict = results_dict.get("correctness_dict", {})
                if metric in c_dict:
                    list_correct_labels.append(c_dict[metric])
                elif 'bleurt' in c_dict:
                    list_correct_labels.append(c_dict['bleurt'])
                elif 'rougeL' in c_dict:
                    list_correct_labels.append(c_dict['rougeL'])
                elif 'exact_match' in c_dict:
                    list_correct_labels.append(c_dict['exact_match'])
                elif len(c_dict) > 0:
                    list_correct_labels.append(list(c_dict.values())[0])
                else:
                    list_correct_labels.append(0.0)

            list_results_dict.append(results_dict)
            dataset_size += 1
            
        except Exception as e:
            if dataset_size == 0 and len(list_results_dict) == 0:
                print(f"[DEBUG prepare_results] Error loading sample #{i}: {e}")
            continue

    return list_results_dict, list_correct_labels, dataset_size

# ---------------------------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def compute_semantic_pairs(generations, 
                           deberta_tokenizer, 
                           deberta_model, 
                           question, 
                           device, 
                           compute_cleaned=False):
    
    semantic_pairs = [np.zeros(shape=(len(generations), len(generations)), dtype=bool),
                      np.zeros(shape=(len(generations), len(generations)), dtype=bool)]

    for i, generation_i  in enumerate(generations):
        for j, generation_j in enumerate(generations):

            if i == j:
                semantic_pairs[0][i, j] = True
                semantic_pairs[1][i, j] = True
                continue

            list_iterations = ['generation_text', 'cleaned_generation_text'] if compute_cleaned else ['generation_text']
            for k, genration_key in enumerate(list_iterations):

                # Note: if cleaned generation text is identiacal to generation text, we don't have to check again
                if genration_key == 'cleaned_generation_text' and generation_i['generation_text'][0] == generation_i['cleaned_generation_text'][0] \
                    and generation_j['generation_text'][0] == generation_j['cleaned_generation_text'][0]:
                    semantic_pairs[1][i, j] = semantic_pairs[0][i, j]
                else:
                    
                    if generation_i[genration_key][0].lower() == generation_j[genration_key][0].lower():
                        semantic_pairs[k][i, j] = True
                    else:
                        qa_i = question + ' ' + generation_i[genration_key][0]
                        qa_j = question + ' ' + generation_j[genration_key][0]

                        input_sequence = qa_i + ' [SEP] ' + qa_j
                        encoded_input = deberta_tokenizer.encode(input_sequence, padding=True)
                        prediction = deberta_model(torch.tensor([encoded_input], device=device))['logits']
                        predicted_label = torch.argmax(prediction, dim=1)

                        semantic_pairs[k][i, j] = True if predicted_label != 0 else False
            
    return {
        'semantic_pairs': semantic_pairs[0],
        'cleaned_semantic_pairs': semantic_pairs[1] if compute_cleaned else [],
    }


@torch.no_grad()
def compute_batched_semantic_pairs(generations, 
                                   deberta_tokenizer, 
                                   deberta_model, 
                                   question, 
                                   device, 
                                   compute_cleaned=False, 
                                   batch_size=32):
    
    semantic_pairs = [np.zeros(shape=(len(generations), len(generations)), dtype=bool),
                      np.zeros(shape=(len(generations), len(generations)), dtype=bool)]

    list_iterations = ['generation_text', 'cleaned_generation_text'] if compute_cleaned else ['generation_text']
    for k, genration_key in enumerate(list_iterations):

        if genration_key == 'cleaned_generation_text':
            if False not in [g['generation_text'][0] == g['cleaned_generation_text'][0] for g in generations]:
                semantic_pairs[1] = semantic_pairs[0]
                break

        num_batches = range(math.ceil(len(generations) / batch_size))

        for batch_i in num_batches:
            for batch_j in num_batches:
                model_input =  []
                # ensemble batch
                for generation_i in generations[batch_size*batch_i:batch_size*(batch_i+1)]:
                    for generation_j in generations[batch_size*batch_j:batch_size*(batch_j+1)]:

                        qa_i = question + ' ' + generation_i[genration_key][0]
                        qa_j = question + ' ' + generation_j[genration_key][0]

                        input_sequence = qa_i + ' [SEP] ' + qa_j
                        model_input.append(input_sequence)

                # compute predictions
                try:
                    # try batched input
                    encoded_input = deberta_tokenizer(model_input, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
                    prediction = deberta_model(**encoded_input)['logits']
                    predicted_labels = torch.argmax(prediction, dim=1)
                except:
                    # Note: if batched input is OOM, compute predictions for each pair individually
                    predicted_labels = []
                    for input_sequence in model_input:
                        encoded_input = deberta_tokenizer.encode(input_sequence, padding=True, truncation=True, max_length=512)
                        encoded_input = torch.tensor([encoded_input], device=device)
                        prediction = deberta_model(encoded_input)['logits']
                        predicted_labels.append(torch.argmax(prediction, dim=1).item())

                # assign values to semantic_pairs
                index = 0
                for i in range(len(generations[batch_size*batch_i:batch_size*(batch_i+1)])):
                    for j in range(len(generations[batch_size*batch_j:batch_size*(batch_j+1)])):
                        semantic_pairs[k][batch_size*batch_i + i, batch_size*batch_j + j] = True if predicted_labels[index] != 0 else False
                        index += 1
    
    return {
        'semantic_pairs': semantic_pairs[0],
        'cleaned_semantic_pairs': semantic_pairs[1] if compute_cleaned else [],
    }


@torch.no_grad()
def compute_semantic_clusters(generations, 
                              semantic_pairs, 
                              cleaned_semantic_pairs, 
                              compute_cleaned=False):

    semantic_clusters = [list(range(0, len(generations))), list(range(0, len(generations)))]
    for i, generation_i  in enumerate(generations):
        for j, generation_j in enumerate(generations):

            if j <= i:
                continue

            list_iterations = ['generation_text', 'cleaned_generation_text'] if compute_cleaned else ['generation_text']
            for k, genration_key in enumerate(list_iterations):

                # [cleaned] if cleaned generation text is identiacal to generation text, we don't have to check again
                if genration_key == 'cleaned_generation_text' and generation_i['generation_text'][0] == generation_i['cleaned_generation_text'][0] \
                    and generation_j['generation_text'][0] == generation_j['cleaned_generation_text'][0]:
                    semantic_clusters[1] = semantic_clusters[0]
                    break

                # [all] if the clusters are already the same, we don't have to check again
                if semantic_clusters[k][j] == semantic_clusters[k][i]:
                    continue
                # [all] if generation text is identical, directly assign to same cluster
                elif generation_i[genration_key][0].lower() == generation_j[genration_key][0].lower():
                    semantic_clusters[k][j] = semantic_clusters[k][i]
                # [all] otherwise, check semantic_pairs
                else:
                    if genration_key == 'generation_text':
                        if semantic_pairs[i, j] == True and semantic_pairs[j, i] == True:
                            # Note: equivalent to: if predicted_label != 0 and reverse_predicted_label != 0
                            semantic_clusters[k][j] = semantic_clusters[k][i]
                    else:
                        if cleaned_semantic_pairs[i, j] == True and cleaned_semantic_pairs[j, i] == True:
                            semantic_clusters[k][j] = semantic_clusters[k][i]
    
        return {
        'semantic_clusters': torch.tensor(semantic_clusters[0]),
        'cleaned_semantic_clusters': torch.tensor(semantic_clusters[1]) if compute_cleaned else torch.tensor([]),
        }


@torch.no_grad()
def compute_semantic_entropy_new(weights,
                            mc_estimate_over_clusters,
                            neg_log_likelihoods,
                            semantic_difference,
                            epistem_entropy_list,
                            compute_cleaned=False):
    results = []

    list_iterations = ['', 'cleaned_'] if compute_cleaned else ['']

    # Safe weight clamping to prevent log(0) -> -inf
    if isinstance(weights, torch.Tensor):
        safe_weights = torch.clamp(weights.clone().detach().float(), min=1e-8)
        if safe_weights.sum() > 0:
            safe_weights = safe_weights / safe_weights.sum()
    else:
        safe_weights = torch.tensor(weights, dtype=torch.float32)
        safe_weights = torch.clamp(safe_weights, min=1e-8)

    for cleaned in list_iterations:
        for nll_key in ["average_neg_log_likelihood", "neg_log_likelihood"]:

            for d in neg_log_likelihoods:
                assert len(d[cleaned+nll_key]) == 1, "only single likelihoods supported. prepare likelihoods!"

            # compute log(p(y|x,w)): converting NLL to LL, and handling NaN and negative infinity values
            log_likelihoods = torch.tensor(
                [-1e12 if math.isnan(d[cleaned+nll_key][0]) or d[cleaned+nll_key][0] == float('-inf') else -d[cleaned+nll_key][0] for d in neg_log_likelihoods]
            ) 
            log_likelihoods = torch.clamp(log_likelihoods, min=-1e6, max=0.0)

            # scale LL by weights (all ones for baseline)
            log_likelihoods += torch.log(safe_weights[:len(log_likelihoods)])

            aggregated_log_likelihoods = []
            aggregated_weights = []
            
            # compute p(c|x,w): aggregate LL over clusters
            for semantic_set_id in torch.unique(semantic_difference[cleaned+'semantic_clusters']):
                c_mask = semantic_difference[cleaned+'semantic_clusters'] == semantic_set_id
                aggregated_log_likelihoods.append(torch.logsumexp(log_likelihoods[c_mask], dim=0))
                aggregated_weights.append(torch.sum(safe_weights[:len(log_likelihoods)][c_mask], dim=0))

            aggregated_log_likelihoods = torch.tensor(aggregated_log_likelihoods)
            aggregated_weights = torch.tensor(aggregated_weights)

            # softmax: normalizing + transforming from log space to probability space 
            aggregated_normalized_likelihoods = torch.softmax(aggregated_log_likelihoods, dim=0)

            if mc_estimate_over_clusters:
                entropy = - torch.sum(aggregated_log_likelihoods, dim=0) / float(max(1, aggregated_log_likelihoods.shape[0]))
            else:
                entropy = - torch.sum(aggregated_log_likelihoods * aggregated_normalized_likelihoods, dim=0)

            if torch.isnan(entropy).any() or torch.isinf(entropy).any():
                entropy_val = 0.0
            else:
                entropy_val = float(entropy.item())

            results.append(entropy_val)
    
    # Calculate average for each metric from epistem_entropy_list
    avg_metrics = {}
    
    if epistem_entropy_list and len(epistem_entropy_list) > 0:
        all_metrics = set()
        for item in epistem_entropy_list:
            if isinstance(item, dict):
                all_metrics.update(item.keys())
        
        for metric in sorted(list(all_metrics)):
            valid_values = []
            for item in epistem_entropy_list:
                if isinstance(item, dict) and metric in item:
                    val = item[metric]
                    if val is None:
                        continue
                    if isinstance(val, torch.Tensor):
                        if val.numel() == 1:
                            f = float(val.item())
                            if math.isfinite(f):
                                valid_values.append(f)
                    elif isinstance(val, (int, float, np.floating, np.integer)):
                        f = float(val)
                        if math.isfinite(f):
                            valid_values.append(f)
            if valid_values:
                avg_metrics[f"{metric}"] = float(sum(valid_values) / len(valid_values))
    
    # Build the final result dictionary
    result_dict = {
        "normalised_semantic_entropy": results[0],
        "unnormalised_semantic_entropy": results[1],
    }
    
    # Add cleaned results if computed
    if compute_cleaned:
        result_dict.update({
            "cleaned_normalised_semantic_entropy": results[2] if len(results) > 2 else [],
            "cleaned_unnormalised_semantic_entropy": results[3] if len(results) > 3 else [],
        })
    
    # Add all averaged metrics from epistem_entropy_list
    result_dict.update(avg_metrics)
    
    return result_dict

import math
import torch
import numpy as np

def compute_semantic_entropy_var(weights,
                                  mc_estimate_over_clusters,
                                  neg_log_likelihoods,
                                  semantic_difference,
                                  epistem_entropy_list,
                                  compute_cleaned=False,
                                  debug=False):
    """
    Compute semantic entropy with epistemic uncertainty from multiple layer ranges.
    
    Args:
        epistem_entropy_list: List[List[Dict]]
            - Outer list: one entry per generation (length = num_generations, e.g., 10)
            - Inner list: one entry per layer range (length = num_ranges, e.g., 31)
            - Dict: contains metrics like 'entropy_mixture'
            
            Example structure:
            [
                [{'entropy_mixture': 0.5}, {'entropy_mixture': 0.4}, ...],  # gen 0, 31 ranges
                [{'entropy_mixture': 0.6}, {'entropy_mixture': 0.5}, ...],  # gen 1, 31 ranges
                ...
            ]
    """
    results = []
    weights = weights.to(dtype=torch.float32)
    
    # Debugging: Check input structure
    if debug:
        print("\n" + "="*80)
        print("DEBUG: compute_semantic_entropy_var called")
        print(f"Number of generations: {len(neg_log_likelihoods)}")
        print(f"Weights shape: {weights.shape}, sum: {weights.sum():.4f}")
        
        if epistem_entropy_list:
            print(f"\nEpistemic entropy structure:")
            print(f"  - Type: {type(epistem_entropy_list)}")
            print(f"  - Num generations: {len(epistem_entropy_list)}")
            
            if len(epistem_entropy_list) > 0:
                print(f"  - Type of first element: {type(epistem_entropy_list[0])}")
                
                if isinstance(epistem_entropy_list[0], list):
                    print(f"  - Ranges per generation: {[len(x) for x in epistem_entropy_list[:3]]}...")
                    
                    if len(epistem_entropy_list[0]) > 0:
                        print(f"  - Type of first range: {type(epistem_entropy_list[0][0])}")
                        print(f"  - Keys in first range: {list(epistem_entropy_list[0][0].keys())}")
                        print(f"  - Sample values from generation 0:")
                        for range_idx in range(min(3, len(epistem_entropy_list[0]))):
                            val = epistem_entropy_list[0][range_idx].get('entropy_mixture', 'MISSING')
                            print(f"      Range {range_idx}: {val}")
    
    list_iterations = ['', 'cleaned_'] if compute_cleaned else ['']
    
    # ============================================================================
    # STEP 1: Compute standard semantic entropy
    # ============================================================================
    for cleaned in list_iterations:
        for nll_key in ["average_neg_log_likelihood", "neg_log_likelihood"]:
            
            # Extract negative log-likelihoods
            for d in neg_log_likelihoods:
                assert len(d[cleaned + nll_key]) == 1, "only single likelihoods supported"
            
            # Convert NLLs to log-likelihoods safely
            log_likelihoods_np = np.array([
                -1e12 if math.isnan(d[cleaned + nll_key][0]) or d[cleaned + nll_key][0] == float('-inf')
                else -d[cleaned + nll_key][0]
                for d in neg_log_likelihoods
            ], dtype=np.float32)
            log_likelihoods = torch.from_numpy(log_likelihoods_np).to(weights.device)
            
            assert torch.all(log_likelihoods <= 0), "likelihood bigger than 1!"
            
            # Apply weights (additive in log-space)
            log_likelihoods += torch.log(weights)
            
            # Aggregate over semantic clusters
            clusters = semantic_difference[cleaned + 'semantic_clusters']
            unique_clusters = torch.unique(clusters)
            
            if debug:
                print(f"\n{cleaned}{nll_key}:")
                print(f"  - Unique clusters: {len(unique_clusters)}")
                print(f"  - Cluster distribution: {[(c.item(), (clusters == c).sum().item()) for c in unique_clusters]}")
            
            aggregated_log_likelihoods = []
            aggregated_weights = []
            
            for semantic_set_id in unique_clusters:
                mask = clusters == semantic_set_id
                aggregated_log_likelihoods.append(torch.logsumexp(log_likelihoods[mask], dim=0))
                aggregated_weights.append(torch.sum(weights[mask]))
            
            aggregated_log_likelihoods = torch.stack(aggregated_log_likelihoods)
            aggregated_weights = torch.stack(aggregated_weights)
            
            # Normalize in log-probability space
            aggregated_normalized_likelihoods = torch.softmax(aggregated_log_likelihoods, dim=0)
            
            if mc_estimate_over_clusters:
                entropy = - torch.mean(aggregated_log_likelihoods)
            else:
                entropy = - torch.sum(aggregated_log_likelihoods * aggregated_normalized_likelihoods)
            
            assert not torch.isinf(entropy).any(), f"semantic_entropy is inf: {entropy}"
            assert not torch.isnan(entropy).any(), f"semantic_entropy is nan: {entropy}"
            
            if debug:
                print(f"  - Computed entropy: {entropy.item():.4f}")
            
            results.append(entropy.item())
    
    # ============================================================================
    # STEP 2: Average epistemic entropy metrics across generations FOR EACH RANGE
    # ============================================================================
    avg_metrics = {}
    
    # Check if we have the expected nested structure
    if (epistem_entropy_list and 
        len(epistem_entropy_list) > 0 and 
        isinstance(epistem_entropy_list[0], list) and
        len(epistem_entropy_list[0]) > 0 and
        isinstance(epistem_entropy_list[0][0], dict)):
        
        num_generations = len(epistem_entropy_list)
        num_ranges = len(epistem_entropy_list[0])  # Should be 31
        all_metric_keys = epistem_entropy_list[0][0].keys()
        
        if debug:
            print(f"\nProcessing epistemic uncertainty:")
            print(f"  - Number of generations: {num_generations}")
            print(f"  - Number of layer ranges: {num_ranges}")
            print(f"  - Metrics to average: {list(all_metric_keys)}")
        
        # Track statistics for debugging
        ranges_processed = 0
        ranges_with_values = 0
        
        # For each layer range (0 to 30)
        for range_idx in range(num_ranges):
            # For each metric (e.g., 'entropy_mixture')
            for metric in all_metric_keys:
                # Collect values across ALL GENERATIONS for this specific range
                values = []
                for gen_idx, gen_entropies in enumerate(epistem_entropy_list):
                    # Check if this generation has data for this range
                    if range_idx < len(gen_entropies) and metric in gen_entropies[range_idx]:
                        values.append(gen_entropies[range_idx][metric])
                
                # Average across generations
                if values:
                    avg_value = sum(values) / len(values)
                    avg_metrics[f"{metric}_{range_idx}"] = avg_value
                    ranges_with_values += 1
                    
                    if debug and metric == 'entropy_mixture' and range_idx < 3:
                        print(f"  - Range {range_idx} ('{metric}'):")
                        print(f"      Values from {len(values)} generations: {[f'{v:.4f}' for v in values[:3]]}...")
                        print(f"      Average: {avg_value:.4f}")
                elif debug and metric == 'entropy_mixture' and range_idx < 3:
                    print(f"  - Range {range_idx}: No values collected (missing data)")
                
                ranges_processed += 1
        
        if debug:
            print(f"\nAggregation summary:")
            print(f"  - Total metric-range combinations: {ranges_processed}")
            print(f"  - Successful aggregations: {ranges_with_values}")
            print(f"  - Missing data: {ranges_processed - ranges_with_values}")
            
            # Show distribution of values
            if 'entropy_mixture_0' in avg_metrics:
                entropy_values = [avg_metrics[f'entropy_mixture_{i}'] for i in range(num_ranges) 
                                 if f'entropy_mixture_{i}' in avg_metrics]
                if entropy_values:
                    print(f"  - Entropy_mixture statistics:")
                    print(f"      Min: {min(entropy_values):.4f}")
                    print(f"      Max: {max(entropy_values):.4f}")
                    print(f"      Mean: {sum(entropy_values)/len(entropy_values):.4f}")
    
    elif epistem_entropy_list and debug:
        print("\nWARNING: Unexpected epistemic_entropy_list structure!")
        print(f"  - Type: {type(epistem_entropy_list)}")
        if len(epistem_entropy_list) > 0:
            print(f"  - First element type: {type(epistem_entropy_list[0])}")
            if isinstance(epistem_entropy_list[0], dict):
                print(f"  - This looks like the OLD format (list of dicts, not list of lists of dicts)")
    
    # ============================================================================
    # STEP 3: Build final result dictionary
    # ============================================================================
    result_dict = {
        "normalised_semantic_entropy": results[0],
        "unnormalised_semantic_entropy": results[1],
    }
    
    if compute_cleaned:
        result_dict.update({
            "cleaned_normalised_semantic_entropy": results[2],
            "cleaned_unnormalised_semantic_entropy": results[3],
        })
    
    # Add all averaged epistemic metrics
    result_dict.update(avg_metrics)
    
    if debug:
        print(f"\nFinal result dictionary:")
        print(f"  - Total keys: {len(result_dict)}")
        print(f"  - Semantic entropy keys: {[k for k in result_dict.keys() if 'semantic' in k]}")
        epistemic_keys = [k for k in result_dict.keys() if 'entropy_mixture' in k]
        print(f"  - Epistemic entropy keys: {len(epistemic_keys)}")
        
        if epistemic_keys:
            sample_keys = epistemic_keys[:3]
            print(f"  - Sample epistemic values:")
            for key in sample_keys:
                print(f"      {key}: {result_dict[key]:.4f}")
        
        print("="*80 + "\n")
    
    return result_dict


@torch.no_grad()
def compute_semantic_entropy(weights, 
                             mc_estimate_over_clusters, 
                             neg_log_likelihoods, 
                             semantic_difference,
                             epistem_entropy_list,
                             compute_cleaned=False):
    """
    Computes Shannon semantic entropy and aggregates all layer-wise epistemic disagreement metrics.
    """
    return compute_semantic_entropy_new(
        weights=weights,
        mc_estimate_over_clusters=mc_estimate_over_clusters,
        neg_log_likelihoods=neg_log_likelihoods,
        semantic_difference=semantic_difference,
        epistem_entropy_list=epistem_entropy_list,
        compute_cleaned=compute_cleaned
    )



# @torch.no_grad()
def compute_semantic_paris(base_path,
                           model_type, 
                           deberta_tokenizer, 
                           deberta_model, 
                           num_instances, 
                           device,
                           nli_model_sent = 'sentence-transformers/sentence-t5-large',
                           # nli_model_sent = 'sentence-transformers/all-MiniLM-L6-v2',
                           offset= 0
                          ):
    for method in ['sdlg', 'baseline']:

        removed_sample_ids = []

        for i in tqdm(range(0, num_instances)):
            i = i + offset
            import gzip
            file_path_gz = os.path.join(base_path, f'results_dict_{i}.pkl.gz')
            file_path_normal = os.path.join(base_path, f'results_dict_{i}.pkl')
            is_compressed = False
            try:
                if os.path.exists(file_path_gz):
                    with gzip.open(file_path_gz, 'rb') as f:
                        results_dict = pickle.load(f)
                    is_compressed = True
                else:
                    with open(file_path_normal, 'rb') as f:
                        results_dict = pickle.load(f)
            except:
                if i not in removed_sample_ids:
                    removed_sample_ids.append(i)
                continue

            if len(results_dict[method]['generations']) == 0 or len(results_dict[method]['generations'][0]['generation_ids'][0]) == 0:
                if i not in removed_sample_ids:
                    removed_sample_ids.append(i)
                continue

            prepared_generation = []
            prepared_likelihoods = []
            for generations, likelihoods in zip(results_dict[method]['generations'], results_dict[method]['likelihoods']):
                prepared_generation += prepare_generated_text(**generations)
                prepared_likelihoods += prepare_likelihood(**likelihoods)
            results_dict[method]['generations'] = prepared_generation
            results_dict[method]['likelihoods'] = prepared_likelihoods
            ensemble_gen_list = [x['generation_text'][0] for x in results_dict[method]['generations']]

            if (f'semantic_pairs_{model_type}' not in results_dict[method].keys() or 
                results_dict[method][f'semantic_pairs_{model_type}']['semantic_pairs'].shape[0] != len(results_dict[method]['generations'])):
                
                pairs = compute_batched_semantic_pairs(generations=results_dict[method]["generations"],
                                                       deberta_tokenizer=deberta_tokenizer, 
                                                       deberta_model=deberta_model, 
                                                       question=results_dict['question'], 
                                                       device=device,
                                                       compute_cleaned=False,
                                                       batch_size=32)
                
                results_dict[method][f'semantic_pairs_{model_type}'] = pairs
                print(i, pairs)
                target_pairs = torch.tensor(pairs['semantic_pairs'].astype(float), dtype=torch.float32, device='cuda:0')
                
                seq_level_imp = sentence_importance(ensemble_gen_list[0], ensemble_gen_list[1:], nli_model_sent, target_pairs)
                seq_level_imp = seq_level_imp.detach().to('cpu')
                results_dict[method]['seq_level_impo'] = seq_level_imp
                
                # print('gg')
                if is_compressed:
                    with gzip.open(file_path_gz, 'wb') as f:
                        pickle.dump(results_dict, f)
                else:
                    with open(file_path_normal, 'wb') as f:
                        pickle.dump(results_dict, f)

        print(f"{base_path} - {method}: removed_sample_ids: {removed_sample_ids}")

# @torch.no_grad()
def compute_semantic_paris_new(base_path,
                           model_type, 
                           deberta_tokenizer, 
                           deberta_model, 
                           num_instances, 
                           device,
                           nli_model_sent = 'sentence-transformers/sentence-t5-large',
                           # nli_model_sent = 'sentence-transformers/all-MiniLM-L6-v2',
                           offset= 0
                          ):
    
    # for method in ['sdlg', 'dola_sdlg']: 
    for method in ['sdlg']: 
        print('method: ', method)
        removed_sample_ids = []
        for i in tqdm(range(0, num_instances)):
            i = i + offset
            import gzip
            file_path_gz = os.path.join(base_path, f'results_dict_{i}.pkl.gz')
            file_path_normal = os.path.join(base_path, f'results_dict_{i}.pkl')
            is_compressed = False
            try:
                if os.path.exists(file_path_gz):
                    with gzip.open(file_path_gz, 'rb') as f:
                        results_dict = pickle.load(f)
                    is_compressed = True
                else:
                    with open(file_path_normal, 'rb') as f:
                        results_dict = pickle.load(f)
            except:
                if i not in removed_sample_ids:
                    removed_sample_ids.append(i)
                continue
            if len(results_dict[method]['generations']) == 0 or len(results_dict[method]['generations'][0]['generation_ids'][0]) == 0:
                if i not in removed_sample_ids:
                    removed_sample_ids.append(i)
                continue
            prepared_generation = []
            prepared_likelihoods = []
            for generations, likelihoods in zip(results_dict[method]['generations'], results_dict[method]['likelihoods']):
                prepared_generation += prepare_generated_text(**generations)
                prepared_likelihoods += prepare_likelihood(**likelihoods)
            results_dict[method]['generations'] = prepared_generation
            results_dict[method]['likelihoods'] = prepared_likelihoods
            ensemble_gen_list = [x['generation_text'][0] for x in results_dict[method]['generations']]
            
            if (f'semantic_pairs_{model_type}' not in results_dict[method].keys() or 
                results_dict[method][f'semantic_pairs_{model_type}']['semantic_pairs'].shape[0] != len(results_dict[method]['generations'])):
                
                pairs = compute_batched_semantic_pairs(generations=results_dict[method]["generations"],
                                                       deberta_tokenizer=deberta_tokenizer, 
                                                       deberta_model=deberta_model, 
                                                       question=results_dict['question'], 
                                                       device=device,
                                                       compute_cleaned=False,
                                                       batch_size=32)

                results_dict[method][f'semantic_pairs_{model_type}'] = pairs
                target_pairs = torch.tensor(pairs['semantic_pairs'].astype(float), dtype=torch.float32, device=device if torch.cuda.is_available() else 'cpu')
                seq_level_imp = sentence_importance(results_dict['question'], ensemble_gen_list[0], ensemble_gen_list[1:], nli_model_sent, target_pairs)

                if seq_level_imp is None:
                    seq_level_imp = torch.ones(len(results_dict[method]['generations'])) / len(results_dict[method]['generations'])

                # NEW: Weight down by generation likelihood
                likelihood_values = []
                for likelihood_dict in prepared_likelihoods:
                    avg_neg_log_likelihood = likelihood_dict['average_neg_log_likelihood'][0]
                    likelihood = torch.exp(torch.tensor(-avg_neg_log_likelihood))
                    likelihood_values.append(likelihood)

                imp_weight = seq_level_imp.detach().to('cpu')
                results_dict[method]['seq_level_impo'] = imp_weight

                if is_compressed:
                    with gzip.open(file_path_gz, 'wb') as f:
                        pickle.dump(results_dict, f)
                else:
                    with open(file_path_normal, 'wb') as f:
                        pickle.dump(results_dict, f)

                # Periodic GPU memory cleanup to prevent fragmentation stalls
                if torch.cuda.is_available() and i % 50 == 0:
                    torch.cuda.empty_cache()