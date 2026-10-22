import os
import sys
import torch
import yaml
import pickle
import gzip
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset

# Add src/ to Python path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from layer_ensemble_var import generate_layer_ensemble_var_sequences, generate_semantically_diverse_output_sequences
from utils import generate_text, compute_likelihood, seed_everything
from args import Args

SUPPORTED_MOE_MODELS = {
    "deepseek-16b": "deepseek-ai/deepseek-moe-16b-chat",
    "qwen-moe": "Qwen/Qwen1.5-MoE-A2.7B-Chat"
}

def setup_moe_special_tokens(tokenizer, llm_model, args):
    """
    Safely configures pad tokens and chat stop token IDs for MoE Chat models.
    """
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            llm_model.resize_token_embeddings(len(tokenizer))
            
    llm_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Collect chat stop token IDs (<|im_end|>, <|endoftext|>, <|eot_id|>)
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_ids.extend(tokenizer.eos_token_id)
        else:
            eos_ids.append(tokenizer.eos_token_id)
            
    for token_str in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in eos_ids:
            eos_ids.append(tid)
            
    args.eos_token_ids = list(set(eos_ids))
    args.invalid_ids = [tokenizer.pad_token_id] if tokenizer.pad_token_id is not None else []


def run_moe_pipeline(model_key="deepseek-16b", num_samples=3):
    """
    Runs the SDLG + layer-wise ensembling pipeline on specified MoE chat model.
    """
    seed_everything(42)
    model_name = SUPPORTED_MOE_MODELS.get(model_key, model_key)
    
    print("=" * 60)
    print(f"RUNNING SDLG PIPELINE ON MOE CHAT MODEL: {model_name}")
    print("=" * 60)
    
    args = Args()
    args.llm_model = model_name
    args.is_moe = True  # Explicitly set MoE architecture flag
    args.num_samples = num_samples
    args.token_prob_threshold = 0.005
    args.num_total_generations = 10
    args.max_length_of_generated_sequence = 30
    args.alphas = (0.33, 0.33, 0.34)
    
    # Restrict MoE layer-wise ensembling to top mature layers (e.g. top 6 layers)
    args.num_top_layers = 6
    args.num_top_layers_list = list(range(1, 7))
    
    auth_token = os.environ.get("HF_TOKEN", None)
    
    print(f"Loading MoE LLM Model and Tokenizer: {args.llm_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True, token=auth_token)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        token=auth_token
    )
    
    device_llm = next(llm_model.parameters()).device
    print(f"MoE LLM loaded on device: {device_llm}")
    
    setup_moe_special_tokens(tokenizer, llm_model, args)
    
    # DeBERTa substitution encoder
    args.deberta_model = "deberta-large-mnli"
    device_deberta = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading DeBERTa Model and Tokenizer: microsoft/{args.deberta_model}...")
    deberta_tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{args.deberta_model}")
    deberta_model = AutoModelForSequenceClassification.from_pretrained(
        f"microsoft/{args.deberta_model}",
        device_map="auto" if torch.cuda.is_available() else None
    )
    if not torch.cuda.is_available():
        deberta_model = deberta_model.to(device_deberta)
        
    deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings(
        torch.tensor([list(range(0, deberta_tokenizer.vocab_size))]).to(device_deberta)
    ).squeeze().detach()
    
    print("Loading TruthfulQA (generation split)...")
    dataset = load_dataset('truthfulqa/truthful_qa', 'generation', split='validation')
    
    output_results = []
    
    for idx in tqdm(range(min(num_samples, len(dataset))), desc=f"Evaluating {model_key}"):
        item = dataset[idx]
        question = item['question']
        
        # Apply Chat Template for MoE Chat Models if available
        messages = [{"role": "user", "content": f"Answer the following question concisely: {question}"}]
        try:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_text = f"Question: {question}\nAnswer:"
            
        input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device_llm)
        len_prompt = input_ids.shape[1]
        
        results_dict = {
            'input_ids': input_ids.cpu(),
            'question': question,
            'sdlg': {'generations': [], 'likelihoods': [], 'epistem_entropies': []},
            'baseline': {'generations': [], 'likelihoods': []}
        }
        
        # Most likely generation
        initial_gen = generate_text(
            args=args,
            model=llm_model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            len_prompt=len_prompt,
            decoding_method='most_likely',
            device=device_llm
        )
        
        initial_likelihood = compute_likelihood(
            prompt=input_ids[0].cpu(),
            generation=initial_gen,
            model=llm_model,
            device=device_llm,
            store_logits=True
        )
        
        results_dict['baseline']['generations'] = initial_gen['generation_text']
        
        # SDLG generation with layer-wise ensembling
        sdlg_results = generate_semantically_diverse_output_sequences(
            results_dict=results_dict,
            deberta_model=deberta_model,
            deberta_tokenizer=deberta_tokenizer,
            device_deberta=device_deberta,
            deberta_embeddings=deberta_embeddings,
            model=llm_model,
            tokenizer=tokenizer,
            device_llm=device_llm,
            input_ids=input_ids,
            prompt=input_ids[0].cpu(),
            question=question,
            initial_generation=initial_gen,
            initial_likelihood=initial_likelihood,
            key='sdlg',
            args=args
        )
        
        output_results.append(sdlg_results)
        
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    os.makedirs(output_dir, exist_ok=True)
    out_filepath = os.path.join(output_dir, f"moe_pipeline_results_{model_key}.pkl")
    
    with open(out_filepath, "wb") as f:
        pickle.dump(output_results, f)
        
    print(f"\nPipeline successfully completed! Results saved to: {out_filepath}")
    return out_filepath

if __name__ == "__main__":
    # Standard entry point for running MoE pipeline
    model_choice = sys.argv[1] if len(sys.argv) > 1 else "deepseek-16b"
    run_moe_pipeline(model_key=model_choice, num_samples=3)
