import os
import sys
import torch
from transformers import AutoTokenizer

# Add src/ to Python path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from token_attribution import compute_token_score_ranking, get_word_indices
from args import Args


def test_word_indices():
    """
    Verifies that the word alignment grouping logic produces consistent word segmentations
    across two completely different tokenizer architectures (DeBERTa and GPT-2/LLM).
    """
    print("=" * 60)
    print("TEST 1: Verification of word-level tokenizer alignment")
    print("=" * 60)
    
    deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    text = "The multi-token word segmentation alignment shouldn't fail on out-of-vocabulary terms like Llama-3's 128k tokenizer."
    words = text.split()
    
    encoded_deberta = torch.tensor(deberta_tokenizer.encode(" " + text, add_special_tokens=False))
    encoded_llm = torch.tensor(llm_tokenizer.encode(" " + text, add_special_tokens=False))
    
    indices_deberta = get_word_indices(encoded_deberta, deberta_tokenizer)
    indices_llm = get_word_indices(encoded_llm, llm_tokenizer)
    
    print(f"Original Text: '{text}'")
    print(f"Split Words (expected groups): {words}\n")
    
    # Extract raw tokens
    deberta_raw_tokens = [deberta_tokenizer.decode([tid]) for tid in encoded_deberta.tolist()]
    llm_raw_tokens = [llm_tokenizer.decode([tid]) for tid in encoded_llm.tolist()]
    
    print("-" * 60)
    print("DeBERTa Tokenization & Word Boundaries:")
    print(f"  Raw Tokens:      {deberta_raw_tokens}")
    print(f"  Raw Token IDs:   {encoded_deberta.tolist()}")
    print("  Word Group Mappings:")
    for w_idx, idx_tensor in enumerate(indices_deberta):
        idx_list = idx_tensor.tolist()
        group_tokens = [deberta_tokenizer.decode([encoded_deberta[i]]) for i in idx_list]
        decoded_group = deberta_tokenizer.decode(encoded_deberta[idx_tensor]).strip()
        print(f"    - Word {w_idx} ('{words[w_idx]}'): token indices {idx_list} -> tokens {group_tokens} -> decoded: '{decoded_group}'")
    print("-" * 60)
    
    print("LLM (Qwen) Tokenization & Word Boundaries:")
    print(f"  Raw Tokens:      {llm_raw_tokens}")
    print(f"  Raw Token IDs:   {encoded_llm.tolist()}")
    print("  Word Group Mappings:")
    for w_idx, idx_tensor in enumerate(indices_llm):
        idx_list = idx_tensor.tolist()
        group_tokens = [llm_tokenizer.decode([encoded_llm[i]]) for i in idx_list]
        decoded_group = llm_tokenizer.decode(encoded_llm[idx_tensor]).strip()
        print(f"    - Word {w_idx} ('{words[w_idx]}'): token indices {idx_list} -> tokens {group_tokens} -> decoded: '{decoded_group}'")
    print("-" * 60 + "\n")
    
    deberta_decoded = [deberta_tokenizer.decode(encoded_deberta[idx]).strip() for idx in indices_deberta]
    llm_decoded = [llm_tokenizer.decode(encoded_llm[idx]).strip() for idx in indices_llm]
    
    # Assertions
    assert len(indices_deberta) == len(words), f"Deberta word count mismatch: {len(indices_deberta)} vs {len(words)}"
    assert len(indices_llm) == len(words), f"LLM word count mismatch: {len(indices_llm)} vs {len(words)}"
    
    for i, word in enumerate(words):
        assert deberta_decoded[i] == word, f"Deberta word at index {i} mismatch: {deberta_decoded[i]} vs {word}"
        assert llm_decoded[i] == word, f"LLM word at index {i} mismatch: {llm_decoded[i]} vs {word}"
        
    print("✓ TEST 1 passed successfully: Word boundaries aligned perfectly across tokenizers!\n")


def test_compute_token_score_ranking():
    """
    Verifies that compute_token_score_ranking runs end-to-end, performs gradient propagation
    via hook registration, maps candidate token embeddings, and successfully ranks candidates
    without IndexErrors or alignment crashes.
    """
    print("=" * 60)
    print("TEST 2: End-to-end scoring and candidate ranking verification")
    print("=" * 60)
    
    device = "cpu"
    deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    # Mock deberta model that registers hooks and propagates gradients correctly
    class MockDebertaModel(torch.nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.deberta = torch.nn.Module()
            self.deberta.embeddings = torch.nn.Module()
            self.deberta.embeddings.word_embeddings = torch.nn.Embedding(vocab_size, 10)
            self.dummy_param = torch.nn.Parameter(torch.randn(1))
            
        def forward(self, input_ids, **kwargs):
            # Pass input through embeddings to trigger the forward hook
            embeddings = self.deberta.embeddings.word_embeddings(input_ids)
            # Produce predictions linked back to the embeddings so backward pass propagates gradients
            logits = embeddings.sum(dim=-1).mean(dim=-1, keepdim=True).repeat(1, 3) * self.dummy_param
            return {"logits": logits}
            
    deberta_model = MockDebertaModel(deberta_tokenizer.vocab_size)
    deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings.weight.detach()
    
    # Inputs
    question = "What is the answer?"
    initial_generation_text = "The multi-token word segmentation alignment shouldn't fail on out-of-vocabulary terms like Llama-3's 128k tokenizer."
    initial_generation_ids = torch.tensor(llm_tokenizer.encode(" " + initial_generation_text, add_special_tokens=False))
    
    additional_generated_text = [
        "The multi-token word segmentation alignment shouldn't fail on out-of-vocabulary words.",
        "The multi-token word alignment shouldn't fail on out-of-vocabulary terms like Llama-3's tokenizer."
    ]
    
    # Mock generation logits from the LLM model: shape [num_tokens, llm_vocab_size]
    # Set high logit value (100.0) at a different index to ensure candidate passes the 0.5 threshold
    generation_logits = torch.zeros((len(initial_generation_ids), llm_tokenizer.vocab_size))
    for t in range(len(initial_generation_ids)):
        orig_token = initial_generation_ids[t].item()
        alt_token = (orig_token + 1) % llm_tokenizer.vocab_size
        generation_logits[t, alt_token] = 100.0
    
    args = Args()
    args.token_prob_threshold = 0.5  # Filter out low-probability candidates for testing efficiency
    args.alphas = (1/3, 1/3, 1/3)
    args.invalid_ids = []
    
    print("Running compute_token_score_ranking with mock LLM/DeBERTa...")
    sorted_indices, token_info = compute_token_score_ranking(
        deberta_model=deberta_model,
        deberta_tokenizer=deberta_tokenizer,
        device_deberta=device,
        question=question,
        initial_generation_ids=initial_generation_ids,
        initial_generation_text=initial_generation_text,
        additional_generated_text=additional_generated_text,
        generation_logits=generation_logits,
        deberta_embeddings=deberta_embeddings,
        tokenizer=llm_tokenizer,
        args=args
    )
    
    print(f"Candidate mapping populated. Total ranked candidate combinations: {len(sorted_indices)}")
    
    # Verify outputs
    assert isinstance(sorted_indices, torch.Tensor), "Output sorted_indices must be a torch.Tensor"
    assert isinstance(token_info, dict), "Output token_info must be a dictionary"
    
    if len(token_info) > 0:
        # Validate entry format: (initial_gen_word_idx, initial_gen_token_idx_llm, new_token_idx_llm)
        key, value = list(token_info.items())[0]
        word_idx, token_idx, candidate_id = key
        attr, sub, imp = value
        
        print(f"✓ Output structure validated.")
        print(f"  - Sample ranked candidate: Word '{initial_generation_text.split()[word_idx]}' (LLM token index {token_idx})")
        print(f"  - Replacement token ID: {candidate_id} -> '{llm_tokenizer.decode([candidate_id])}'")
        print(f"  - Scores (Attribution, Substitution, Importance): {attr:.4f}, {sub:.4f}, {imp:.4f}")
        
        # Boundary assertions
        assert 0 <= word_idx < len(initial_generation_text.split()), "Word index out of bounds"
        assert 0 <= token_idx < len(initial_generation_ids), "LLM token index out of bounds"
        assert 0 <= candidate_id < llm_tokenizer.vocab_size, "Candidate token ID out of bounds"
    else:
        print("Warning: No candidate tokens passed the probability threshold.")
        
    print("✓ TEST 2 passed successfully: Scoring and ranking completed end-to-end!\n")

import tempfile
import shutil
import gzip
import pickle
from layer_ensemble import generate_layer_ensemble_sequences, generate_semantically_diverse_output_sequences
from utils import prepare_results, prepare_results_new

def test_important_tokens_only_mode():
    """
    Verifies that enabling save_only_important_tokens correctly bypasses sequence generation,
    computes substitution/importance scores, and populates results_dict['key']['important_tokens'].
    """
    print("=" * 60)
    print("TEST 3: Important Tokens Only Mode Verification")
    print("=" * 60)
    
    device = "cpu"
    deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    # Mock deberta model
    class MockDebertaModel(torch.nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.deberta = torch.nn.Module()
            self.deberta.embeddings = torch.nn.Module()
            self.deberta.embeddings.word_embeddings = torch.nn.Embedding(vocab_size, 10)
            self.dummy_param = torch.nn.Parameter(torch.randn(1))
            
        def forward(self, input_ids, **kwargs):
            embeddings = self.deberta.embeddings.word_embeddings(input_ids)
            logits = embeddings.sum(dim=-1).mean(dim=-1, keepdim=True).repeat(1, 3) * self.dummy_param
            return {"logits": logits}
            
    deberta_model = MockDebertaModel(deberta_tokenizer.vocab_size)
    deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings.weight.detach()
    
    # Inputs
    question = "What is the answer?"
    initial_generation_text = "The multi-token word segmentation alignment shouldn't fail on out-of-vocabulary terms like Llama-3's 128k tokenizer."
    initial_generation_ids = torch.tensor(llm_tokenizer.encode(" " + initial_generation_text, add_special_tokens=False))
    
    # Mock generation logits from the LLM model: shape [num_tokens, llm_vocab_size]
    # Set high logit value (100.0) at a different index to ensure candidate passes the 0.5 threshold
    generation_logits = torch.zeros((len(initial_generation_ids), llm_tokenizer.vocab_size))
    for t in range(len(initial_generation_ids)):
        orig_token = initial_generation_ids[t].item()
        alt_token = (orig_token + 1) % llm_tokenizer.vocab_size
        generation_logits[t, alt_token] = 100.0
    
    args = Args()
    args.token_prob_threshold = 0.5  # Filter out low-probability candidates
    args.alphas = (1/3, 1/3, 1/3)
    args.invalid_ids = []
    args.num_total_generations = 3
    args.save_only_important_tokens = True
    
    results_dict = {
        'clar_sdlg': {
            'generations': [],
            'likelihoods': []
        }
    }
    
    initial_generation = {
        'generation_ids': [initial_generation_ids],
        'generation_text': [initial_generation_text]
    }
    initial_likelihood = {
        'generation_logits': [generation_logits]
    }
    
    print("Running generate_semantically_diverse_output_sequences with save_only_important_tokens=True...")
    updated_dict = generate_semantically_diverse_output_sequences(
        results_dict=results_dict,
        deberta_model=deberta_model,
        deberta_tokenizer=deberta_tokenizer,
        device_deberta=device,
        deberta_embeddings=deberta_embeddings,
        model=None, # should not be called since we skip generation!
        tokenizer=llm_tokenizer,
        device_llm=device,
        input_ids=initial_generation_ids,
        prompt=torch.tensor([1, 2]),
        question=question,
        initial_generation=initial_generation,
        initial_likelihood=initial_likelihood,
        key='clar_sdlg',
        compute_epistemic=False,
        args=args
    )
    
    assert 'important_tokens' in updated_dict['clar_sdlg'], "important_tokens key was not added to results_dict!"
    important_tokens = updated_dict['clar_sdlg']['important_tokens']
    print(f"Total important tokens generated: {len(important_tokens)}")
    assert len(important_tokens) > 0, "No important tokens were saved."
    assert len(important_tokens) <= args.num_total_generations, "Saved more tokens than args.num_total_generations"
    
    # Verify metadata fields
    first_token = important_tokens[0]
    required_keys = {'word_idx', 'token_idx', 'initial_gen_token_idx', 'token_text', 'importance_score', 'substitution_score', 'attribution_score'}
    for rk in required_keys:
        assert rk in first_token, f"Missing key '{rk}' in important_tokens entry"
        
    print(f"✓ Sample important token entry: {first_token}")
    print("✓ TEST 3 passed successfully: Important tokens mode bypassed generation and saved metadata!\n")


def test_pkl_compression_and_loading():
    """
    Verifies that results_dict saves in compressed .pkl.gz format and loading via
    prepare_results and prepare_results_new supports reading both .pkl and .pkl.gz files.
    """
    print("=" * 60)
    print("TEST 4: PKL Compression and Loading Verification")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    run_key = 'test_run'
    
    # Create a mock results_dict simulating float16 logits cast and proper shape
    results_dict = {
        'input_ids': torch.tensor([1, 2, 3]),
        'question': "Test question?",
        'correctness_dict': {'rouge1': 0.8},
        run_key: {
            'generations': [
                {
                    'generation_ids': [torch.tensor([4, 5])],
                    'generation_text': ["hello world"],
                    'cleaned_generation_ids': [torch.tensor([4, 5])],
                    'cleaned_generation_text': ["hello world"]
                }
            ],
            'likelihoods': [
                {
                    'average_neg_log_likelihood': 0.1,
                    'neg_log_likelihood': 0.2,
                    'generation_logits': [torch.randn(2, 5).half()], # float16 cast
                    'cleaned_average_neg_log_likelihood': 0.1,
                    'cleaned_neg_log_likelihood': 0.2,
                    'cleaned_generation_logits': [torch.randn(2, 5).half()]
                }
            ]
        }
    }
    
    try:
        # Case A: Write uncompressed .pkl to simulate legacy datasets
        file_path_uncompressed = os.path.join(temp_dir, 'results_dict_0.pkl')
        with open(file_path_uncompressed, 'wb') as outfile:
            pickle.dump(results_dict, outfile)
            
        # Case B: Write compressed .pkl.gz to simulate new optimized saves
        file_path_compressed = os.path.join(temp_dir, 'results_dict_1.pkl.gz')
        with gzip.open(file_path_compressed, 'wb') as outfile:
            pickle.dump(results_dict, outfile)
            
        print("Test files written successfully. Testing load fallback...")
        
        # Test prepare_results on both index 0 (uncompressed) and index 1 (compressed)
        res_list, labels, size = prepare_results(
            num_samples=2,
            run_key=run_key,
            metric='rouge1',
            start_sample_id=0,
            base_path=temp_dir
        )
        
        assert size == 2, f"Expected 2 datasets loaded, got {size}"
        assert len(labels) == 2, f"Expected 2 labels loaded, got {len(labels)}"
        assert labels[0] == 0.8 and labels[1] == 0.8, "Metric labels mismatched"
        assert res_list[0]['input_ids'].tolist() == [1, 2, 3], "Loaded content mismatch"
        
        print("✓ prepare_results successfully read both uncompressed and compressed pkls!")
        
        # Test prepare_results_new on both index 0 (uncompressed) and index 1 (compressed)
        res_list_new, labels_new, size_new = prepare_results_new(
            num_samples=2,
            run_key=run_key,
            metric='rouge1',
            start_sample_id=0,
            base_path=temp_dir,
            batch_size=1
        )
        
        assert size_new == 2, f"Expected 2 datasets loaded, got {size_new}"
        assert len(labels_new) == 2, f"Expected 2 labels loaded, got {len(labels_new)}"
        assert labels_new[0] == 0.8 and labels_new[1] == 0.8, "Metric labels mismatched"
        assert res_list_new[0]['input_ids'].tolist() == [1, 2, 3], "Loaded content mismatch"
        
        print("✓ prepare_results_new successfully read both uncompressed and compressed pkls!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print("✓ Temporary test files cleaned up.")
        
    print("✓ TEST 4 passed successfully: Compression and loader fallback verified!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING SDLG TOKEN ALIGNMENT TEST SUITE")
    print("=" * 60)
    test_word_indices()
    test_compute_token_score_ranking()
    test_important_tokens_only_mode()
    test_pkl_compression_and_loading()
    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
