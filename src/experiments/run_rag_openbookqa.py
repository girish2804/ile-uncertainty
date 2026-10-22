import os
import sys

# TensorFlow safety pre-allocations to prevent OOM conflicts with PyTorch
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gc
import re
import glob
import gzip
import math
import json
import pickle
import zipfile
import io
import collections
import urllib.request
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
import evaluate

# Dynamic path resolution to root, src, and experiments
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, ".."))
experiments_dir = os.path.abspath(os.path.join(src_dir, 'experiments'))
for p in [script_dir, src_dir, experiments_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from openbookqa_rag_args import OpenBookQARAGArgs
from disagree import calculate_all_disagreement_metrics
from utils import seed_everything


# ==============================================================================
# Dense Semantic Retrieval Engine (Fix 4: all-MiniLM-L6-v2 Semantic Embeddings)
# ==============================================================================

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class DenseRetriever:
    """
    Lightweight, high-accuracy dense semantic retriever over OpenBookQA facts.
    Uses all-MiniLM-L6-v2 with mean pooling and L2 normalization (cosine similarity).
    Operates without extra C-dependencies, utilizing standard transformers.AutoModel.
    """
    def __init__(self, corpus, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu", hf_token=None):
        self.raw_corpus = [c.strip().strip('"').strip("'") for c in corpus if c.strip()]
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModel.from_pretrained(model_name, token=hf_token).to(device)
        self.model.eval()
        self.corpus_embeddings = None

    def encode(self, texts, batch_size=64):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
            with torch.no_grad():
                out = self.model(**encoded)
                embs = _mean_pooling(out, encoded['attention_mask'])
                embs = F.normalize(embs, p=2, dim=1)
            all_embeddings.append(embs.cpu())
        return torch.cat(all_embeddings, dim=0)

    def precompute_corpus(self, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            try:
                self.corpus_embeddings = torch.load(cache_path, map_location="cpu")
                if self.corpus_embeddings.shape[0] == len(self.raw_corpus):
                    return
            except Exception:
                pass
                
        print(f"Pre-encoding {len(self.raw_corpus)} science facts via {self.model_name}...")
        self.corpus_embeddings = self.encode(self.raw_corpus)
        if cache_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                torch.save(self.corpus_embeddings, cache_path)
            except Exception:
                pass

    def retrieve(self, query_text, top_k=3):
        query_emb = self.encode([query_text])  # [1, d] on CPU
        scores = torch.mm(query_emb, self.corpus_embeddings.T)[0]  # Cosine similarity
        top_vals, top_indices = torch.topk(scores, k=top_k)
        
        results = []
        for score_val, idx in zip(top_vals, top_indices):
            results.append({
                'fact': self.raw_corpus[idx.item()],
                'score': float(score_val.item()),
                'corpus_idx': idx.item()
            })
        return results


# ==============================================================================
# BM25 Corpus Indexing & Retrieval Engine (Fallback/Baseline Mode)
# ==============================================================================

class BM25Retriever:
    """
    Fast, in-memory BM25 retrieval engine over OpenBookQA core science facts.
    Uses standard Okapi BM25 formulation (k1=1.5, b=0.75).
    """
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.raw_corpus = [c.strip().strip('"').strip("'") for c in corpus if c.strip()]
        self.docs = [re.findall(r'\w+', doc.lower()) for doc in self.raw_corpus]
        self.doc_lens = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.docs))
        self.k1 = k1
        self.b = b
        
        self.df = collections.defaultdict(int)
        for d in self.docs:
            for term in set(d):
                self.df[term] += 1
                
        n_docs = len(self.docs)
        self.idf = {
            term: math.log((n_docs - df_val + 0.5) / (df_val + 0.5) + 1.0)
            for term, df_val in self.df.items()
        }

    def retrieve(self, query_text, top_k=3):
        query_terms = re.findall(r'\w+', query_text.lower())
        scores = []
        
        for idx, doc_terms in enumerate(self.docs):
            doc_len = self.doc_lens[idx]
            counts = collections.Counter(doc_terms)
            score = 0.0
            
            for term in query_terms:
                if term in counts:
                    freq = counts[term]
                    idf_val = self.idf.get(term, 0.0)
                    num = freq * (self.k1 + 1.0)
                    denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    score += idf_val * (num / denom)
                    
            scores.append((score, idx))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        top_results = [
            {'fact': self.raw_corpus[idx], 'score': float(sc), 'corpus_idx': idx}
            for sc, idx in scores[:top_k]
        ]
        return top_results


class BM25OptimizedRetriever:
    """
    Optimized BM25+ retrieval engine over OpenBookQA core science facts.
    Key Enhancements:
      1. Lexical Normalization: Porter stemming unifies morphological variations across science vocabulary.
      2. Stopword & Template Filtering: Strips interrogatives, filler, and multiple-choice prompt boilerplate.
      3. BM25+ Lower Bound: Incorporates delta offset (delta=0.5, k1=1.5, b=0.75) to prevent over-penalizing long facts.
      4. Query Distillation: Formulates queries strictly on the question stem, avoiding distractor choice corruption.
    """
    def __init__(self, corpus, k1=1.5, b=0.75, delta=0.5):
        self.raw_corpus = [c.strip().strip('"').strip("'") for c in corpus if c.strip()]
        self.k1 = k1
        self.b = b
        self.delta = delta
        
        try:
            from nltk.stem import PorterStemmer
            _stemmer = PorterStemmer()
            self.stem = _stemmer.stem
        except Exception:
            def _basic_stem(w):
                for sfx in ['ing', 'ies', 'es', 's', 'ed', 'ly', 'ment', 'tion', 'ness', 'able', 'al']:
                    if w.endswith(sfx) and len(w) > len(sfx) + 2:
                        return w[:-len(sfx)]
                return w
            self.stem = _basic_stem
            
        self.stopwords = {
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
            'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
            'by', 'can', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
            'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him',
            'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me',
            'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
            'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 'should',
            'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
            'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until',
            'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
            'why', 'will', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves',
            'following', 'likely', 'best', 'example', 'statement', 'one', 'two', 'describe', 'describes',
            'means', 'true', 'false', 'called', 'cause', 'causes', 'caused', 'result', 'results'
        }
        
        self.docs = [self._tokenize_and_stem(doc) for doc in self.raw_corpus]
        self.doc_lens = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.docs))
        
        self.df = collections.defaultdict(int)
        for d in self.docs:
            for term in set(d):
                self.df[term] += 1
                
        n_docs = len(self.docs)
        self.idf = {
            term: math.log((n_docs - df_val + 0.5) / (df_val + 0.5) + 1.0)
            for term, df_val in self.df.items()
        }

    def _tokenize_and_stem(self, text):
        tokens = re.findall(r'\w+', text.lower())
        return [self.stem(t) for t in tokens if t not in self.stopwords]

    def retrieve(self, query_text, top_k=3):
        query_terms = self._tokenize_and_stem(query_text)
        if not query_terms:
            query_terms = re.findall(r'\w+', query_text.lower())
            
        scores = []
        for idx, doc_terms in enumerate(self.docs):
            doc_len = self.doc_lens[idx]
            counts = collections.Counter(doc_terms)
            score = 0.0
            
            for term in query_terms:
                if term in counts:
                    freq = counts[term]
                    idf_val = self.idf.get(term, 0.0)
                    num = freq * (self.k1 + 1.0)
                    denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    score += idf_val * ((num / denom) + self.delta)
                    
            scores.append((score, idx))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        top_results = [
            {'fact': self.raw_corpus[idx], 'score': float(sc), 'corpus_idx': idx}
            for sc, idx in scores[:top_k]
        ]
        return top_results


def load_or_fetch_openbook_corpus(corpus_path, dataset=None):
    """
    Ensures the 1,326 OpenBookQA core facts exist on disk.
    """
    if os.path.exists(corpus_path):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            facts = [line.strip().strip('"').strip("'") for line in f if line.strip()]
        if len(facts) >= 1300:
            return facts
            
    print(f"\nCorpus file not found at '{corpus_path}'. Fetching official OpenBookQA facts...")
    os.makedirs(os.path.dirname(os.path.abspath(corpus_path)), exist_ok=True)
    
    try:
        url = "https://s3-us-west-2.amazonaws.com/ai2-website/data/OpenBookQA-V1-Sep2018.zip"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as res:
            with zipfile.ZipFile(io.BytesIO(res.read())) as z:
                txt = z.read('OpenBookQA-V1-Sep2018/Data/Main/openbook.txt').decode('utf-8')
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write(txt)
        facts = [l.strip().strip('"').strip("'") for l in txt.strip().split('\n') if l.strip()]
        print(f"Successfully downloaded and indexed {len(facts)} official OpenBookQA science facts!")
        return facts
    except Exception as e:
        print(f"Direct S3 download failed ({e}). Falling back to dataset fact extraction...")

    if dataset is not None:
        extracted = set()
        for split_name in ['train', 'validation', 'test']:
            if split_name in dataset:
                for row in dataset[split_name]:
                    fact = row.get('fact1', '').strip()
                    if fact:
                        extracted.add(fact)
        facts = sorted(list(extracted))
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(facts))
        print(f"Extracted and saved {len(facts)} unique facts from dataset splits to '{corpus_path}'.")
        return facts
        
    raise RuntimeError(f"Could not load or download OpenBookQA facts corpus to '{corpus_path}'.")


# ==============================================================================
# Persistent Retrieval Cache Manager (Ensures Frozen & Instant Retrieval)
# ==============================================================================

def build_or_load_retrieval_cache(args, dataset, test_set, corpus_facts, device):
    """
    Loads pre-cached retrieved facts from disk if available, or executes retrieval
    once across all test samples, persisting to JSON for all subsequent model runs.
    Guarantees zero retriever overhead on subsequent runs and perfectly aligned context.
    """
    cache_file = args.retrieval_cache_path
    
    if os.path.exists(cache_file) and not getattr(args, 'force_reretrieve', False):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            if len(cached_data) >= len(test_set):
                print("\n" + "=" * 85)
                print(f"[CACHE HIT] Loaded {len(cached_data)} pre-retrieved facts from '{cache_file}'.")
                print("Skipping retriever model loading entirely! (Zero VRAM and zero latency overhead)")
                print("=" * 85)
                return cached_data
        except Exception as e:
            print(f"[CACHE ERROR] Failed reading existing cache ({e}). Re-computing retrieval...")

    # Fast-path derivation for Dense Top-1 from existing Dense Top-3 cache
    if args.retrieval_mode == 'dense' and args.top_k_retrieval == 1:
        top3_cache = os.path.join(os.path.dirname(os.path.abspath(cache_file)), 'openbook_retrieved_dense_top3.json')
        if os.path.exists(top3_cache):
            try:
                print(f"[FAST-PATH] Slicing Top-1 cache directly from '{top3_cache}'...")
                with open(top3_cache, 'r', encoding='utf-8') as f:
                    c3 = json.load(f)
                sliced_cache = {}
                hits = 0
                for idx_str, item in c3.items():
                    passages = item.get('retrieved_passages', [])[:1]
                    details = item.get('retrieved_details', [])[:1]
                    sample_idx = int(idx_str)
                    oracle_fact = test_set[sample_idx].get('fact1', '').strip().lower()
                    h1 = int(any(oracle_fact in p.lower() or p.lower() in oracle_fact for p in passages))
                    if h1:
                        hits += 1
                    sliced_cache[idx_str] = {
                        'retrieved_passages': passages,
                        'retrieved_details': details,
                        'retriever_hit_at_k': h1
                    }
                os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(sliced_cache, f, indent=2)
                print(f"[FAST-PATH COMPLETE] Created '{cache_file}' with Hit@1 = {hits}/{len(sliced_cache)} ({hits/len(sliced_cache)*100:.2f}%)!")
                return sliced_cache
            except Exception as e:
                print(f"[FAST-PATH WARNING] Slicing failed ({e}), falling back to direct computation...")

    print("\n" + "=" * 85)
    print(f"[CACHE MISS] Executing one-time {args.retrieval_mode.upper()} retrieval pass across all {len(test_set)} queries...")
    print(f"Results will be permanently cached to '{cache_file}' for all future model runs.")
    print("=" * 85)
    
    if args.retrieval_mode == 'dense':
        retriever = DenseRetriever(
            corpus_facts,
            model_name=args.dense_model_name,
            device=device,
            hf_token=args.hf_token
        )
        emb_cache = os.path.join(os.path.dirname(os.path.abspath(args.corpus_path)), 'openbook_dense_embeddings_minilm.pt')
        retriever.precompute_corpus(cache_path=emb_cache)
    elif args.retrieval_mode in ['bm25_opt', 'bm25+']:
        retriever = BM25OptimizedRetriever(
            corpus_facts,
            k1=getattr(args, 'bm25_k1', 1.5),
            b=getattr(args, 'bm25_b', 0.75),
            delta=getattr(args, 'bm25_delta', 0.5)
        )
    elif args.retrieval_mode == 'bm25':
        retriever = BM25Retriever(corpus_facts, k1=args.bm25_k1, b=args.bm25_b)
    else:
        retriever = None

    retrieval_cache = {}
    for idx, sample in enumerate(tqdm(test_set, desc=f"Building {args.retrieval_mode.upper()} Retrieval Cache")):
        question = sample['question_stem']
        choices_text = sample.get('choices', {}).get('text', [])
        formatted_choices = ", ".join(choices_text) if choices_text else ""
        oracle_fact1 = sample.get('fact1', '')
        
        if args.retrieval_mode in ['bm25_opt', 'bm25+']:
            query_text = question
            retrieved_results = retriever.retrieve(query_text, top_k=args.top_k_retrieval)
            retrieved_passages = [r['fact'] for r in retrieved_results]
            
            oracle_norm = oracle_fact1.lower().strip().strip('"').strip("'")
            hit_at_k = int(any(oracle_norm in p.lower() or p.lower() in oracle_norm for p in retrieved_passages))
        elif args.retrieval_mode in ['dense', 'bm25']:
            query_text = f"{question} {formatted_choices}"
            retrieved_results = retriever.retrieve(query_text, top_k=args.top_k_retrieval)
            retrieved_passages = [r['fact'] for r in retrieved_results]
            
            oracle_norm = oracle_fact1.lower().strip().strip('"').strip("'")
            hit_at_k = int(any(oracle_norm in p.lower() or p.lower() in oracle_norm for p in retrieved_passages))
        elif args.retrieval_mode in ['distractor', 'random_distractor', 'random']:
            import random
            rng = random.Random(42 + idx)
            oracle_norm = oracle_fact1.lower().strip().strip('"').strip("'")
            cand_indices = [ci for ci, f in enumerate(corpus_facts) if oracle_norm not in f.lower() and f.lower() not in oracle_norm]
            chosen_idx = rng.choice(cand_indices) if cand_indices else (idx % len(corpus_facts))
            distractor_fact = corpus_facts[chosen_idx].strip().strip('"').strip("'")
            retrieved_results = [{'fact': distractor_fact, 'score': 0.0, 'corpus_idx': chosen_idx}]
            retrieved_passages = [distractor_fact]
            hit_at_k = 0
        else:
            retrieved_results = [{'fact': oracle_fact1, 'score': 1.0, 'corpus_idx': -1}]
            retrieved_passages = [oracle_fact1]
            hit_at_k = 1
            
        retrieval_cache[str(idx)] = {
            'retrieved_passages': retrieved_passages,
            'retrieved_details': retrieved_results,
            'retriever_hit_at_k': hit_at_k
        }

    # Save to disk
    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(retrieval_cache, f, indent=2)
        
    hit_rate = sum(item['retriever_hit_at_k'] for item in retrieval_cache.values()) / max(1, len(retrieval_cache))
    print(f"Successfully cached all {len(retrieval_cache)} retrieved queries to '{cache_file}' (Hit@{args.top_k_retrieval} = {hit_rate*100:.2f}%)!")
    
    # Completely free retriever from memory
    if 'retriever' in locals() and retriever is not None:
        del retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return retrieval_cache


# ==============================================================================
# In-situ Hidden State Uncertainty Probing Engine (Batched Stacking for ALL Layers)
# ==============================================================================

def extract_in_situ_uncertainties(model, lm_head, input_ids, args):
    """
    Performs single-pass hidden state extraction at prompt terminal position
    and calculates layer-wise ensembling entropies (H_final, H_mix for all k depths).
    Stack all layers retaining 100% of layer depth with 1 single batched kernel launch.
    """
    with torch.no_grad():
        model_outputs = model(input_ids, return_dict=True, output_hidden_states=True)
        final_logits = model_outputs.logits[:, -1, :].float()
        total_layers = len(model_outputs.hidden_states) - 1
        
        # Stack all layers (retaining 100% of layer depth for EU)
        target_dev = getattr(lm_head, 'weight', None).device if (hasattr(lm_head, 'weight') and hasattr(lm_head.weight, 'device')) else final_logits.device
        hs_stack = torch.stack([model_outputs.hidden_states[l][:, -1, :].to(target_dev) for l in range(total_layers)], dim=0)
        all_logits = lm_head(hs_stack) # 1 single batched kernel launch for all layers
        all_premature_logits = {l: all_logits[l].to(final_logits.device) for l in range(total_layers)}
            
        metrics_by_depth = {}
        for k in range(1, total_layers):
            cand_layers = list(range(total_layers - k, total_layers))
            cand_logits = {l: all_premature_logits[l] for l in cand_layers}
            m_res = calculate_all_disagreement_metrics(
                final_logits,
                cand_logits,
                relative_top=args.relative_top,
                min_tokens_to_keep=args.min_tokens_to_keep,
                skip_top_neighbors=args.skip_top_neighbors
            )
            metrics_by_depth[f'entropy_mixture_top_{k}'] = float(m_res.get('entropy_mixture', 0.0))
            metrics_by_depth[f'mutual_info_top_{k}'] = float(m_res.get('mutual_info', 0.0))
            metrics_by_depth[f'gated_mutual_info_top_{k}'] = float(m_res.get('gated_mutual_info', 0.0))
            
        metrics_by_depth['entropy_final'] = float(m_res.get('entropy_final', 0.0))
        metrics_by_depth['total_layers'] = total_layers
        
    return metrics_by_depth


# ==============================================================================
# Main Paired RAG Experiment Pipeline
# ==============================================================================

def run_experiment(args=None):
    if args is None:
        args = OpenBookQARAGArgs()
        
    if getattr(args, 'retrieval_mode', '') == 'paired':
        import copy
        print("=" * 85)
        print(f"RUNNING PAIRED OPENBOOKQA ABLATIONS FOR MODEL: {args.llm_model}")
        print("Condition 1: BM25-Optimized (Top-3 Facts, Query Distillation)")
        print("Condition 2: Dense Top-1 (Cleanest Semantic Fact, Zero Distractors)")
        print("=" * 85)
        
        # 1. BM25-Optimized
        print(f"\n{'#' * 85}")
        print(f"# [1/2] RUNNING BM25-OPTIMIZED ON: {args.llm_model}")
        print(f"{'#' * 85}")
        bm25_args = copy.deepcopy(args)
        bm25_args.retrieval_mode = 'bm25_opt'
        bm25_args.top_k_retrieval = 3
        run_experiment(bm25_args)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 2. Dense Top-1
        print(f"\n{'#' * 85}")
        print(f"# [2/2] RUNNING DENSE TOP-1 ON: {args.llm_model}")
        print(f"{'#' * 85}")
        top1_args = copy.deepcopy(args)
        top1_args.retrieval_mode = 'dense'
        top1_args.top_k_retrieval = 1
        run_experiment(top1_args)
        
        print("\n" + "=" * 85)
        print(f"ALL PAIRED ABLATIONS FOR {args.llm_model} COMPLETED SUCCESSFULLY!")
        print("=" * 85)
        return
        
    seed_everything(args.seed_value)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model name sanitization and dedicated versioned directory
    model_name_clean = args.llm_model.split("/")[-1].lower()
    model_name_clean = model_name_clean.replace("-instruct", "").replace("-chat", "").replace("_", "-")
    run_id = f"{args.output_dir_prefix}_{model_name_clean}"
    
    canonical_result_dir = os.path.join(experiments_dir, 'result')
    base_path = os.path.join(canonical_result_dir, run_id)
    os.makedirs(base_path, exist_ok=True)
    args.args_to_yaml(base_path)
    
    print("=" * 85)
    print("STARTING OPENBOOKQA EMPIRICAL RAG ROUTING EVALUATION")
    print(f"Run ID:            {run_id}")
    print(f"Model:             {args.llm_model}")
    print(f"Retrieval Mode:    {args.retrieval_mode.upper()} (Top-{args.top_k_retrieval} Facts)")
    if args.retrieval_mode == 'dense':
        print(f"Dense Retriever:   {args.dense_model_name}")
    print(f"Target Directory:  {base_path}")
    print(f"Device:            {device}")
    print("=" * 85)
    
    # 1. Load Dataset
    print("\nLoading OpenBookQA dataset ('additional' config)...")
    try:
        dataset = load_dataset('allenai/openbookqa', 'additional')
    except Exception:
        dataset = load_dataset('openbookqa', 'additional')
    test_set = dataset['test']
    total_samples = len(test_set)
    print(f"Dataset loaded. Total test instances: {total_samples}")
    
    # 2. Build or Load Persistent Retrieval Cache
    corpus_facts = load_or_fetch_openbook_corpus(args.corpus_path, dataset=dataset)
    retrieval_cache = build_or_load_retrieval_cache(args, dataset, test_set, corpus_facts, device)
    
    start_idx = args.start_idx
    num_samples = args.num_samples if args.num_samples != -1 else (total_samples - start_idx)
    end_idx = min(start_idx + num_samples, total_samples)
    print(f"Evaluating slice: indices {start_idx} to {end_idx} ({end_idx - start_idx} instances)")
    
    # 3. Check for Resume Checkpoint
    existing_files = glob.glob(os.path.join(base_path, 'results_dict_*.pkl*'))
    max_existing_idx = -1
    for f in existing_files:
        basename = os.path.basename(f)
        try:
            val = int(basename.replace('results_dict_', '').split('.')[0])
            if val > max_existing_idx:
                max_existing_idx = val
        except ValueError:
            continue
            
    resume_idx = start_idx
    if max_existing_idx >= start_idx and not args.force_overwrite:
        resume_idx = max_existing_idx + 1
        print(f"Found existing progress. Resuming from index {resume_idx}...")
        
    if resume_idx >= end_idx:
        print("All requested samples have already been generated. Proceeding directly to scoring.")
        args.skip_generation = True
        
    # 4. Model & Generation Setup
    if not args.skip_generation:
        print(f"\nLoading causal LM: {args.llm_model} in bfloat16 precision...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model,
            token=args.hf_token,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            token=args.hf_token,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        model.eval()
        lm_head = model.get_output_embeddings()
        
        # Configure EOS IDs
        eos_ids = []
        if model.config.eos_token_id is not None:
            if isinstance(model.config.eos_token_id, list):
                eos_ids.extend(model.config.eos_token_id)
            else:
                eos_ids.append(model.config.eos_token_id)
        for token_str in ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|end|>", "\n\n"]:
            try:
                enc_ids = tokenizer.encode(token_str, add_special_tokens=False)
                for tid in enc_ids:
                    if tid not in eos_ids:
                        eos_ids.append(tid)
            except Exception:
                pass
        eos_ids = list(set(eos_ids))
        
        rouge_metric = evaluate.load('rouge')
        failed_samples = []
        
        # 5. Paired Generation Loop (Direct Answer vs. Dense RAG Grounded)
        for idx in tqdm(range(resume_idx, end_idx), desc="Running Paired RAG Generations"):
            try:
                sample = test_set[idx]
                question = sample['question_stem']
                choices_dict = sample.get('choices', {})
                choices_text = choices_dict.get('text', [])
                choices_labels = choices_dict.get('label', [])
                ans_key = sample.get('answerKey', '')
                
                ans_idx = choices_labels.index(ans_key) if ans_key in choices_labels else -1
                correct_choice = choices_text[ans_idx] if (0 <= ans_idx < len(choices_text)) else ""
                oracle_fact1 = sample.get('fact1', '')
                
                correct_answers = []
                if correct_choice:
                    correct_answers.append(correct_choice)
                if oracle_fact1 and oracle_fact1 not in correct_answers:
                    correct_answers.append(oracle_fact1)
                if not correct_answers:
                    correct_answers = [ans_key] if ans_key else [""]
                
                incorrect_answers = [c for c_i, c in enumerate(choices_text) if c_i != ans_idx]
                formatted_choices = ", ".join(choices_text) if choices_text else ""
                
                # ------------------------------------------------------------------
                # A. DIRECT ANSWERING (Parametric Baseline - Pre-RAG)
                # ------------------------------------------------------------------
                sys_msg = "You are a helpful assistant. Based on the given choices, answer the question in a single sentence. Do not repeat the prompt."
                direct_user_msg = f"Choices: {formatted_choices}\nQuestion: {question}" if formatted_choices else question
                
                try:
                    direct_prompt = tokenizer.apply_chat_template(
                        [{"role": "system", "content": sys_msg}, {"role": "user", "content": direct_user_msg}],
                        tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    direct_prompt = f"Choices: {formatted_choices}\nQuestion: {question}\nAnswer:"
                    
                direct_inputs = tokenizer(direct_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
                
                with torch.no_grad():
                    gen_direct_out = model.generate(
                        input_ids=direct_inputs["input_ids"],
                        max_new_tokens=args.max_length_of_generated_sequence,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        eos_token_id=eos_ids,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                    )
                
                gen_direct_ids = gen_direct_out[0, direct_inputs["input_ids"].shape[-1]:].cpu()
                gen_direct_text = tokenizer.decode(gen_direct_ids, skip_special_tokens=True).strip()
                direct_layer_metrics = extract_in_situ_uncertainties(model, lm_head, direct_inputs["input_ids"], args)
                
                # ------------------------------------------------------------------
                # B. RETRIEVAL LOOKUP (Instant from Cache) & KNOWLEDGE-AUGMENTED ANSWERING
                # ------------------------------------------------------------------
                cached_retrieval = retrieval_cache.get(str(idx), {})
                retrieved_passages = cached_retrieval.get('retrieved_passages', [oracle_fact1])
                retrieved_results = cached_retrieval.get('retrieved_details', [{'fact': oracle_fact1, 'score': 1.0}])
                hit_at_k = cached_retrieval.get('retriever_hit_at_k', 0)
                
                knowledge_block = "\n".join([f"- {p}" for p in retrieved_passages])
                rag_user_msg = f"Knowledge:\n{knowledge_block}\nChoices: {formatted_choices}\nQuestion: {question}"
                
                try:
                    rag_prompt = tokenizer.apply_chat_template(
                        [{"role": "system", "content": sys_msg}, {"role": "user", "content": rag_user_msg}],
                        tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    rag_prompt = f"{rag_user_msg}\nAnswer:"
                    
                rag_inputs = tokenizer(rag_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
                
                with torch.no_grad():
                    gen_rag_out = model.generate(
                        input_ids=rag_inputs["input_ids"],
                        max_new_tokens=args.max_length_of_generated_sequence,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        eos_token_id=eos_ids,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                    )
                
                gen_rag_ids = gen_rag_out[0, rag_inputs["input_ids"].shape[-1]:].cpu()
                gen_rag_text = tokenizer.decode(gen_rag_ids, skip_special_tokens=True).strip()
                rag_layer_metrics = extract_in_situ_uncertainties(model, lm_head, rag_inputs["input_ids"], args)
                
                # ROUGE-L diagnostics
                rouge_direct = rouge_metric.compute(predictions=[gen_direct_text], references=[correct_answers])['rougeL']
                rouge_rag = rouge_metric.compute(predictions=[gen_rag_text], references=[correct_answers])['rougeL']
                
                tot_l = direct_layer_metrics['total_layers']
                ka_idx = min(2, tot_l - 1)
                ke_idx = min(27, tot_l - 1)
                
                tu_pre = direct_layer_metrics['entropy_final']
                au_pre = direct_layer_metrics.get(f'entropy_mixture_top_{ka_idx}', 0.0)
                eu_pre = tu_pre - direct_layer_metrics.get(f'entropy_mixture_top_{ke_idx}', 0.0)
                
                tu_post = rag_layer_metrics['entropy_final']
                au_post = rag_layer_metrics.get(f'entropy_mixture_top_{ka_idx}', 0.0)
                eu_post = tu_post - rag_layer_metrics.get(f'entropy_mixture_top_{ke_idx}', 0.0)
                
                delta_eu = eu_pre - eu_post
                delta_au = au_pre - au_post
                
                # Assemble sample record (Preserving exact token sequences for frozen replay)
                sample_record = {
                    'idx': idx,
                    'question': question,
                    'choices': choices_text,
                    'oracle_fact1': oracle_fact1,
                    'retrieved_passages': retrieved_passages,
                    'retrieved_details': retrieved_results,
                    'retriever_hit_at_k': hit_at_k,
                    'correct_answers': correct_answers,
                    'incorrect_answers': incorrect_answers,
                    'direct': {
                        'prompt': direct_prompt,
                        'generation_ids': gen_direct_ids.numpy(),
                        'generation_text': gen_direct_text,
                        'rougeL': rouge_direct,
                        'layer_metrics': direct_layer_metrics
                    },
                    'rag': {
                        'prompt': rag_prompt,
                        'generation_ids': gen_rag_ids.numpy(),
                        'generation_text': gen_rag_text,
                        'rougeL': rouge_rag,
                        'layer_metrics': rag_layer_metrics
                    },
                    'delta_eu_estimate': delta_eu
                }
                
                save_file = os.path.join(base_path, f'results_dict_{idx}.pkl.gz')
                with gzip.open(save_file, 'wb') as f_out:
                    pickle.dump(sample_record, f_out)
                    
                # ------------------------------------------------------------------
                # UNABRIDGED STRUCTURED CONSOLE LOGGING (NO TRUNCATION / NO '...')
                # ------------------------------------------------------------------
                hit_str = "YES (Oracle Fact Recovered)" if hit_at_k else "NO (Oracle Fact Absent in Top-K)"
                if args.retrieval_mode == 'dense':
                    engine_tag = f"Dense MiniLM (Top-{args.top_k_retrieval})"
                elif 'opt' in args.retrieval_mode:
                    engine_tag = f"BM25+ Optimized (Top-{args.top_k_retrieval})"
                elif args.retrieval_mode == 'oracle':
                    engine_tag = "Oracle Ground Truth Fact"
                elif args.retrieval_mode in ['distractor', 'random_distractor', 'random']:
                    engine_tag = "Random Distractor Fact (Negative Control)"
                else:
                    engine_tag = f"BM25 (Top-{args.top_k_retrieval})"
                
                tqdm.write("\n" + "=" * 85)
                tqdm.write(f"[SAMPLE #{idx} / Total Test Index: {end_idx - 1}]")
                tqdm.write(f"Question       : {question}")
                if choices_text:
                    tqdm.write(f"Choices        : {', '.join(choices_text)}")
                tqdm.write(f"Reference Ans  : {correct_answers}")
                tqdm.write(f"Oracle Fact    : {oracle_fact1}")
                tqdm.write(f"Retriever Hit  : {hit_str}")
                tqdm.write(f"{engine_tag} Retrieved Facts (Top-{len(retrieved_passages)}):")
                for r_i, r_info in enumerate(retrieved_results, 1):
                    tqdm.write(f"  [{r_i}] (Score: {r_info['score']:.4f}) {r_info['fact']}")
                tqdm.write(f"Direct Answer  : '{gen_direct_text}'  [ROUGE-L = {rouge_direct:.4f}]")
                tqdm.write(f"RAG Answer     : '{gen_rag_text}'  [ROUGE-L = {rouge_rag:.4f}]")
                tqdm.write("Uncertainty Decomposition & Dissipation:")
                tqdm.write(f"  • Pre-RAG  : Total H(p) = {tu_pre:.4f} | Aleatoric H_mix = {au_pre:.4f} | Epistemic EU = {eu_pre:.4f}")
                tqdm.write(f"  • Post-RAG : Total H(p) = {tu_post:.4f} | Aleatoric H_mix = {au_post:.4f} | Epistemic EU = {eu_post:.4f}")
                tqdm.write(f"  • Shifts   : ΔEU = {delta_eu:+.4f} (Epistemic Dissipation) | ΔAU = {delta_au:+.4f}")
                tqdm.write("=" * 85)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                failed_samples.append((idx, str(e)))
                print(f"[ERROR] Sample index {idx} failed: {e}")
                traceback.print_exc()
                if "CUDA error" in str(e) or "out of memory" in str(e):
                    print("Fatal GPU Memory error. Exiting loop.")
                    sys.exit(1)
                    
        # Free LLM weights from VRAM before BLEURT
        print("\nFreeing LLM model from memory before BLEURT scoring pass...")
        del model
        del lm_head
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # ----------------------------------------------------------------------
    # 6. Batched BLEURT Scoring Pass (Standard Checkpoint: bleurt-base-128)
    # ----------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("STARTING BATCHED BLEURT EVALUATION (Checkpoint: bleurt-base-128)")
    print("=" * 85)
    
    bleurt_metric = evaluate.load('bleurt', 'bleurt-base-128')
    
    files_gz = sorted(
        glob.glob(os.path.join(base_path, "results_dict_*.pkl.gz")),
        key=lambda x: int(os.path.basename(x).replace('results_dict_', '').split('.')[0])
    )
    
    if not files_gz:
        print("No raw sample files found for scoring. Exiting.")
        return
        
    print(f"Loaded {len(files_gz)} raw sample pickles. Executing batched scoring...")
    
    summary_rows = []
    batch_size = args.bleurt_batch_size
    
    for b_start in tqdm(range(0, len(files_gz), batch_size), desc="Scoring BLEURT Batches"):
        b_files = files_gz[b_start:b_start + batch_size]
        b_data = []
        direct_pairs = []
        rag_pairs = []
        
        for fp in b_files:
            with gzip.open(fp, 'rb') as f_in:
                item = pickle.load(f_in)
            b_data.append((fp, item))
            
            d_text = item['direct']['generation_text']
            r_text = item['rag']['generation_text']
            refs = item['correct_answers']
            
            for ref in refs:
                direct_pairs.append((d_text, ref))
                rag_pairs.append((r_text, ref))
                
        d_preds = [p[0] for p in direct_pairs]
        d_refs = [p[1] for p in direct_pairs]
        r_preds = [p[0] for p in rag_pairs]
        r_refs = [p[1] for p in rag_pairs]
        
        d_scores = bleurt_metric.compute(predictions=d_preds, references=d_refs)['scores']
        r_scores = bleurt_metric.compute(predictions=r_preds, references=r_refs)['scores']
        
        d_ptr = 0
        r_ptr = 0
        for fp, item in b_data:
            n_refs = len(item['correct_answers'])
            sample_d_scores = d_scores[d_ptr : d_ptr + n_refs]
            sample_r_scores = r_scores[r_ptr : r_ptr + n_refs]
            d_ptr += n_refs
            r_ptr += n_refs
            
            best_d_score = float(max(sample_d_scores)) if sample_d_scores else 0.0
            best_r_score = float(max(sample_r_scores)) if sample_r_scores else 0.0
            
            item['direct']['bleurt'] = best_d_score
            item['direct']['is_correct'] = int(best_d_score >= args.bleurt_threshold)
            
            item['rag']['bleurt'] = best_r_score
            item['rag']['is_correct'] = int(best_r_score >= args.bleurt_threshold)
            
            with gzip.open(fp, 'wb') as f_out:
                pickle.dump(item, f_out)
                
            tot_l = item['direct']['layer_metrics']['total_layers']
            ka_idx = min(2, tot_l - 1)
            ke_idx = min(27, tot_l - 1)
            
            h_fin_pre = item['direct']['layer_metrics']['entropy_final']
            h_mix_ka_pre = item['direct']['layer_metrics'].get(f'entropy_mixture_top_{ka_idx}', 0.0)
            h_mix_ke_pre = item['direct']['layer_metrics'].get(f'entropy_mixture_top_{ke_idx}', 0.0)
            eu_pre = h_fin_pre - h_mix_ke_pre
            au_pre = h_mix_ka_pre
            
            h_fin_post = item['rag']['layer_metrics']['entropy_final']
            h_mix_ka_post = item['rag']['layer_metrics'].get(f'entropy_mixture_top_{ka_idx}', 0.0)
            h_mix_ke_post = item['rag']['layer_metrics'].get(f'entropy_mixture_top_{ke_idx}', 0.0)
            eu_post = h_fin_post - h_mix_ke_post
            au_post = h_mix_ka_post
            
            summary_rows.append({
                'idx': item['idx'],
                'question': item['question'],
                'oracle_fact1': item.get('oracle_fact1', ''),
                'retriever_hit_at_k': item.get('retriever_hit_at_k', 0),
                'top_1_retrieved_fact': item.get('retrieved_passages', [''])[0],
                'direct_answer': item['direct']['generation_text'],
                'rag_answer': item['rag']['generation_text'],
                'bleurt_direct': best_d_score,
                'is_correct_direct': item['direct']['is_correct'],
                'bleurt_rag': best_r_score,
                'is_correct_rag': item['rag']['is_correct'],
                'delta_correct': item['rag']['is_correct'] - item['direct']['is_correct'],
                'h_final_pre': h_fin_pre,
                'eu_pre': eu_pre,
                'au_pre': au_pre,
                'h_final_post': h_fin_post,
                'eu_post': eu_post,
                'au_post': au_post,
                'delta_eu': eu_pre - eu_post,
                'delta_au': au_pre - au_post
            })
            
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(base_path, 'rag_evaluation_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Export standalone portable summary to canonical result directory for zero-friction cross-instance transfer
    parent_result_dir = os.path.dirname(base_path)
    portable_csv_path = os.path.join(parent_result_dir, f"{run_id}_summary.csv")
    summary_df.to_csv(portable_csv_path, index=False)
    
    print("\n" + "=" * 85)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"Total Valid Processed Samples : {len(summary_df)}")
    print(f"Retriever Hit@{args.top_k_retrieval} Rate         : {summary_df['retriever_hit_at_k'].mean() * 100:.2f}%")
    print(f"Baseline Direct Accuracy      : {summary_df['is_correct_direct'].mean() * 100:.2f}%")
    print(f"RAG Accuracy ({args.retrieval_mode.upper()} Top-{args.top_k_retrieval}): {summary_df['is_correct_rag'].mean() * 100:.2f}%")
    print(f"Net Factual Recovery Delta    : {(summary_df['is_correct_rag'].mean() - summary_df['is_correct_direct'].mean()) * 100:+.2f}%")
    print(f"Mean Epistemic Dissipation ΔEU: {summary_df['delta_eu'].mean():+.4f}")
    print(f"Mean Aleatoric Change ΔAU     : {summary_df['delta_au'].mean():+.4f}")
    print(f"Summary Exported To           : {summary_csv_path}")
    print(f"Portable Summary (Standalone) : {portable_csv_path}")
    print("=" * 85)


if __name__ == "__main__":
    run_experiment()
