import os
import sys
import glob
import gzip
import pickle
import gc
import re
from tqdm import tqdm

# Ensure TensorFlow pre-allocation safety flags
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass


def split_reference_sentences(paragraph):
    """
    Decomposes a long-form reference paragraph into constituent factual sentences.
    Preserves abbreviations (e.g. H. H. Asquith, U.S., 3 A.M.) while splitting at genuine sentence boundaries.
    """
    if not paragraph or len(str(paragraph).split()) < 12:
        return [str(paragraph).strip()] if str(paragraph).strip() else []

    text = str(paragraph).strip()

    # Try NLTK punkt tokenizer if available
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        from nltk.tokenize import sent_tokenize
        nltk_sents = sent_tokenize(text)
        if len(nltk_sents) > 1:
            clean = [s.strip() for s in nltk_sents if len(s.strip().split()) >= 4]
            if clean:
                return clean
    except Exception:
        pass

    # High-accuracy regex fallback:
    # Match sentence punctuation followed by whitespace and capital letter/number/quote,
    # with lookbehind guards against initials, decimals, and common abbreviations.
    pattern = r'(?<!\b[A-Z]\.)(?<!\b[A-Z])(?<!\bU\.S)(?<!\bA\.M)(?<!\bP\.M)(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)(?<!\b\d)(?<=[.!?])\s+(?=[A-Z0-9"\'])'
    parts = re.split(pattern, text)
    clean_sents = [p.strip() for p in parts if len(p.strip().split()) >= 4]
    return clean_sents if clean_sents else [text]


def run_batched_bleurt_scoring(base_path, batch_size=64, force_rescore=False):
    """
    Performs fast, vectorized batched BLEURT evaluation comparing each sample's
    greedy generation against all valid reference answers (e.g. choice text and full solution).
    Takes the maximum BLEURT score across reference candidates.
    Updates each results_dict_*.pkl.gz file in-place with the computed score.
    """
    import evaluate

    print("\n" + "=" * 60)
    print("STARTING FAST BATCHED BLEURT SCORING")
    print(f"Target Directory: {base_path}")
    print(f"Batch Size:       {batch_size}")
    print(f"Force Rescore:    {force_rescore}")
    print("=" * 60)

    # 1. Discover all results_dict files
    files_gz = glob.glob(os.path.join(base_path, "results_dict_*.pkl.gz"))
    files_raw = glob.glob(os.path.join(base_path, "results_dict_*.pkl"))
    
    # Filter out semantic matrices or other helper files
    valid_files = [
        f for f in (files_gz + files_raw)
        if os.path.basename(f).startswith("results_dict_")
        and not os.path.basename(f).startswith("results_dict_semantic_matrices_")
    ]
    
    def _extract_idx(p):
        base = os.path.basename(p)
        clean = base.replace("results_dict_", "").split(".")[0]
        try:
            return int(clean)
        except ValueError:
            return 999999

    valid_files = sorted(valid_files, key=_extract_idx)

    if not valid_files:
        print(f"[WARNING] No result files found in {base_path} for BLEURT scoring.")
        return {}

    # Lazy-load ScienceQA reference solutions if needed (for backwards compatibility with runs missing 'solution')
    scienceqa_ds = None
    if 'science' in base_path.lower():
        try:
            from datasets import load_dataset
            scienceqa_ds = load_dataset('tasksource/ScienceQA_text_only', split='test')
            print(f"[INFO] Loaded ScienceQA reference dataset ({len(scienceqa_ds)} instances) for solution retrieval.")
        except Exception:
            try:
                from datasets import load_dataset
                scienceqa_ds = load_dataset('allenai/sciq', split='test')
            except Exception:
                scienceqa_ds = None

    # Lazy-load OpenBookQA reference facts if needed (for backwards compatibility with runs missing 'fact1')
    openbookqa_ds = None
    if any(k in base_path.lower() for k in ['openbook', 'obqa']):
        try:
            from datasets import load_dataset
            openbookqa_ds = load_dataset('allenai/openbookqa', 'additional', split='test')
            print(f"[INFO] Loaded OpenBookQA reference dataset ({len(openbookqa_ds)} instances) for fact1 retrieval.")
        except Exception:
            try:
                from datasets import load_dataset
                openbookqa_ds = load_dataset('openbookqa', 'additional', split='test')
            except Exception:
                openbookqa_ds = None

    # 2. Index samples and check which need BLEURT scoring
    loaded_samples = []
    pairs_to_score = []

    for fpath in tqdm(valid_files, desc="Indexing samples for BLEURT"):
        try:
            is_gz = fpath.endswith('.gz')
            if is_gz:
                with gzip.open(fpath, 'rb') as f:
                    res = pickle.load(f)
            else:
                with open(fpath, 'rb') as f:
                    res = pickle.load(f)

            gen_text = ""
            if 'sdlg' in res and 'generations' in res['sdlg'] and len(res['sdlg']['generations']) > 0:
                gen_text = res['sdlg']['generations'][0].get('generation_text', [''])[0]
            elif 'generated_text' in res:
                gen_text = res['generated_text']

            # Collect all candidate reference answers (both choice text and full solution)
            ref_answers = list(res.get('correct_answers', []))
            best_ans = res.get('best_answer', '')
            if best_ans and best_ans not in ref_answers:
                ref_answers.append(best_ans)

            s_idx = _extract_idx(fpath)

            # Dynamically retrieve ScienceQA full solution if missing
            if scienceqa_ds is not None:
                q_id = res.get('question_id', s_idx)
                if isinstance(q_id, int) and 0 <= q_id < len(scienceqa_ds):
                    sq_sample = scienceqa_ds[q_id]
                    sol = sq_sample.get('solution', '')
                    if sol and str(sol).strip() and str(sol).strip() not in ref_answers:
                        ref_answers.append(str(sol).strip())
                    choices = sq_sample.get('choices', [])
                    ans_idx = sq_sample.get('answer', -1)
                    if choices and 0 <= ans_idx < len(choices):
                        choice_text = str(choices[ans_idx]).strip()
                        if choice_text and choice_text not in ref_answers:
                            ref_answers.append(choice_text)

            # Dynamically retrieve OpenBookQA fact1 if missing
            if openbookqa_ds is not None:
                q_id = res.get('question_id', s_idx)
                if isinstance(q_id, int) and 0 <= q_id < len(openbookqa_ds):
                    ob_sample = openbookqa_ds[q_id]
                    fact1 = ob_sample.get('fact1', '')
                    if fact1 and str(fact1).strip() and str(fact1).strip() not in ref_answers:
                        ref_answers.append(str(fact1).strip())
                    choices_dict = ob_sample.get('choices', {})
                    choices_text = choices_dict.get('text', [])
                    choices_labels = choices_dict.get('label', [])
                    ans_key = ob_sample.get('answerKey', '')
                    if ans_key in choices_labels:
                        c_idx = choices_labels.index(ans_key)
                        if 0 <= c_idx < len(choices_text):
                            choice_text = str(choices_text[c_idx]).strip()
                            if choice_text and choice_text not in ref_answers:
                                ref_answers.append(choice_text)

            # Sentence decomposition for long-form references (ASQA / AmbigQA)
            # Allows single-sentence model predictions to be scored symmetrically
            # against individual factual sentences without length asymmetry penalties.
            is_long_form_ds = any(k in base_path.lower() for k in ['asqa', 'ambig']) and any(len(str(r).split()) > 20 for r in ref_answers)
            if is_long_form_ds:
                expanded_refs = list(ref_answers)
                for r in ref_answers:
                    sents = split_reference_sentences(r)
                    for s in sents:
                        if s not in expanded_refs and len(s.split()) >= 4:
                            expanded_refs.append(s)
                ref_answers = expanded_refs

            clean_refs = [str(r).strip() for r in ref_answers if str(r).strip()]
            if not clean_refs:
                clean_refs = [""]

            c_dict = res.get('correctness_dict', {})

            loaded_samples.append({
                'fpath': fpath,
                'is_gz': is_gz,
                'res': res,
                'gen_text': str(gen_text).strip(),
                'ref_answers': clean_refs,
                'best_answer': clean_refs[0],
                's_idx': s_idx,
                'c_dict': c_dict
            })

            # Check if BLEURT needs evaluation
            needs_eval = force_rescore or ('bleurt' not in c_dict) or (c_dict['bleurt'] is None)
            # Auto-detect collapsed score on ScienceQA / OpenBookQA (< 0.0 with multiple references): rescore
            if not needs_eval and any(k in base_path.lower() for k in ['science', 'openbook', 'obqa']) and c_dict.get('bleurt', 0.0) < 0.0 and len(clean_refs) > 1:
                needs_eval = True

            if needs_eval:
                pairs_to_score.append((len(loaded_samples) - 1, str(gen_text).strip(), clean_refs))

        except Exception as e:
            print(f"[WARNING] Error reading {fpath}: {e}")

    print(f"Indexed {len(loaded_samples)} samples. {len(pairs_to_score)} samples need BLEURT evaluation.")

    # 3. Compute batched BLEURT if pairs exist
    scores_map = {}
    if pairs_to_score:
        try:
            print("Loading permanent BLEURT standard: evaluate.load('bleurt', 'bleurt-base-128')...")
            bleurt = evaluate.load("bleurt", "bleurt-base-128")
        except Exception as e:
            print(f"[WARNING] Could not load bleurt-base-128 ({e}). Falling back to default bleurt...")
            try:
                bleurt = evaluate.load("bleurt")
            except Exception as e2:
                print(f"[ERROR] Failed to load BLEURT metric: {e2}")
                bleurt = None

        if bleurt is not None:
            flat_preds = []
            flat_refs = []
            sample_mapping = []  # list of (sample_pos_idx, start_pos, end_pos)

            for sample_pos_idx, gen_str, refs_list in pairs_to_score:
                start_p = len(flat_preds)
                for r in refs_list:
                    flat_preds.append(gen_str)
                    flat_refs.append(r)
                end_p = len(flat_preds)
                sample_mapping.append((sample_pos_idx, start_p, end_p))

            total_pairs = len(flat_preds)
            print(f"Running vectorized BLEURT inference across {total_pairs} pairs...")

            computed_scores = []
            for b_start in tqdm(range(0, total_pairs, batch_size), desc="BLEURT Batched Inference"):
                b_end = min(b_start + batch_size, total_pairs)
                chunk_preds = flat_preds[b_start:b_end]
                chunk_refs = flat_refs[b_start:b_end]
                try:
                    res = bleurt.compute(predictions=chunk_preds, references=chunk_refs)
                    computed_scores.extend(res['scores'])
                except Exception as b_err:
                    print(f"[ERROR] BLEURT batch [{b_start}:{b_end}] failed: {b_err}")
                    computed_scores.extend([0.0] * (b_end - b_start))

            # Map scores back as max across reference answers per sample
            for sample_pos_idx, start_p, end_p in sample_mapping:
                if end_p > start_p and end_p <= len(computed_scores):
                    scores_map[sample_pos_idx] = float(max(computed_scores[start_p:end_p]))
                else:
                    scores_map[sample_pos_idx] = 0.0

            # Free BLEURT model from memory
            del bleurt
            gc.collect()

    # 4. Update files in-place with computed BLEURT score
    updated_count = 0
    final_scores_by_idx = {}

    for pos, sample in enumerate(tqdm(loaded_samples, desc="Writing BLEURT scores to sample files")):
        res = sample['res']
        c_dict = sample['c_dict']
        s_idx = sample['s_idx']

        if pos in scores_map:
            c_dict['bleurt'] = scores_map[pos]
            res['correctness_dict'] = c_dict
            res['correct_answers'] = sample['ref_answers']

            if sample['is_gz']:
                with gzip.open(sample['fpath'], 'wb') as f:
                    pickle.dump(res, f)
            else:
                with open(sample['fpath'], 'wb') as f:
                    pickle.dump(res, f)
            updated_count += 1

        final_scores_by_idx[s_idx] = c_dict.get('bleurt', 0.0)

    # Print summary statistics
    all_scores_list = list(final_scores_by_idx.values())
    if all_scores_list:
        import numpy as np
        arr = np.array(all_scores_list)
        corr_0 = np.mean(arr >= 0.0) * 100
        corr_neg35 = np.mean(arr >= -0.35) * 100
        print(f"\n[SUMMARY] BLEURT Score Distribution ({len(arr)} samples):")
        print(f"  - Mean:                {arr.mean():.4f}")
        print(f"  - Min / Max:           {arr.min():.4f} / {arr.max():.4f}")
        print(f"  - Accuracy (>= 0.0):   {corr_0:.1f}%")
        print(f"  - Accuracy (>= -0.35): {corr_neg35:.1f}%")

    print(f"Batched BLEURT scoring complete. Updated {updated_count} sample files.")
    print("=" * 60 + "\n")
    return final_scores_by_idx


if __name__ == '__main__':
    # Standalone execution for re-scoring existing results without re-generating
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, '..'))
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    target_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if target_dir is None:
        cand_dirs = glob.glob(os.path.join(script_dir, 'result', '*'))
        cand_dirs = [d for d in cand_dirs if os.path.isdir(d) and not os.path.basename(d).startswith('entropies')]
        if cand_dirs:
            print("Found result directories:")
            for i, d in enumerate(cand_dirs):
                print(f"  [{i}] {os.path.basename(d)}")
            target_dir = cand_dirs[0]
            print(f"Targeting: {target_dir}")
        else:
            print("Usage: python batch_scoring.py <path_to_result_dir>")
            sys.exit(0)

    scores = run_batched_bleurt_scoring(target_dir, batch_size=64, force_rescore=True)

    # Auto-trigger batch compute to refresh result/entropies_unified/
    try:
        from batch_compute import run_batch_compute
        from args import Args
        a = Args()
        print("\n>>> Auto-refreshing batch compute & entropy pickles...")
        run_batch_compute(args=a, base_path=target_dir)
        print(">>> Batch compute completed successfully.")
    except Exception as e:
        print(f"[WARNING] Could not auto-run batch_compute: {e}")
