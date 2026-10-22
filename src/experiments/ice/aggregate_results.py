import os
import glob
import gzip
import pickle
import pandas as pd
from tqdm import tqdm


def aggregate_ice_results(target_run_id=None, *args, **kwargs):
    if target_run_id is None and len(args) > 0:
        target_run_id = args[0]
    ice_dir = os.path.dirname(os.path.abspath(__file__))
    result_parent = os.path.join(ice_dir, 'result')
    
    if not os.path.exists(result_parent):
        print(f"No result directory found at {result_parent}")
        return

    # If no target specified or set to auto, default to active run_id in ice_args.py
    if target_run_id is None or target_run_id == 'auto':
        try:
            from ice_args import ICEArgs
            args = ICEArgs()
            configured_run = getattr(args, 'run_id', None)
            if configured_run and os.path.isdir(os.path.join(result_parent, configured_run)):
                target_run_id = configured_run
                print(f"[*] Targeting active run from ice_args: {target_run_id}")
        except Exception:
            pass

    if target_run_id and target_run_id not in ['--all', '-a', 'all']:
        target_clean = os.path.basename(target_run_id.rstrip("/\\"))
        candidate_dir = os.path.join(result_parent, target_clean)
        if os.path.isdir(candidate_dir):
            runs = [target_clean]
        elif os.path.isdir(target_run_id):
            runs = [os.path.basename(target_run_id)]
            result_parent = os.path.dirname(target_run_id)
        else:
            print(f"Specified run '{target_run_id}' not found in {result_parent}")
            return
    else:
        runs = [d for d in os.listdir(result_parent) if os.path.isdir(os.path.join(result_parent, d))]
        if not runs:
            print(f"No completed runs found inside {result_parent}")
            return

    for run_id in runs:
        run_dir = os.path.join(result_parent, run_id)
        print("-" * 60)
        print(f"Aggregating run: {run_id}")
        
        files = glob.glob(os.path.join(run_dir, "results_dict_*.pkl.gz")) + glob.glob(os.path.join(run_dir, "results_dict_*.pkl"))
        files.sort(key=lambda x: int(os.path.basename(x).replace("results_dict_", "").split(".")[0]))
        
        if not files:
            print(f"No result files found in {run_dir}")
            continue

        aggregated_data = []
        for filepath in tqdm(files, desc="Parsing sample files"):
            try:
                if filepath.endswith('.gz'):
                    with gzip.open(filepath, 'rb') as f:
                        data = pickle.load(f)
                else:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)

                sample_id = data.get('question_id', 0)
                question = data.get('question', '')
                gen_text = data.get('most_likely_generation', {}).get('generation_text', '')
                bleurt_score = data.get('correctness_dict', {}).get('bleurt', 0.0)
                
                ice = data.get('ice', {})
                epistemic = ice.get('epistemic_uncertainty', 0.0)
                total = ice.get('total_uncertainty', 0.0)
                aleatoric = ice.get('aleatoric_uncertainty', 0.0)
                num_clusters = ice.get('num_clusters', 0)
                sem_ent_norm = ice.get('semantic_entropy_norm', 0.0)

                aggregated_data.append({
                    'question_id': sample_id,
                    'question': question,
                    'generated_response': gen_text,
                    'bleurt_score': bleurt_score,
                    'epistemic_uncertainty': epistemic,
                    'total_uncertainty': total,
                    'aleatoric_uncertainty': aleatoric,
                    'num_clusters': num_clusters,
                    'semantic_entropy_norm': sem_ent_norm
                })
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        if aggregated_data:
            df = pd.DataFrame(aggregated_data)
            
            # Save as single CSV in result_parent and run_dir
            csv_out = os.path.join(result_parent, f"{run_id}_summary.csv")
            df.to_csv(csv_out, index=False)
            df.to_csv(os.path.join(run_dir, f"{run_id}_summary.csv"), index=False)
            print(f"Saved consolidated CSV summary ({len(df)} samples) to: {csv_out}")

            # Save as single PKL in result_parent and run_dir
            pkl_out = os.path.join(result_parent, f"{run_id}_summary.pkl")
            with open(pkl_out, 'wb') as f:
                pickle.dump(aggregated_data, f)
            with open(os.path.join(run_dir, f"{run_id}_summary.pkl"), 'wb') as f:
                pickle.dump(aggregated_data, f)
            print(f"Saved consolidated PKL summary to: {pkl_out}")


if __name__ == '__main__':
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else None
    aggregate_ice_results(target_run_id=target)
