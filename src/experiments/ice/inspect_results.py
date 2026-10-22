import os
import glob
import gzip
import pickle
import numpy as np


def inspect_ice_results(result_dir=None):
    """
    Utility script to inspect and verify ICE baseline experiment outputs.
    """
    ice_dir = os.path.dirname(os.path.abspath(__file__))
    
    if result_dir is None:
        result_parent = os.path.join(ice_dir, 'result')
        if not os.path.exists(result_parent):
            print(f"No result directory found at {result_parent}")
            return
        runs = [d for d in os.listdir(result_parent) if os.path.isdir(os.path.join(result_parent, d))]
        if not runs:
            print(f"No completed runs found inside {result_parent}")
            return
        result_dir = os.path.join(result_parent, runs[0])

    print("=" * 70)
    print(f"INSPECTING ICE RESULTS DIRECTORY: {result_dir}")
    print("=" * 70)

    files = glob.glob(os.path.join(result_dir, "results_dict_*.pkl.gz")) + glob.glob(os.path.join(result_dir, "results_dict_*.pkl"))
    files.sort(key=lambda x: int(os.path.basename(x).replace("results_dict_", "").split(".")[0]))

    if not files:
        print(f"No result files found in {result_dir}")
        return

    print(f"Found {len(files)} result files.\n")

    epistemic_list = []
    total_list = []
    aleatoric_list = []
    bleurt_list = []

    # Display detailed info for first 3 samples
    for idx, filepath in enumerate(files[:3]):
        try:
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

            sample_id = data.get('question_id', idx)
            question = data.get('question', 'N/A')
            gen_text = data.get('most_likely_generation', {}).get('generation_text', 'N/A')
            bleurt_score = data.get('correctness_dict', {}).get('bleurt', 0.0)
            
            ice = data.get('ice', {})
            epistemic = ice.get('epistemic_uncertainty', 0.0)
            total = ice.get('total_uncertainty', 0.0)
            aleatoric = ice.get('aleatoric_uncertainty', 0.0)
            num_clusters = ice.get('num_clusters', 0)
            clarifications = data.get('ice_clarifications', [])

            epistemic_list.append(epistemic)
            total_list.append(total)
            aleatoric_list.append(aleatoric)
            bleurt_list.append(bleurt_score)

            print(f"--- Sample #{sample_id} ---")
            print(f"Question     : {question}")
            print(f"Direct Answer: {gen_text}")
            print(f"BLEURT Score : {bleurt_score:.4f}")
            print(f"Clarifications ({len(clarifications)}):")
            for c_i, c_text in enumerate(clarifications[:3]):
                print(f"  [{c_i+1}] {c_text}")
            if len(clarifications) > 3:
                print(f"  ... ({len(clarifications) - 3} more)")
            print(f"Uncertainties:")
            print(f"  - Epistemic Uncertainty (EU) : {epistemic:.4f}")
            print(f"  - Total Uncertainty (TU)     : {total:.4f}")
            print(f"  - Aleatoric Uncertainty (AU) : {aleatoric:.4f}  (TU - EU)")
            print(f"  - Semantic Clusters          : {num_clusters}")
            print("-" * 70 + "\n")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    # Process all remaining files for dataset-wide summary statistics
    for filepath in files[3:]:
        try:
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

            bleurt_list.append(data.get('correctness_dict', {}).get('bleurt', 0.0))
            ice = data.get('ice', {})
            epistemic_list.append(ice.get('epistemic_uncertainty', 0.0))
            total_list.append(ice.get('total_uncertainty', 0.0))
            aleatoric_list.append(ice.get('aleatoric_uncertainty', 0.0))
        except Exception:
            continue

    print("=" * 70)
    print("DATASET SUMMARY STATISTICS:")
    print(f"Total Processed Samples : {len(files)}")
    print(f"Mean BLEURT Score       : {np.mean(bleurt_list):.4f}")
    print(f"Mean Epistemic (EU)     : {np.mean(epistemic_list):.4f}")
    print(f"Mean Total (TU)         : {np.mean(total_list):.4f}")
    print(f"Mean Aleatoric (AU)     : {np.mean(aleatoric_list):.4f}")
    print(f"Negative AU Samples     : {sum(1 for a in aleatoric_list if a < 0)} / {len(aleatoric_list)}")
    print("=" * 70)


if __name__ == '__main__':
    inspect_ice_results()
