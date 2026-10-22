import os
import json
try:
    import yaml
except ImportError:
    yaml = None


class TokURUnifiedArgs:
    """
    Dedicated argument configuration for the unified TokUR baseline (ICLR 2026).
    Evaluates the unquantized / full-precision model across all benchmark datasets
    (ScienceQA, TruthfulQA, OpenBookQA) in a single unified execution pipeline.
    """
    def __init__(self):
        # Target Model: Llama 3.2 1B Instruct (Full Precision)
        self.llm_model = 'meta-llama/Llama-3.2-1B-Instruct'

        # Benchmark Datasets to evaluate across sequentially
        self.datasets = [
            'scienceqa',
            'truthfulqa',
            'openbookqa'
        ]

        # Hugging Face token configuration
        self.hf_token = os.environ.get('HF_TOKEN', None)

        # Precision & Quantization Controls
        # None = unquantized baseline (full precision)
        self.quantization = None
        # 'float32' for pure FP32 full precision, or 'bfloat16' / 'float16'
        self.torch_dtype = 'float32'

        # Sampling & Batch Controls
        self.start_idx = 0
        self.num_samples = -1  # -1 for all available samples in each dataset

        # TokUR Specific Low-Rank Perturbation Parameters
        self.num_perturbs = 5  # Stochastic weight perturbation passes (M)
        self.rank = 4          # Low-rank dimension (r) for weight perturbation
        self.sigma = 0.05      # Noise standard deviation scaling factor (\sigma)
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Target attention projections

        # Generation Hyperparameters (Greedy decoding)
        self.num_beams_most_likely = 1
        self.num_return_sequences_most_likely = 1
        self.do_sample_most_likely = False
        self.temperature_most_likely = 1.0
        self.top_p_most_likely = 1.0

        # Evaluation Standards & System Controls
        self.bleurt_threshold = 0.1  # Project standard correctness threshold
        self.seed_value = 42
        self.save_frequency = 5      # Flush checkpoint every N samples

    def get_max_length_for_dataset(self, dataset_name: str) -> int:
        """Standard sequence generation length per benchmark dataset."""
        return 45 if 'open' in str(dataset_name).lower() else 32

    def get_run_id_for_dataset(self, dataset_name: str) -> str:
        """Dynamically builds canonical run_id for each dataset matching project standards."""
        ds_lower = str(dataset_name).lower()
        if any(t in ds_lower for t in ['truth', 'truthful']):
            ds_clean = 'truthfulqa'
        elif any(s in ds_lower for s in ['science', 'sciq']):
            ds_clean = 'scienceqa'
        elif any(o in ds_lower for o in ['open', 'openbook']):
            ds_clean = 'openbookqa'
        else:
            ds_clean = ds_lower.replace('_', '')

        raw_model = str(self.llm_model).split('/')[-1].lower().replace('-instruct', '').replace('-chat', '').replace('_', '-')
        if any(m in raw_model for m in ['llama-3-8b', 'meta-llama-3-8b', 'llama3-8b', 'llama8b']):
            model_clean = 'meta-llama-3-8b'
        elif any(m in raw_model for m in ['llama-3.2-3b', 'llama3.2-3b']):
            model_clean = 'llama-3.2-3b'
        elif any(m in raw_model for m in ['llama-3.2-1b', 'llama3.2-1b']):
            model_clean = 'llama-3.2-1b'
        else:
            model_clean = raw_model

        return f"tokur_{ds_clean}_{model_clean}"

    def args_to_yaml(self, base_path: str, dataset_name: str = None):
        """Saves current experiment configuration to config.yaml / config.json."""
        os.makedirs(base_path, exist_ok=True)
        serializable_attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        if dataset_name is not None:
            serializable_attrs['current_dataset'] = dataset_name
            serializable_attrs['run_id'] = self.get_run_id_for_dataset(dataset_name)
        if yaml is not None:
            with open(os.path.join(base_path, 'config.yaml'), 'w') as file:
                yaml.dump(serializable_attrs, file, sort_keys=False)
        else:
            with open(os.path.join(base_path, 'config.json'), 'w') as file:
                json.dump(serializable_attrs, file, indent=2)
