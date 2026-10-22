import os
import json
try:
    import yaml
except ImportError:
    yaml = None


class TokURQuantizedArgs:
    """
    Dedicated argument configuration for the TokUR baseline (Wang-ML-Lab / ICLR 2026)
    running large (e.g. 70B/72B) quantized models (4-bit NF4 / 8-bit BitsAndBytes).
    Configures low-rank random weight perturbation and generation parameters
    with memory-efficient optimizations for 70B/72B architectures.
    """
    def __init__(self):
        # (0) Dataset configuration (select via index)
        # [0] 'truthfulqa'
        # [1] 'openbookqa'
        # [2] 'scienceqa'
        self.dataset = [
            'truthfulqa',
            'openbookqa',
            'scienceqa'
        ][0]

        # (1) Large Language Model selection (32B / 70B / 72B / large scale)
        # [0] 'meta-llama/Llama-3.1-70B-Instruct'
        # [1] 'meta-llama/Meta-Llama-3-70B-Instruct'
        # [2] 'Qwen/Qwen2.5-72B-Instruct'
        # [3] 'Qwen/Qwen2.5-32B-Instruct'
        # [4] 'meta-llama/Meta-Llama-3-8B-Instruct'
        # [5] 'Qwen/Qwen2.5-14B-Instruct'
        self.llm_model = [
            'meta-llama/Llama-3.1-70B-Instruct',
            'meta-llama/Meta-Llama-3-70B-Instruct',
            'Qwen/Qwen2.5-72B-Instruct',
            'Qwen/Qwen2.5-32B-Instruct',
            'meta-llama/Meta-Llama-3-8B-Instruct',
            'Qwen/Qwen2.5-14B-Instruct'
        ][0]

        # (2) Quantization Configuration (BitsAndBytes)
        # [0] '4bit' (NF4 double-quant with bfloat16 compute dtype - recommended for 70B on 48GB GPU)
        # [1] '8bit'
        # [2] None (Full 16-bit unquantized)
        self.quantization = ['4bit', '8bit', None][0]

        # (3) TokUR Specific Low-Rank Perturbation Parameters
        self.num_perturbs = 5  # Number of stochastic weight perturbation forward passes (M)
        self.rank = 4          # Low-rank dimension (r) for weight perturbation
        self.sigma = 0.05      # Noise standard deviation scaling factor (\sigma)
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Target attention projections

        # (4) Attention Implementation & Hardware Efficiency
        self.attn_implementation = 'sdpa'  # PyTorch native scaled dot product attention

        # Sampling & Batch Controls
        self.start_idx = 0
        self.num_samples = -1  # Number of samples to process (-1 for complete dataset)
        self.max_length_of_generated_sequence = 45 if 'open' in str(self.dataset).lower() else 35

        # Generation Hyperparameters
        self.num_beams_most_likely = 1  # Greedy decoding for fast 70B inference
        self.num_return_sequences_most_likely = 1
        self.do_sample_most_likely = False
        self.temperature_most_likely = 1.0
        self.top_p_most_likely = 1.0

        # Output & Logits Controls
        self.eos_token_ids = None  # Dynamically set per model tokenizer
        self.invalid_ids = []
        self.compute_cleaned = False
        self.store_logits = True
        self.save_only_important_tokens = False
        self.compress_pkl = True

        # Hardware & Access Controls
        self.seed_value = 42
        self.hf_token = os.environ.get('HF_TOKEN', None)
        self._custom_run_id = None

    @property
    def run_id(self):
        """Automatically create run_id from dataset, LLM model name, and quantization."""
        if getattr(self, '_custom_run_id', None):
            return self._custom_run_id
        dataset_lower = str(self.dataset).lower()
        if any(t in dataset_lower for t in ['truth', 'truthful']):
            dataset_clean = 'truthfulqa'
        elif any(s in dataset_lower for s in ['science', 'sciq']):
            dataset_clean = 'scienceqa'
        elif any(o in dataset_lower for o in ['open', 'openbook']):
            dataset_clean = 'openbookqa'
        else:
            dataset_clean = dataset_lower.replace('_', '')

        raw_model = str(self.llm_model).split('/')[-1].lower().replace('-instruct', '').replace('-chat', '').replace('_', '-')
        quant_suffix = f"_{self.quantization}" if getattr(self, 'quantization', None) else ""
        return f"tokur_{dataset_clean}_{raw_model}{quant_suffix}"

    @run_id.setter
    def run_id(self, value):
        self._custom_run_id = value

    @property
    def dynamic_run_id(self):
        return self.run_id

    def args_to_yaml(self, base_path):
        """Saves current experiment configuration to config.yaml."""
        os.makedirs(base_path, exist_ok=True)
        serializable_attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        serializable_attrs['run_id'] = self.run_id
        if yaml is not None:
            with open(os.path.join(base_path, 'config.yaml'), 'w') as file:
                yaml.dump(serializable_attrs, file, sort_keys=False)
        else:
            with open(os.path.join(base_path, 'config.json'), 'w') as file:
                json.dump(serializable_attrs, file, indent=2)
