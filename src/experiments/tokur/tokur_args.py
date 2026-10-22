import os
import json
try:
    import yaml
except ImportError:
    yaml = None


class TokURArgs:
    """
    Dedicated argument configuration for the TokUR baseline (ICLR 2026).
    Configures low-rank random weight perturbation and generation parameters
    matching TokUR (Wang-ML-Lab / ICLR 2026).
    """
    def __init__(self):
        # (0) Dataset & Models (manually select via index)
        self.dataset = [
            'scienceqa',
            'truthfulqa',
            'openbookqa'
        ][2]

        self.llm_model = [
            'Qwen/Qwen2.5-1.5B-Instruct',
            'Qwen/Qwen2.5-3B-Instruct',
            'Qwen/Qwen2.5-7B-Instruct',
            'Qwen/Qwen2.5-14B-Instruct',
            'meta-llama/Llama-3.2-3B-Instruct',
            'meta-llama/Meta-Llama-3-8B-Instruct',
            'microsoft/Phi-3-mini-4k-instruct',
            'microsoft/Phi-3-medium-4k-instruct',
            'deepseek-ai/deepseek-moe-16b-chat'
        ][0]

        self.deberta_model = [
            "deberta-base-mnli",
            "deberta-large-mnli",
            "deberta-xlarge-mnli"
        ][1]

        self.hf_token = os.environ.get('HF_TOKEN', None)

        # Sampling & Batch Controls
        self.start_idx = 0
        self.num_samples = -1  # Number of samples to process (-1 for complete dataset)
        self.max_length_of_generated_sequence = 45 if 'open' in str(self.dataset).lower() else 32

        # TokUR Specific Low-Rank Perturbation Parameters
        self.num_perturbs = 5  # Number of stochastic weight perturbation forward passes (M)
        self.rank = 4          # Low-rank dimension (r) for weight perturbation
        self.sigma = 0.05       # Noise standard deviation scaling factor (\sigma)
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Linear layers to perturb

        # Generation Hyperparameters
        self.num_beams_most_likely = 1
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
        self.quantization = None  # None, '4bit', or '8bit'
        self.seed_value = 42
        self._custom_run_id = None

    @property
    def run_id(self):
        """Automatically create run_id from dataset and LLM model name."""
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
        if any(m in raw_model for m in ['llama-3-8b', 'meta-llama-3-8b', 'llama3-8b', 'llama8b']):
            model_clean = 'meta-llama-3-8b'
        elif any(m in raw_model for m in ['llama-3.2-3b', 'llama3.2-3b']):
            model_clean = 'llama-3.2-3b'
        else:
            model_clean = raw_model

        return f"tokur_{dataset_clean}_{model_clean}"

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

