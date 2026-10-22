import os
try:
    import yaml
except ImportError:
    yaml = None


class Args:
    """Experiment configuration and hyperparameters for Intermediate Layer Ensembling (ILE).

    Attributes:
        dataset: Target benchmark dataset ('truthful_qa', 'scienceqa', 'openbookqa').
        llm_model: Hugging Face model identifier for generation.
        deberta_model: NLI classifier checkpoint for semantic clustering.
        quantization: Weight precision ('4bit', '8bit', or None).
        hf_token: Authentication token for gated Hugging Face repositories.
        alphas: Weight tuple (attribution, substitution, importance) for token scoring.
    """

    def __init__(self):
        # Dataset and model selection
        self.dataset = ['truthful_qa', 'scienceqa', 'openbookqa'][0]
        self.llm_model = [
            'meta-llama/Llama-3.1-70B-Instruct',
            'Qwen/Qwen2.5-72B-Instruct',
            'Qwen/Qwen2.5-32B-Instruct',
            'Qwen/Qwen2.5-1.5B-Instruct',
            'Qwen/Qwen2.5-3B-Instruct',
            'meta-llama/Llama-3.2-3B-Instruct',
            'microsoft/Phi-3-mini-4k-instruct',
            'deepseek-ai/deepseek-moe-16b-chat'
        ][0]
        self.deberta_model = [
            "deberta-base-mnli",
            "deberta-large-mnli",
            "deberta-xlarge-mnli",
            "deberta-v2-xlarge-mnli",
            "deberta-v2-xxlarge-mnli"
        ][1]
        self.start_idx = 0
        self.num_samples = -1  # Process all samples by default
        self.num_top_layers = None
        self.num_top_layers_list = None
        self.skip_generation = False
        self.quantization = ['4bit', '8bit', None][0]
        self.hf_token = os.environ.get('HF_TOKEN', None)  # Resolved from environment or local HF cache
        self.is_moe = False
        self.force_overwrite = False

        # Generation parameters
        self.seed_value = 42
        self.num_total_generations = 10
        self.max_length_of_generated_sequence = 35
        self.eos_token_ids = None
        self.invalid_ids = []

        self.compute_cleaned = False
        self.store_logits = True
        self.save_only_important_tokens = False
        self.compress_pkl = True

        is_large = any(k in str(self.llm_model).lower() for k in ['32b', '70b', '72b'])

        # Greedy / Most likely generation
        self.num_beams_most_likely = 1 if is_large else 5
        self.num_return_sequences_most_likely = 1
        self.do_sample_most_likely = False
        self.temperature_most_likely = 1.0
        self.top_p_most_likely = 1.0

        # Intermediate layer ensembling & perturbation generation
        self.num_beams_ensemble = 1 if is_large else 5
        self.num_return_sequences_ensemble = 1
        self.do_sample_ensemble = False
        self.temperature_ensemble = 1.0
        self.top_p_ensemble = 1.0
        # Backward-compatibility aliases
        self.num_beams_sdlg = self.num_beams_ensemble
        self.num_return_sequences_sdlg = self.num_return_sequences_ensemble
        self.do_sample_sdlg = self.do_sample_ensemble
        self.temperature_sdlg = self.temperature_ensemble
        self.top_p_sdlg = self.top_p_ensemble
        self.token_prob_threshold = 0.001
        self.alphas = (1/3, 1/3, 1/3)  # Attribution, substitution, importance score weighting

        # Multinomial sampling baseline
        self.num_beams_baseline = 1
        self.num_return_sequences_baseline = 1
        self.num_beam_groups_baseline = 1
        self.diversity_penalty_baseline = 0.0
        self.do_sample_baseline = False
        self.temperature_baseline = 1.0
        self.top_p_baseline = 1.0

        # Epistemic disagreement parameters
        self.depth_ratio = 0.75
        self.skip_top_neighbors = 2
        self.relative_top = 0.1
        self.min_tokens_to_keep = 32
        self.bleurt_batch_size = 64
        self.auto_batch_compute = True
        self.entropies_dir = 'entropies_unified'
        self._custom_run_id = None

    @property
    def is_moe_architecture(self):
        """Check if model uses Mixture-of-Experts routing."""
        if getattr(self, 'is_moe', False):
            return True
        return "moe" in str(self.llm_model).lower()

    @property
    def run_id(self):
        """Derive standard run identifier from dataset, model name, and quantization."""
        if getattr(self, '_custom_run_id', None):
            return self._custom_run_id
        dataset_name = str(self.dataset).lower()
        if 'truth' in dataset_name:
            dataset_clean = 'truthfulqa'
        elif 'open' in dataset_name:
            dataset_clean = 'openbookqa'
        elif 'science' in dataset_name:
            dataset_clean = 'scienceqa'
        else:
            dataset_clean = dataset_name.replace('_', '')

        model_clean = str(self.llm_model).split('/')[-1].lower().replace('-instruct', '').replace('-chat', '').replace('_', '-')
        quant_suffix = f"_{self.quantization}" if getattr(self, 'quantization', None) else ""
        return f"{dataset_clean}_{model_clean}{quant_suffix}"

    @run_id.setter
    def run_id(self, value):
        self._custom_run_id = value

    @property
    def dynamic_run_id(self):
        return self.run_id

    def args_to_yaml(self, base_path):
        """Serialize configuration parameters to YAML format."""
        if yaml is None:
            return
        os.makedirs(base_path, exist_ok=True)
        serializable_attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        serializable_attrs['run_id'] = self.run_id
        with open(os.path.join(base_path, 'config.yaml'), 'w') as file:
            yaml.dump(serializable_attrs, file, sort_keys=False)
