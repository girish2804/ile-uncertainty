import os
try:
    import yaml
except ImportError:
    yaml = None


class OpenBookQAArgs:
    """
    Dedicated argument configuration for OpenBookQA benchmark evaluation.
    Conforms to repository rule: Separate Args File for Separate Methods/Datasets (NO CLI parameters).
    """
    def __init__(self):
        # (0) Dataset & Models
        self.dataset = 'openbookqa'
        self.split = 'test'  # 'test' split has 500 instances in allenai/openbookqa
        
        # Model selection:
        # [0] 'meta-llama/Llama-3.1-70B-Instruct'
        # [1] 'Qwen/Qwen2.5-72B-Instruct'
        # [2] 'Qwen/Qwen2.5-32B-Instruct'
        # [3] 'Qwen/Qwen2.5-1.5B-Instruct'
        # [4] 'Qwen/Qwen2.5-3B-Instruct'
        # [5] 'Qwen/Qwen2.5-7B-Instruct'
        # [6] 'Qwen/Qwen2.5-14B-Instruct'
        # [7] 'meta-llama/Llama-3.2-3B-Instruct'
        # [8] 'meta-llama/Meta-Llama-3-8B-Instruct'
        # [9] 'microsoft/Phi-3-mini-4k-instruct'
        # [10] 'microsoft/Phi-3-medium-4k-instruct'
        # [11] 'deepseek-ai/deepseek-moe-16b-chat'
        self.llm_model = [
            'meta-llama/Llama-3.1-70B-Instruct',
            'Qwen/Qwen2.5-72B-Instruct',
            'Qwen/Qwen2.5-32B-Instruct',
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

        self.start_idx = 0
        self.num_samples = -1  # Number of samples to process (-1 for all 500 test samples)
        self.skip_generation = False  # Set to True to only run downstream evaluation
        self.quantization = ['4bit', '8bit', None][2]  # Quantization mode: '4bit', '8bit', or None
        self.hf_token = os.environ.get('HF_TOKEN', None)  # Hugging Face token passed via Args
        self.is_moe = False
        self.force_overwrite = False

        # (1.1) General generation parameters
        self.seed_value = 42
        self.num_total_generations = 10
        self.max_length_of_generated_sequence = 45  # Single-sentence response matching ScienceQA
        self.eos_token_ids = None
        self.invalid_ids = []

        self.compute_cleaned = False
        self.store_logits = True
        self.save_only_important_tokens = False
        self.compress_pkl = True

        # For 32B/70B/72B architectures, use greedy decoding (num_beams=1) to prevent VRAM explosion; 5 for smaller models
        is_large_model = any(k in str(self.llm_model).lower() for k in ['32b', '70b', '72b'])

        # (1.2) Most likely (greedy) generation
        self.num_beams_most_likely = 1 if is_large_model else 5
        self.num_return_sequences_most_likely = 1
        self.do_sample_most_likely = False
        self.temperature_most_likely = 1.0
        self.top_p_most_likely = 1.0

        # (2.1) Intermediate layer ensembling & diverse generation
        self.num_beams_ensemble = 1 if is_large_model else 5
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
        self.alphas = (1/3, 1/3, 1/3)

        # (2.2) Baseline sampling
        self.num_beams_baseline = 1
        self.num_return_sequences_baseline = 1
        self.do_sample_baseline = True
        self.temperature_baseline = 0.5
        self.top_p_baseline = 0.99

        # Downstream evaluation parameters
        self.bleurt_batch_size = 64
        self.auto_batch_compute = True
        self.entropies_dir = 'entropies_unified'

    @property
    def run_id(self):
        model_name_clean = str(self.llm_model).split("/")[-1].lower()
        model_name_clean = model_name_clean.replace("-instruct", "").replace("-chat", "").replace("_", "-")
        quant_suffix = f"_{self.quantization}" if getattr(self, 'quantization', None) else ""
        return f"openbookqa_{model_name_clean}{quant_suffix}"

    def args_to_yaml(self, base_path):
        if yaml is not None:
            config_path = os.path.join(base_path, 'config.yaml')
            with open(config_path, 'w') as f:
                d = dict(self.__dict__)
                d['run_id'] = self.run_id
                yaml.dump(d, f, default_flow_style=False)
