import os
try:
    import yaml
except ImportError:
    yaml = None


class ASQAArgs:
    """
    Dedicated argument configuration for ASQA benchmark evaluation.
    Conforms to repository rule: Separate Args File for Separate Methods/Datasets (NO CLI parameters).
    """
    def __init__(self):
        # (0) Dataset & Models
        self.dataset = 'asqa'
        self.split = 'dev'  # 'dev' split has 948 instances (standard literature benchmark)
        
        # Model selection:
        # [0] 'Qwen/Qwen2.5-1.5B-Instruct'
        # [1] 'Qwen/Qwen2.5-3B-Instruct'
        # [2] 'Qwen/Qwen2.5-7B-Instruct'
        # [3] 'Qwen/Qwen2.5-14B-Instruct'
        # [4] 'meta-llama/Llama-3.2-3B-Instruct'
        # [5] 'meta-llama/Meta-Llama-3-8B-Instruct'
        # [6] 'microsoft/Phi-3-mini-4k-instruct'
        # [7] 'microsoft/Phi-3-medium-4k-instruct'
        # [8] 'deepseek-ai/deepseek-moe-16b-chat'
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

        self.start_idx = 0
        self.num_samples = -1  # Number of samples to process (-1 for all 948 dev samples)
        self.skip_generation = False  # Set to True to only run downstream evaluation
        self.quantization = [None, '4bit', '8bit'][0]  # Quantization mode: None, '4bit', or '8bit'
        self.hf_token = os.environ.get('HF_TOKEN', None)  # Hugging Face token passed via Args
        self.is_moe = False
        self.force_overwrite = False

        # (1.1) General generation parameters
        self.seed_value = 42
        self.num_total_generations = 10
        self.max_length_of_generated_sequence = 128
        self.eos_token_ids = None
        self.invalid_ids = []

        self.compute_cleaned = False
        self.store_logits = True
        self.save_only_important_tokens = False
        self.compress_pkl = True

        # (1.2) Most likely (greedy) generation
        self.num_beams_most_likely = 5
        self.num_return_sequences_most_likely = 1
        self.do_sample_most_likely = False
        self.temperature_most_likely = 1.0
        self.top_p_most_likely = 1.0

        # (2.1) SDLG diverse generation
        self.num_beams_sdlg = 5
        self.num_return_sequences_sdlg = 1
        self.do_sample_sdlg = False
        self.temperature_sdlg = 1.0
        self.top_p_sdlg = 1.0
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

    def args_to_yaml(self, base_path):
        if yaml is not None:
            config_path = os.path.join(base_path, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.__dict__, f, default_flow_style=False)
