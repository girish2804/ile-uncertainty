import os
import yaml


class AllLayersArgs:
    def __init__(self):
        # (0) Dataset & Model configuration
        self.dataset = 'truthful_qa'  # 'truthful_qa', 'coqa', 'openbookqa', 'scienceqa'
        self.llm_model = 'meta-llama/Llama-3.1-70B-Instruct'  # 'meta-llama/Llama-3.1-70B-Instruct', 'Qwen/Qwen2.5-72B-Instruct', etc.
        self.deberta_model = 'deberta-large-mnli'
        self.start_idx = 0
        self.num_samples = -1  # -1 for all available samples
        self.quantization = ['4bit', '8bit', None][0]  # '4bit' NF4 for 70B models on 48GB GPU
        self.hf_token = os.environ.get('HF_TOKEN', None)
        self.is_moe = False
        self.force_overwrite = False
        self.custom_run_id = None
        self.entropies_dir = 'entropies3'  # Keep entropies2 completely untouched  # Optional custom override for folder name

        # (1) Generation Parameters (Guaranteed identical to standard SDLG)
        self.seed_value = 42
        self.num_total_generations = 10
        self.max_length_of_generated_sequence = 35
        self.compute_cleaned = False
        self.store_logits = True
        self.compress_pkl = True
        self.save_only_important_tokens = False

        # (1.1) Most Likely Generation Parameters
        self.num_beams_most_likely = 5
        self.num_return_sequences_most_likely = 1
        self.do_sample_most_likely = False
        self.temperature_most_likely = 1.0
        self.top_p_most_likely = 1.0

        # (1.2) SDLG Token Ranking & Epistemic Parameters
        self.num_beams_sdlg = 5
        self.num_return_sequences_sdlg = 1
        self.do_sample_sdlg = False
        self.temperature_sdlg = 1.0
        self.top_p_sdlg = 1.0
        self.token_prob_threshold = 0.001
        self.alphas = (1/3, 1/3, 1/3)  # Attribution, substitution, importance

        # (2) Multi-Layer & Epistemic Uncertainty Configuration
        self.depth_ratio = 0.75  # Evaluate up to 75% of model layers
        self.skip_top_neighbors = 2  # Exclude top 2 adjacent layers for calibration noise reduction
        self.relative_top = 0.1
        self.min_tokens_to_keep = 32
        self.correctness_threshold = -0.35
        self.bleurt_batch_size = 64  # Configurable batch size for BLEURT GPU vector inference
        self.eos_token_ids = []
        self.invalid_ids = []

    @property
    def run_id(self):
        if self.custom_run_id:
            return self.custom_run_id
        dataset_clean = 'truth' if 'truth' in self.dataset.lower() else ('open' if 'open' in self.dataset.lower() else ('scienceqa' if 'science' in self.dataset.lower() else self.dataset.lower()))
        model_clean = self.llm_model.split('/')[-1].lower().replace('-instruct', '').replace('-chat', '').replace('_', '-')
        quant_suffix = f"_{self.quantization}" if getattr(self, 'quantization', None) else ""
        return f"all_layers_{dataset_clean}_{model_clean}{quant_suffix}"

    def args_to_yaml(self, base_path):
        os.makedirs(base_path, exist_ok=True)
        serializable_attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        serializable_attrs['run_id'] = self.run_id
        with open(os.path.join(base_path, 'config.yaml'), 'w') as file:
            yaml.dump(serializable_attrs, file, sort_keys=False)
