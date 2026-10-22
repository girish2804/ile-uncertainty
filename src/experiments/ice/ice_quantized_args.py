import os
try:
    import yaml
except ImportError:
    yaml = None


class ICEQuantizedArgs:
    """
    Dedicated argument configuration for the ICE (Input Clarification Ensembling) baseline
    running large (e.g. 70B) quantized models (4-bit NF4 / 8-bit BitsAndBytes).
    Configures generation parameters matching the official ICE paper (Hou et al., ICML 2024)
    with memory-efficient optimizations for 70B architectures.
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

        # (3) Clarification Generator Model (Stage 1)
        # Standard ICE clarification generator (Hou et al. 2024)
        self.clarification_model = [
            'meta-llama/Meta-Llama-3-8B-Instruct',
            'meta-llama/Llama-3.2-3B-Instruct',
            'Qwen/Qwen2.5-7B-Instruct'
        ][0]

        # (4) DeBERTa Model for Semantic Clustering (Stage 3)
        self.deberta_model = [
            "deberta-base-mnli",
            "deberta-large-mnli",
            "deberta-xlarge-mnli"
        ][1]
        self.deberta_device = 'cuda'  # Standard GPU execution for DeBERTa (identical to SDLG)

        # (5) Attention Implementation & Hardware Efficiency
        self.attn_implementation = 'sdpa'  # PyTorch native scaled dot product attention

        # Options handling: automatically enabled for multiple-choice datasets (ScienceQA, OpenBookQA)
        clean_d = self.dataset.lower().replace("_", "").replace("-", "")
        self.use_options = any(d in clean_d for d in ['science', 'open', 'sciq'])

        self._run_id = None
        self.use_icl_prompt = True  # Enable In-Context Learning (few-shot) clarification prompting
        self.cache_clarifications = True  # Cache clarifications to disk and reuse across models
        self.force_overwrite = False  # Set to True to re-run and overwrite existing samples
        
        # Sampling & Batch Controls
        self.start_idx = 0
        self.num_samples = -1  # Number of samples to process (-1 for complete dataset)
        self.num_clarifications = 5  # Number of input clarification variants per question (K=5)
        self.max_length_of_generated_sequence = 45 if 'open' in clean_d else 35
        
        # Stage 1: Clarification Generation Hyperparameters (Nucleus Sampling T=0.7, top_p=0.95)
        self.num_beams_baseline = 1
        self.num_beam_groups_baseline = 1
        self.diversity_penalty_baseline = 0.0
        self.num_return_sequences_baseline = 1
        self.do_sample_baseline = True
        self.temperature_baseline = 0.7
        self.top_p_baseline = 0.95
        
        # Stage 2: Response Generation Hyperparameters on 70B
        # Greedy decoding (num_beams=1) matches official ICE paper and ensures scalable 70B inference
        self.num_beams_most_likely = 1
        self.num_return_sequences_most_likely = 1
        self.do_sample_most_likely = False
        self.temperature_most_likely = 1.0
        self.top_p_most_likely = 1.0

        # General / Logits Controls
        self.eos_token_ids = None  # Dynamically set per model tokenizer
        self.invalid_ids = []
        self.compute_cleaned = False
        self.store_logits = True
        self.save_only_important_tokens = False
        self.compress_pkl = True

        # Hardware & Access Controls
        self.seed_value = 42
        self.hf_token = os.environ.get('HF_TOKEN', None)

    @property
    def run_id(self):
        if getattr(self, '_run_id', None):
            return self._run_id
        model_name_clean = str(self.llm_model).split("/")[-1].lower().replace("-instruct", "").replace("-chat", "").replace("_", "-")
        dataset_clean = str(self.dataset).lower().replace("_", "").replace("-", "")
        quant_suffix = f"_{self.quantization}" if getattr(self, 'quantization', None) else ""
        return f"ice-{dataset_clean}-{model_name_clean}{quant_suffix}"

    @run_id.setter
    def run_id(self, val):
        self._run_id = val

    def args_to_yaml(self, base_path):
        """Saves current experiment configuration to config.yaml (or config.json if yaml missing)."""
        os.makedirs(base_path, exist_ok=True)
        serializable_attrs = {k: v for k, v in self.__dict__.items()}
        if yaml is not None:
            with open(os.path.join(base_path, 'config.yaml'), 'w') as file:
                yaml.dump(serializable_attrs, file, sort_keys=False)
        else:
            import json
            with open(os.path.join(base_path, 'config.json'), 'w') as file:
                json.dump(serializable_attrs, file, indent=2)
