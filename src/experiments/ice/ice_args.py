import os
try:
    import yaml
except ImportError:
    yaml = None


class ICEArgs:
    """
    Dedicated argument configuration for the ICE (Input Clarification Ensembling) baseline.
    Configures exact generation parameters matching the official ICE paper (Hou et al., ICML 2024).
    """
    def __init__(self):
        # (0) Dataset & Models (manually select via index)
        # (0) Dataset & Models (manually select via index)
        self.dataset = [
            'scienceqa',
            'truthfulqa',
            'openbookqa'
        ][2]

        # Model selection:
        # [0] 'Qwen/Qwen2.5-1.5B-Instruct'
        # [1] 'Qwen/Qwen2.5-3B-Instruct'       (failed model on legacy runs)
        # [2] 'Qwen/Qwen2.5-7B-Instruct'       (failed model on legacy runs)
        # [3] 'Qwen/Qwen2.5-14B-Instruct'
        # [4] 'meta-llama/Llama-3.2-3B-Instruct'
        # [5] 'meta-llama/Meta-Llama-3-8B-Instruct'
        # [6] 'microsoft/Phi-3-mini-4k-instruct'
        # [7] 'microsoft/Phi-3-medium-4k-instruct' (failed model on legacy runs)
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
            'deepseek-ai/deepseek-moe-16b-chat',
            'Qwen/Qwen1.5-MoE-A2.7B-Chat'
        ][0]

        self.clarification_model = [
            'meta-llama/Meta-Llama-3-8B-Instruct',
            'meta-llama/Llama-3.2-3B-Instruct',
            'Qwen/Qwen2.5-7B-Instruct',
            'microsoft/Phi-3-mini-4k-instruct'
        ][0]

        self.deberta_model = [
            "deberta-base-mnli",
            "deberta-large-mnli",
            "deberta-xlarge-mnli"
        ][1]

        # Options handling: automatically enabled for multiple-choice datasets (ScienceQA, OpenBookQA)
        clean_d = self.dataset.lower().replace("_", "").replace("-", "")
        self.use_options = any(d in clean_d for d in ['science', 'open', 'sciq'])

        # Dynamic run_id formulation
        model_name_clean = self.llm_model.split("/")[-1].lower().replace("-instruct", "").replace("-chat", "").replace("_", "-")
        dataset_clean = self.dataset.lower().replace("_", "").replace("-", "")
        self.run_id = f"ice-{dataset_clean}-{model_name_clean}"

        self.use_icl_prompt = True  # Enable In-Context Learning (few-shot) clarification prompting
        self.cache_clarifications = True  # Cache clarifications to disk and reuse across models
        self.force_overwrite = False  # Set to True to re-run and overwrite existing samples
        
        # Sampling & Batch Controls
        self.start_idx = 0
        self.num_samples = -1  # Number of samples to process (-1 for complete dataset)
        self.num_clarifications = 5  # Number of input clarification variants per question (K=5 or 10)
        self.max_length_of_generated_sequence = 45 if 'open' in clean_d else 35
        
        # Stage 1: Clarification Generation Hyperparameters (Nucleus Sampling T=0.7, top_p=0.95)
        self.num_beams_baseline = 1
        self.num_beam_groups_baseline = 1
        self.diversity_penalty_baseline = 0.0
        self.num_return_sequences_baseline = 1
        self.do_sample_baseline = True
        self.temperature_baseline = 0.7
        self.top_p_baseline = 0.95
        
        # Stage 2: Most Likely / Greedy Response Generation Hyperparameters
        self.num_beams_most_likely = 5
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
