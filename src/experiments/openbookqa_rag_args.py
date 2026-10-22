import os
try:
    import yaml
except ImportError:
    yaml = None


class OpenBookQARAGArgs:
    """
    Unified argument configuration for OpenBookQA RAG Routing & Epistemic Dissipation Evaluation.
    Supports Dense Semantic Search (Top-1 / Top-3), BM25-Optimized (BM25+), Legacy BM25, and Oracle retrieval.
    Conforms to repository rules:
      - Dedicated Args class per method/experiment (NO CLI parameters).
      - Hugging Face token passed directly via Args attribute.
      - Safe versioning for output paths.
    """
    def __init__(self):
        # 0. Dataset Configuration
        self.dataset = 'openbookqa'
        self.split = 'test'  # 500 test samples in allenai/openbookqa 'additional' config
        
        # 1. Model Selection
        # Options:
        # [0] 'Qwen/Qwen2.5-1.5B-Instruct'
        # [1] 'Qwen/Qwen2.5-3B-Instruct'
        # [2] 'Qwen/Qwen2.5-7B-Instruct'
        # [3] 'Qwen/Qwen2.5-14B-Instruct'
        # [4] 'meta-llama/Llama-3.2-1B-Instruct'
        # [5] 'meta-llama/Llama-3.2-3B-Instruct'
        # [6] 'meta-llama/Meta-Llama-3-8B-Instruct'
        # [7] 'microsoft/Phi-3-mini-4k-instruct'
        # [8] 'microsoft/Phi-3-medium-4k-instruct'
        self.llm_model = [
            'Qwen/Qwen2.5-1.5B-Instruct',
            'Qwen/Qwen2.5-3B-Instruct',
            'Qwen/Qwen2.5-7B-Instruct',
            'Qwen/Qwen2.5-14B-Instruct',
            'meta-llama/Llama-3.2-1B-Instruct',
            'meta-llama/Llama-3.2-3B-Instruct',
            'meta-llama/Meta-Llama-3-8B-Instruct',
            'microsoft/Phi-3-mini-4k-instruct',
            'microsoft/Phi-3-medium-4k-instruct'
        ][0]

        # Hugging Face access token passed directly through Args
        self.hf_token = os.environ.get('HF_TOKEN', None)
        
        # 2. Retrieval Configuration
        # Modes:
        #   - 'paired'   : Runs BOTH BM25-Optimized (Top-3) and Dense Top-1 sequentially for self.llm_model
        #   - 'dense'    : Dense Semantic Embeddings (all-MiniLM-L6-v2)
        #   - 'bm25_opt' : Optimized BM25+ (Query Distillation, Stemming, Stopwords, BM25+ delta=0.5)
        #   - 'bm25'     : Legacy Naive BM25 (Concatenated choices query)
        #   - 'oracle'   : Perfect retrieval (Ground truth fact1)
        self.retrieval_mode = 'paired'  # 'paired', 'dense', 'bm25_opt', 'bm25', 'oracle'
        self.dense_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.top_k_retrieval = 3  # 1 for Top-1 ablation, 3 for standard Top-3
        
        # BM25 / BM25+ Parameters
        self.bm25_k1 = 1.5
        self.bm25_b = 0.75
        self.bm25_delta = 0.5  # Additive TF offset for bm25_opt
        
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
        self.corpus_path = os.path.join(data_dir, 'openbook.txt')
        self._retrieval_cache_path = None
        self.force_reretrieve = False  # Set to True to re-compute retrieval cache from scratch
        
        # Execution Slice
        self.start_idx = 0
        self.num_samples = -1  # -1 for all 500 test samples
        self.skip_generation = False  # Set to True to re-run metrics/BLEURT on existing raw files
        self.force_overwrite = False
        
        # Generation Parameters (64 max tokens to prevent mid-sentence cut-offs)
        self.seed_value = 42
        self.max_length_of_generated_sequence = 64
        self.eos_token_ids = None
        self.invalid_ids = []
        
        # Greedy generation settings
        self.num_beams = 1
        self.do_sample = False
        self.temperature = 1.0
        self.top_p = 1.0
        
        # Uncertainty & In-situ Layer Hidden State Probing Parameters
        self.relative_top = 0.1
        self.min_tokens_to_keep = 32
        self.skip_top_neighbors = 0
        self.compress_pkl = True
        
        # Evaluation & BLEURT Parameters
        self.bleurt_batch_size = 64
        self.bleurt_threshold = 0.0  # Standard BLEURT correctness threshold
        
        # Output directory naming (Auto-derived from retrieval_mode & top_k, or explicit override)
        self._output_dir_prefix = None

    @property
    def output_dir_prefix(self):
        if self._output_dir_prefix is not None:
            return self._output_dir_prefix
        if self.retrieval_mode == 'dense' and self.top_k_retrieval == 1:
            return "openbookqa_top1_rag"
        elif 'opt' in self.retrieval_mode:
            return "openbookqa_bm25opt_rag"
        elif self.retrieval_mode == 'dense':
            return "openbookqa_dense_rag"
        elif self.retrieval_mode == 'bm25':
            return "openbookqa_rag"
        elif self.retrieval_mode == 'oracle' and self.top_k_retrieval == 1:
            return "openbookqa_oracle_top1_rag"
        elif self.retrieval_mode == 'oracle':
            return "openbookqa_oracle_rag"
        elif self.retrieval_mode in ['distractor', 'random_distractor', 'random']:
            return "openbookqa_distractor_rag"
        return f"openbookqa_{self.retrieval_mode}_top{self.top_k_retrieval}_rag"

    @output_dir_prefix.setter
    def output_dir_prefix(self, val):
        self._output_dir_prefix = val

    @property
    def retrieval_cache_path(self):
        if self._retrieval_cache_path is not None:
            return self._retrieval_cache_path
        data_dir = os.path.dirname(self.corpus_path)
        return os.path.join(data_dir, f'openbook_retrieved_{self.retrieval_mode}_top{self.top_k_retrieval}.json')

    @retrieval_cache_path.setter
    def retrieval_cache_path(self, val):
        self._retrieval_cache_path = val

    def args_to_yaml(self, base_path):
        if yaml is not None:
            config_path = os.path.join(base_path, 'config.yaml')
            data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
            data['output_dir_prefix'] = self.output_dir_prefix
            data['retrieval_cache_path'] = self.retrieval_cache_path
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False)
