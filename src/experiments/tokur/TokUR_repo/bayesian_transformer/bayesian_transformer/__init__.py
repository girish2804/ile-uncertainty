from .config import BayesianLMConfig
from .model import BayesianLMModel, BayesianForCausalLMModel
from transformers import (
    AutoConfig, 
    AutoModel, 
    AutoModelForCausalLM, 
    AutoTokenizer, TOKENIZER_MAPPING, LlamaConfig, Qwen2Config # used for registering the llama tokenizer.
)

AutoConfig.register(BayesianLMConfig.model_type, BayesianLMConfig)
AutoModel.register(BayesianLMConfig, BayesianLMModel)
AutoModelForCausalLM.register(BayesianLMConfig, BayesianForCausalLMModel)

from .vllm_models.tfb_llama import TFBLlamaConfig, LlamaModel, LlamaForCausalLM
AutoConfig.register(TFBLlamaConfig.model_type, TFBLlamaConfig)
# register the llama tokenizer for tfb-config.
# pass in the slow and fast tokenizer classes at the time of registration.
AutoTokenizer.register(TFBLlamaConfig, *TOKENIZER_MAPPING[LlamaConfig])
AutoModel.register(TFBLlamaConfig, LlamaModel)
AutoModelForCausalLM.register(TFBLlamaConfig, LlamaForCausalLM)

from .vllm_models.tfb_qwen2_vllm import TFBQwen2Config
AutoConfig.register(TFBQwen2Config.model_type, TFBQwen2Config)
# register the llama tokenizer for tfb-config.
# pass in the slow and fast tokenizer classes at the time of registration.
AutoTokenizer.register(TFBQwen2Config, *TOKENIZER_MAPPING[Qwen2Config])

# vllm model registry: https://docs.vllm.ai/en/v0.4.3/models/adding_model.html
from vllm import ModelRegistry
from .vllm_models.tfb_llama_vllm import (
    LlamaModel as VllmLlamaModel,
    LlamaForCausalLM as VllmLlamaForCausalLM,
    TFBLlamaConfig as VllmLlamaConfig,
)
from .vllm_models.tfb_qwen2_vllm import (
    Qwen2Model as VllmQwen2Model,
    Qwen2ForCausalLM as VllmQwen2ForCausalLM,
    TFBQwen2Config as VllmQwen2Config,
)
ModelRegistry.register_model("VllmTFBLLamaForCausalLM", VllmLlamaForCausalLM)
ModelRegistry.register_model("VllmTFBQwen2ForCausalLM", VllmQwen2ForCausalLM)

__all__ = ['BayesianLMConfig', 'BayesianLMModel']