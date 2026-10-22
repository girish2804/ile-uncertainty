# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, List, Optional, Tuple, Union

import torch
from dataclasses import dataclass
from transformers.utils import ModelOutput
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding, LlamaAttention, LlamaMLP, LlamaRMSNorm,
    LLAMA_START_DOCSTRING, LLAMA_INPUTS_DOCSTRING,
    LlamaDecoderLayer as BaseLlamaDecoderLayer,
)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "TFBLlamaConfig"


def pop_last_state_from_cache(cache):
    """pop the last state from the cache."""
    assert isinstance(cache, Cache), f"The cache should be a `Cache` object, while it's {type(cache)}."
    
    last_key, last_value = [], []

    num_layers = len(cache.key_cache)
    for layer in range(num_layers): 
        # here we assume the cache is a list of tuples, 
        # and keys & values are of the same shape.
        last_key.append(cache.key_cache[layer][:, :, -1:, :])
        last_value.append(cache.value_cache[layer][:, :, -1:, :])
        cache.key_cache[layer] = cache.key_cache[layer][:, :, :-1, :]
        cache.value_cache[layer] = cache.value_cache[layer][:, :, :-1, :]
    
    return cache, last_key, last_value


class TFBLlamaConfig(LlamaConfig):
    model_type = "tfb_llama"

    def __init__(
        self,
        base_lm_path: str = None,
        bayes_sigma: float = 1e-1,
        bayes_noise: str = 'right',
        basis_idx: tuple = (0,),
        sample: bool = True,
        num_samples: int = 10,
        lowrank: bool = True,
        init_basis_vectors: bool = True,
        use_tfb_for_generation: bool = False,
        **kwargs,
    ):
        if bayes_noise not in ["right", "left", "std", "self-scaling"]:
            raise ValueError(f"`bayes_noise` must be either 'right' or 'left', got {bayes_noise}.")
                
        # now the Bayesian Config only stores the path of the base_lm_config.
        self.base_lm_path = base_lm_path
        self.bayes_sigma = bayes_sigma
        self.bayes_noise = bayes_noise
        self.basis_idx = basis_idx
        self.sample = sample
        self.num_samples = num_samples
        self.lowrank = lowrank
        self.init_basis_vectors = init_basis_vectors
        self.use_tfb_for_generation = use_tfb_for_generation

        super().__init__(**kwargs)


class LowRankBayesianLinear(nn.Linear):
    def __init__(
        self,
        base_layer=None, 
        in_features=None, 
        out_features=None, 
        bias=True, 
        device=None, 
        dtype=None,
        bayes_sigma=1e-3,
        bayes_noise='right',
        basis_idx=(0,),
    ):
        if base_layer is None:
            assert in_features is not None and out_features is not None, "in_features and out_features should be provided."
            super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        elif isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
            super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
            # copy the weight and bias from the base_layer.
            self.weight.data.copy_(base_layer.weight.data)
            if bias:
                self.bias.data.copy_(base_layer.bias.data)
        else:
            raise ValueError(f'base_layer {base_layer} must be None or nn.Linear')
        
        # TFB-specific
        assert bayes_noise in ('right', 'left'), f'bayes_noise {bayes_noise} must be `right` or `left`'
        self.bayes_noise = bayes_noise
        self.bayes_sigma = bayes_sigma
        self.basis_idx = basis_idx

        # not computing the basis_vectors by default if we load the parameters.
        dtype = self.weight.dtype
        if bayes_noise == 'right': 
            buffer_shape = (self.weight.shape[0], len(basis_idx))
        else:
            buffer_shape = (self.weight.shape[1], len(basis_idx))
        self.register_buffer("basis_vectors", torch.zeros(buffer_shape).to(dtype))

        # by default, use sampling when forward.
        self.set_sample(sample=True)
    
    def forward(self, x):
        # make self.sample as an internal status of a layer.
        sample = self.sample
        # return the mean prediction
        if not sample: 
            return super().forward(x)
        
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        if self.bayes_noise == 'right':
            # \sum_{i=1}_{r} \eps_i b_i, where \eps_i ~ N(0, \sigma_q^2 * ||x||_2^2)
            # noise.shape: (batch_size, r)
            noise = torch.normal(0, 1, size=(x.shape[0], self.basis_vectors.shape[1]), device=x.device, dtype=x.dtype)
            # noise is proportional to the norm of x.
            noise *= self.bayes_sigma*x.square().sum(-1, keepdims=True).sqrt() # shape: (1, batch_size)
            # x.shape: (batch_size, in_features)
            # self.bayes_vector.shape: (out_features, r)
            return (super().forward(x) + noise @ self.basis_vectors.T).view(*original_shape[:-1], -1)
        elif self.bayes_noise == 'left':
            # noise_scale.shape: (batch_size, 1)
            noise_scale = self.bayes_sigma * (x @ self.basis_vectors).square().sum(-1, keepdims=True).sqrt()
            noise = noise_scale * torch.normal(0, 1, size=(x.shape[0], self.out_features), device=x.device, dtype=x.dtype)
            return (super().forward(x) + noise).view(*original_shape[:-1], -1)
    
    @torch.no_grad()
    def fetch_basis_vectors(self):
        assert hasattr(self, 'basis_vectors'), "basis_vectors should be registered as a buffer before fetching."
        # need to use SVD to calculate the basis vectors, which could be a bit slow.
        old_dtype = self.weight.dtype
        U, _, V = torch.svd(self.weight.float())
        if self.bayes_noise == 'right':
            # right vector is noise, so self.bayes_vector = left vector
            self.basis_vectors.copy_(U[:, self.basis_idx].to(old_dtype))
        else:
            # left vector is noise so self.bayes_vector = right vector 
            self.basis_vectors.copy_(V[:, self.basis_idx].to(old_dtype))
    
    def set_sample(self, sample=True):
        self.sample = sample
        
    def set_sigma(self, sigma=1e-3):
        self.bayes_sigma = sigma


class LlamaAttentionTFB(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: TFBLlamaConfig, layer_idx: int):
        super(LlamaAttention, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = LowRankBayesianLinear(
            base_layer=None,
            in_features=config.hidden_size, 
            out_features=config.num_attention_heads * self.head_dim, 
            bias=config.attention_bias,
            bayes_sigma=config.bayes_sigma,
            bayes_noise=config.bayes_noise,
            basis_idx=config.basis_idx,
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = LowRankBayesianLinear(
            base_layer=None,
            in_features=config.hidden_size, 
            out_features=config.num_key_value_heads * self.head_dim, 
            bias=config.attention_bias,
            bayes_sigma=config.bayes_sigma,
            bayes_noise=config.bayes_noise,
            basis_idx=config.basis_idx,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )


class LlamaDecoderLayer(BaseLlamaDecoderLayer):
    def __init__(self, config: TFBLlamaConfig, layer_idx: int):
        super(BaseLlamaDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size

        ### TFB-specific
        self.self_attn = LlamaAttentionTFB(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = TFBLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # TFB-specific
    def post_init(self):
        super().post_init()
        # register all the low-rank Bayesian Linear layers.
        self.bayesian_layers = []
        for _, module in self.named_modules():
            if isinstance(module, LowRankBayesianLinear):
                self.bayesian_layers.append(module)

    # TFB-specific
    def __init__(self, config: TFBLlamaConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.bayesian_layers = []
        self.num_samples = config.num_samples
        self.sample = config.sample
    
    # TFB-specific
    def set_sample(self, sample=True):
        """set the sample status of all Bayesian layers."""
        self.sample = sample
        for layer in self.bayesian_layers:
            layer.set_sample(sample)
    
    # TFB-specific
    def set_sigma(self, sigma=1e-3):
        """set the tfb standard deviation sigma of all Bayesian layers."""
        for layer in self.bayesian_layers:
            layer.set_sigma(sigma)

    # TFB-specific
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        if model.config.init_basis_vectors:
            for layer in model.bayesian_layers:
                layer.fetch_basis_vectors()
        return model


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: TFBLlamaConfig
    """

    def __init__(self, config: TFBLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids, **kwargs):
        """forward pass of the Bayesian model, shared by the LlamaModel and LlamaForCausalModel."""
        if not self.sample:
            return self.single_forward(input_ids, **kwargs)
        
        num_samples = self.num_samples
        assert num_samples > 1, f"num_samples should be greater than when self.sample=True, while it's {self.num_samples}."

        # use the mean of the weights, and use the rest of the output as usual.
        self.set_sample(sample=False)
        aggregated_output = self.single_forward(input_ids, **kwargs)

        # collect all samples
        samples = []
        samples.append(aggregated_output.logits.clone())

        # convert the logits first to probability.
        aggregated_output.logits = torch.softmax(aggregated_output.logits, dim=-1)

        # aggregated_last_key and aggregated_last_value are the aggregated key and value of the past_key_values.
        # shape of .key_cache & .value_cache: 
        #   [num_layers, num_heads, seq_len, head_dim]
        # kv_cache points to the aggregated_output['past_key_values'], i.e., they are the same object.
        if aggregated_output.past_key_values is not None:
            kv_cache, aggregated_last_key, aggregated_last_value = pop_last_state_from_cache(aggregated_output['past_key_values'])
            num_layers = len(kv_cache.key_cache)
        
        self.set_sample(sample=True)
        for i in range(1, num_samples):
            sampled_output = self.single_forward(input_ids, **kwargs)

            # collect sample
            samples.append(sampled_output.logits.clone())

            # aggregating logits 
            aggregated_output.logits.add_(torch.softmax(sampled_output.logits, dim=-1))

            # aggregating the kv_cache
            if aggregated_output.past_key_values is not None:
                # since `forward` of the Attn module automatically updates the cache; 
                # need to pop the last state for kv_cache aggregation and update the cache.
                _, last_key, last_value = pop_last_state_from_cache(sampled_output.past_key_values)

                # aggregating the last key and value of the past_key_values.
                # past_key_values[batch_idx][:-1] are exactly the same; only aggregate the last one.
                for layer in range(num_layers):
                    # aggregated_last_key[layer] = (aggregated_last_key[layer]*i + last_key[layer]) / (i+1)
                    # aggregated_last_value[layer] = (aggregated_last_value[layer]*i + last_value[layer]) / (i+1)
                    aggregated_last_key[layer].add_(last_key[layer])
                    aggregated_last_value[layer].add_(last_value[layer])

            # aggregating hidden states.
            if aggregated_output.hidden_states is not None:
                aggregated_output.hidden_states[-1].add_(sampled_output.hidden_states[-1])
        
        # average the logits (prob. and then log back)
        aggregated_output.logits.div_(num_samples).log_()

        # average the kv_cache.
        if aggregated_output.past_key_values is not None:
            for layer in range(num_layers):
                aggregated_last_key[layer].div_(num_samples)
                aggregated_last_value[layer].div_(num_samples)
                kv_cache.key_cache[layer] = torch.cat([kv_cache.key_cache[layer], aggregated_last_key[layer]], dim=-2)
                kv_cache.value_cache[layer] = torch.cat([kv_cache.value_cache[layer], aggregated_last_value[layer]], dim=-2)

        # average the last hidden states.
        if aggregated_output.hidden_states is not None:
            aggregated_output.hidden_states[-1].div_(num_samples)

        # create new output with samples
        return CausalLMOutputWithSamples(
            loss=aggregated_output.loss,
            logits=aggregated_output.logits,
            past_key_values=aggregated_output.past_key_values,
            hidden_states=aggregated_output.hidden_states,
            attentions=aggregated_output.attentions,
            samples=torch.stack(samples, dim=0)
        )
    # TODO: forward function needs to be updated for LlamaModel.
    # def forward(self, input_ids, **kwargs):
    #     """forward pass of the Bayesian model, shared by the LlamaModel and LlamaForCausalModel."""
    #     if not self.sample:
    #         return self.single_forward(input_ids, **kwargs)

    #     ### original output type:
    #     # output = BaseModelOutputWithPast(
    #     #     last_hidden_state=hidden_states,
    #     #     past_key_values=past_key_values if use_cache else None,
    #     #     hidden_states=all_hidden_states,
    #     #     attentions=all_self_attns,
    #     # )
    #     # last_hidden_state: [batch_size, seq_len, hidden_size]
    #     # past_key_values: Cache (done)
    #     # hidden_states: tuple of [batch_size, seq_len, hidden_size]
    #     # attentions: no need to worry about, simply return the attentions given by the weights mean (w/o sampling).
        
    #     num_samples = self.num_samples
    #     assert num_samples > 1, f"num_samples should be greater than when self.sample=True, while it's {self.num_samples}."

    #     # use the mean of the weights, and use the rest of the output as usual.
    #     self.set_sample(sample=False)
    #     aggregated_output = self.single_forward(input_ids, **kwargs)

    #     # aggregated_last_key and aggregated_last_value are the aggregated key and value of the past_key_values.
    #     # shape of .key_cache & .value_cache: 
    #     #   [num_layers, num_heads, seq_len, head_dim]
    #     # kv_cache points to the aggregated_output['past_key_values'], i.e., they are the same object.
    #     if aggregated_output.past_key_values is not None:
    #         kv_cache, aggregated_last_key, aggregated_last_value = pop_last_state_from_cache(aggregated_output.past_key_values)

    #     num_layers = len(kv_cache.key_cache)
    #     self.set_sample(sample=True)
    #     for i in range(1, num_samples):
    #         sampled_output = self.single_forward(input_ids, **kwargs)

    #         # aggregating hidden states.
    #         aggregated_output.last_hidden_state.add_(sampled_output.last_hidden_state)

    #         if aggregated_output.past_key_values is not None:
    #             # since `forward` of the Attn module automatically updates the cache; 
    #             # need to pop the last state for kv_cache aggregation and update the cache.
    #             _, last_key, last_value = pop_last_state_from_cache(sampled_output.past_key_values)
                
    #             # aggregating the last key and value of the past_key_values.
    #             # past_key_values[batch_idx][:-1] are exactly the same; only aggregate the last one.
    #             for layer in range(num_layers):
    #                 # aggregated_last_key[layer] = (aggregated_last_key[layer]*i + last_key[layer]) / (i+1)
    #                 # aggregated_last_value[layer] = (aggregated_last_value[layer]*i + last_value[layer]) / (i+1)
    #                 aggregated_last_key[layer].add_(last_key[layer])
    #                 aggregated_last_value[layer].add_(last_value[layer])
            
    #         # del sampled_output
        
    #     # average the hidden states.
    #     aggregated_output.last_hidden_state.div_(num_samples)

    #     # average the kv_cache if returned.
    #     if aggregated_output.past_key_values is not None:
    #         for layer in range(num_layers):
    #             aggregated_last_key[layer].div_(num_samples)
    #             aggregated_last_value[layer].div_(num_samples)
    #             kv_cache.key_cache[layer] = torch.cat([kv_cache.key_cache[layer], aggregated_last_key[layer]], dim=-2)
    #             kv_cache.value_cache[layer] = torch.cat([kv_cache.value_cache[layer], aggregated_last_value[layer]], dim=-2)
        
    #     # aggregated_output.hidden_states are just tuple of all the last_hidden_states.
    #     if aggregated_output.hidden_states is not None:
    #         aggregated_output.hidden_states = aggregated_output.hidden_states[:-1] + (aggregated_output.last_hidden_state,)

    #     # ignore the attention mask, keep as it is.
    #     return aggregated_output

    
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def single_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """original forward method of Llama."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self, input_ids, **kwargs):
        """forward pass of the Bayesian model, shared by the LlamaModel and LlamaForCausalModel."""
        if not self.sample:
            return self.single_forward(input_ids, **kwargs)
        
        num_samples = self.num_samples
        assert num_samples > 1, f"num_samples should be greater than when self.sample=True, while it's {self.num_samples}."

        # use the mean of the weights, and use the rest of the output as usual.
        self.set_sample(sample=False)
        aggregated_output = self.single_forward(input_ids, **kwargs)

        # collect all samples
        samples = []
        samples.append(aggregated_output.logits.clone())

        # convert the logits first to probability.
        aggregated_output.logits = torch.softmax(aggregated_output.logits, dim=-1)

        # aggregated_last_key and aggregated_last_value are the aggregated key and value of the past_key_values.
        # shape of .key_cache & .value_cache: 
        #   [num_layers, num_heads, seq_len, head_dim]
        # kv_cache points to the aggregated_output['past_key_values'], i.e., they are the same object.
        if aggregated_output.past_key_values is not None:
            kv_cache, aggregated_last_key, aggregated_last_value = pop_last_state_from_cache(aggregated_output['past_key_values'])
            num_layers = len(kv_cache.key_cache)
        
        self.set_sample(sample=True)
        for i in range(1, num_samples):
            sampled_output = self.single_forward(input_ids, **kwargs)

            # collect sample
            samples.append(sampled_output.logits.clone())

            # aggregating logits 
            aggregated_output.logits.add_(torch.softmax(sampled_output.logits, dim=-1))

            # aggregating the kv_cache
            if aggregated_output.past_key_values is not None:
                # since `forward` of the Attn module automatically updates the cache; 
                # need to pop the last state for kv_cache aggregation and update the cache.
                _, last_key, last_value = pop_last_state_from_cache(sampled_output.past_key_values)

                # aggregating the last key and value of the past_key_values.
                # past_key_values[batch_idx][:-1] are exactly the same; only aggregate the last one.
                for layer in range(num_layers):
                    # aggregated_last_key[layer] = (aggregated_last_key[layer]*i + last_key[layer]) / (i+1)
                    # aggregated_last_value[layer] = (aggregated_last_value[layer]*i + last_value[layer]) / (i+1)
                    aggregated_last_key[layer].add_(last_key[layer])
                    aggregated_last_value[layer].add_(last_value[layer])

            # aggregating hidden states.
            if aggregated_output.hidden_states is not None:
                aggregated_output.hidden_states[-1].add_(sampled_output.hidden_states[-1])
        
        # average the logits (prob. and then log back)
        aggregated_output.logits.div_(num_samples).log_()

        # average the kv_cache.
        if aggregated_output.past_key_values is not None:
            for layer in range(num_layers):
                aggregated_last_key[layer].div_(num_samples)
                aggregated_last_value[layer].div_(num_samples)
                kv_cache.key_cache[layer] = torch.cat([kv_cache.key_cache[layer], aggregated_last_key[layer]], dim=-2)
                kv_cache.value_cache[layer] = torch.cat([kv_cache.value_cache[layer], aggregated_last_value[layer]], dim=-2)

        # average the last hidden states.
        if aggregated_output.hidden_states is not None:
            aggregated_output.hidden_states[-1].div_(num_samples)

        # create new output with samples
        aggregated_output.samples = torch.stack(samples, dim=0)
        aggregated_output['samples'] = aggregated_output.samples
        return aggregated_output
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def single_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # create a new key
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.single_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# If we want to support other models, then we need to copy other classes from the original file.


__all__ = [
    "TFBLlamaConfig",
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
]
