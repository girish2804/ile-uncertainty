import transformers
from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoModelForCausalLM
from transformers.utils.generic import ModelOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from .layers import LowRankBayesianLinear, FullRankBayesianLinear
from .config import BayesianLMConfig

import torch
import torch.nn as nn
from typing import List

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

class BayesianLMModel(PreTrainedModel):
    config_class = BayesianLMConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.base_lm = AutoModel.from_pretrained(config.base_lm_path)
        self.config = config
        self.bayesian_layers = []
        self.modify_layers(self, config=config)
        self.num_samples = config.num_samples
        self.sample = config.sample

    def modify_layers(self, module, config):
        """modify the target layers of the base_lm to be Bayesian layers."""
        for name, child in module.named_children():
            if name.split('.')[-1] in config.target_modules and isinstance(child, nn.Linear):
                if self.config.lowrank:
                    bayesian_layer = LowRankBayesianLinear(
                        child,
                        bayes_sigma=config.bayes_sigma,
                        bayes_noise=config.bayes_noise,
                        basis_idx=config.basis_idx,
                    )
                else:
                    bayesian_layer = FullRankBayesianLinear(
                        child,
                        bayes_sigma=config.bayes_sigma,
                        bayes_noise=config.bayes_noise,
                    )
                setattr(module, name, bayesian_layer)
                self.bayesian_layers.append(bayesian_layer)
            self.modify_layers(child, config=config)
                
    def set_sample(self, sample=True):
        """set the sample status of all Bayesian layers."""
        self.sample = sample
        for layer in self.bayesian_layers:
            layer.set_sample(sample)
    
    def set_sigma(self, sigma=1e-3):
        """set the tfb standard deviation sigma of all Bayesian layers."""
        for layer in self.bayesian_layers:
            layer.set_sigma(sigma)

    def forward(self, input_ids, **kwargs):
        """forward pass of the BayesianLMModel passes everything through the base_lm."""
        if not self.sample:
            return self.base_lm(input_ids, **kwargs)
        
        num_samples = self.num_samples
        assert num_samples > 1, f"num_samples should be greater than when self.sample=True, while it's {self.num_samples}."

        # use the mean of the weights, and use the rest of the output as usual.
        self.set_sample(sample=False)
        aggregated_output = self.base_lm(input_ids, **kwargs)
        aggregated_output['logits'] = torch.softmax(aggregated_output['logits'], dim=-1)

        # aggregated_last_key and aggregated_last_value are the aggregated key and value of the past_key_values.
        # shape of .key_cache & .value_cache: 
        #   [num_layers, num_heads, seq_len, head_dim]
        # kv_cache points to the aggregated_output['past_key_values'], i.e., they are the same object.
        kv_cache, aggregated_last_key, aggregated_last_value = pop_last_state_from_cache(aggregated_output['past_key_values'])

        num_layers = len(kv_cache.key_cache)
        self.set_sample(sample=True)
        for i in range(1, num_samples):
            sampled_output = self.base_lm(input_ids, **kwargs)
            # since `forward` of the Attn module automatically updates the cache; 
            # need to pop the last state for kv_cache aggregation and update the cache.
            _, last_key, last_value = pop_last_state_from_cache(sampled_output['past_key_values'])
            
            # aggregating logits 
            aggregated_output['logits'].add_(torch.softmax(sampled_output['logits'], dim=-1))

            # aggregating the last key and value of the past_key_values.
            # past_key_values[batch_idx][:-1] are exactly the same; only aggregate the last one.
            for layer in range(num_layers):
                # aggregated_last_key[layer] = (aggregated_last_key[layer]*i + last_key[layer]) / (i+1)
                # aggregated_last_value[layer] = (aggregated_last_value[layer]*i + last_value[layer]) / (i+1)
                aggregated_last_key[layer].add_(last_key[layer])
                aggregated_last_value[layer].add_(last_value[layer])
            
            # del sampled_output
        
        aggregated_output['logits'].div_(num_samples).log_()
        for layer in range(num_layers):
            aggregated_last_key[layer].div_(num_samples)
            aggregated_last_value[layer].div_(num_samples)
            kv_cache.key_cache[layer] = torch.cat([kv_cache.key_cache[layer], aggregated_last_key[layer]], dim=-2)
            kv_cache.value_cache[layer] = torch.cat([kv_cache.value_cache[layer], aggregated_last_value[layer]], dim=-2)
        return aggregated_output


class BayesianForCausalLMModel(BayesianLMModel, GenerationMixin):
    config_class = BayesianLMConfig

    def __init__(self, config, *inputs, **kwargs):
        """The only difference is it initializes the base_lm with AutoModelForCausalLM."""
        super(BayesianLMModel, self).__init__(config, *inputs, **kwargs)

        self.base_lm = AutoModelForCausalLM.from_pretrained(config.base_lm_path)
        self.config = config
        self.bayesian_layers = []
        self.modify_layers(self, config=config)
        self.num_samples = config.num_samples
        self.sample = config.sample

        # when calling generation, the model will collect all the kwargs arguments 
        # as a set by inspect.signature(self.prepare_inputs_for_generation).parameters,
        # the original implementation is: 
        ############
        # def prepare_inputs_for_generation(self, *args, **kwargs):
        #     return self.base_lm.prepare_inputs_for_generation(*args, **kwargs)
        ############
        # unrecognized arguments such as "attention_mask" will cause the bug of `not used arguments error.`
        # setattr(self, 'prepare_inputs_for_generation', self.base_lm.prepare_inputs_for_generation)
    
    def get_input_embeddings(self):
        return self.base_lm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_lm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.base_lm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.base_lm.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.base_lm.set_decoder(decoder)

    def get_decoder(self):
        return self.base_lm.get_decoder()
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_lm.prepare_inputs_for_generation(*args, **kwargs)