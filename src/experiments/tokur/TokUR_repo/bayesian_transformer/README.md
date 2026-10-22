# Training-Free Bayesian Transformers
Supported Features:
- [x] Base Bayesian Transformer in HuggingFace Transformers
- [x] Bayesian Transformer for Causal LM in HuggingFace Transformers
- [x] Bayesian Transformer for Causal LM in vLLM

## Installation

In the `bayesian_transformer` directory, install the package:
```bash
pip install -e .
```

Then import the package:
```python
import bayesian_transformer
```
This will automatically register the models (see `bayesian_transformer/__init__.py` for details).

## Usage

### With `transformers`
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("n1h111sm/TFB-Llama3.2-1B-Instruct")
```

### With `vllm`

> **Note:** The currently supported version of vLLM is `0.7.3`. Please refer to the [forked version of vLLM](https://github.com/haizhou-shi/vllm).
>
> **Important:** The `enforce_eager` flag must be set to `True` to disable CUDAGraph compilation (default in vLLM v1), which would otherwise disable the Bayesian sampling process.
```python
from vllm import LLM
model = LLM(
    "n1h111sm/TFB-Llama3.2-1B-Instruct",
    enforce_eager=True,
)
```

A complete example:

```python
from vllm import LLM, SamplingParams

model = LLM(
    "n1h111sm/TFB-Llama3.2-1B-Instruct",
    enforce_eager=True,
)

prompt = "What is the meaning of life?"
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
    n=1,
    max_tokens=1024,
    logprobs=1,
)
output = model.generate(prompt, sampling_params)
print(output[0].outputs[0].text)
```