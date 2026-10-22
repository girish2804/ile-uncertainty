#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import os
import sys
import json
import pickle
import logging
import click
from tqdm import tqdm

from typing import List, Dict
from datasets import load_dataset

from vllm import LLM, SamplingParams
import bayesian_transformer

from utils.config import Config


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)    


def prepare_batch_prompts(samples: List[Dict], config: Config, tokenizer, prompt_type=None) -> List[str]:
    """
    Prepare a batch of prompts for inference
    
    Args:
        samples: List of samples to process
        config: Configuration object
        tokenizer: Model tokenizer
        qwen_prompt: Whether to use Qwen-specific prompt format
    
    Returns:
        List of formatted prompts
    """
    prompts = []
    
    if 'qwen' in prompt_type:
        first_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    else:
        first_prompt = config.system_prompt
    
    system = [
        {
            "role": "system",
            "content": first_prompt,
        }
    ]
    
    for sample in samples:
        question = sample.get("problem", sample.get("question"))
        if question is None:
            raise KeyError("Sample missing 'problem' or 'question' field required for prompting.")
        
        if 'qwen' in prompt_type:
            prompt = tokenizer.apply_chat_template(
                system + [{"role": "user", "content": f"{question} Let's think step by step and output the final answer within \\boxed{{}}."}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = tokenizer.apply_chat_template(
                system + [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(prompt)
    
    return prompts


def batch_inference(samples: List[Dict], config: Config, llm: LLM, sampling_params: SamplingParams, prompt_type=None) -> List[Dict]:
    """
    Perform batch inference on a list of samples
    
    Args:
        samples: List of samples to process
        config: Configuration object
        llm: vLLM model instance
        sampling_params: Sampling parameters for generation
        prompt_type: Whether to use Qwen-specific prompt format
    
    Returns:
        List of results with generation outputs
    """
    tokenizer = llm.get_tokenizer()
    
    # Prepare batch prompts
    prompts = prepare_batch_prompts(samples, config, tokenizer, prompt_type)
    
    # Perform batch generation
    results = llm.generate(prompts, sampling_params, use_tqdm=False)
    # Format results
    batch_results = []
    for sample, result in zip(samples, results):
        if 'unique_id' in sample:
            raw_id = sample['unique_id']
        elif 'metadata' in sample and 'source_index' in sample['metadata']:
            raw_id = sample['metadata']['source_index']
        else:
            raw_id = None

        if raw_id is None:
            raise KeyError("Sample missing 'unique_id' or 'source_idx' field required for identification.")
        question_id = str(raw_id).replace("/", "_").replace(".json", "")
        batch_results.append({
            "unique_id": question_id,
            "problem": sample.get('problem', sample.get('question')),
            "result": result,
            'answer': sample['answer'],
            'formatted_prompt': result.prompt,
            'old_math_orz_id': sample.get('old_math_orz_id', None),  # Keep original ID if available
            "level": sample.get("qwen2.5-3b-instruct-pass-at-10", None)  # Difficulty level if available
        })
    
    return batch_results


@click.command()
@click.option(
    "--dataset-path",
    default=None,
    type=str,
    help="Path to the dataset.",
    show_default=True,
)
@click.option(
    "--dataset-start",
    default=0,
    type=int,
    help="Start index of the dataset to process.",
    show_default=True,
)
@click.option(
    "--dataset-end",
    default=38,
    type=int,
    help="End index of the dataset to process.",
    show_default=True,
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False, writable=True),
    help="Output directory to save the results.",
)
@click.option(
    "--model-path",
    default="meta-llama/Llama-3.2-1B-Instruct",
    type=str,
    help="Path to the language model.",
    show_default=True,
)
@click.option(
    "--seed",
    default=96,
    type=int,
    help="Random seed for reproducibility.",
    show_default=True,
)
@click.option(
    "--batch-size",
    default=8,
    type=int,
    help="Batch size for inference.",
    show_default=True,
)
@click.option(
    "--save-every-n-batches",
    default=10,
    type=int,
    help="Save results every N batches.",
    show_default=True,
)
def main(
    dataset_start: int,
    dataset_end: int,
    output_dir: str,
    model_path: str,
    seed: int,
    dataset_path: str,
    batch_size: int,
    save_every_n_batches: int,
    ):  
    config = Config()
    config.dataset_start = dataset_start
    config.dataset_end = dataset_end
    config.output_dir = output_dir
    config.model_path = model_path

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)

    # GPU configuration
    num_gpus = 1
    print("SEED:", seed)
    print("BATCH_SIZE:", batch_size)
    
    with open(os.path.join(config.model_path, "config.json"), "r") as f:
        model_config = json.load(f)
        
    print("Sigma:", model_config['bayes_sigma'])
    
    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=seed,
        tensor_parallel_size=num_gpus,
        enforce_eager=True,
    )

    # Load dataset
    if dataset_path is not None:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        # dataset = get_dataset(config)
        raise ValueError("Dataset path is required")
        

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        logprobs=1,
        top_k=1,
        max_tokens=config.max_tokens,
        stop_token_ids=(
            [151645, 151643]
            if "qwen2" in config.model_path.lower()
            else None
        ),
    )

    # Select dataset range
    dataset = dataset.select(range(config.dataset_start, config.dataset_end))
    print("Length of dataset:", len(dataset))

    # Process dataset in batches
    all_results = []
    batch_save_counter = 0
    
    # Create batches
    total_samples = len(dataset)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    print(f"Processing {total_samples} samples in {num_batches} batches of size {batch_size}")
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        
        # Get current batch
        current_batch = dataset.select(range(start_idx, end_idx))
        batch_samples = [sample for sample in current_batch]
        
        # Perform batch inference
        try:
            batch_results = batch_inference(
                batch_samples,
                config=config,
                llm=llm,
                sampling_params=sampling_params,
                prompt_type=config.model_path.lower().split("/")[-1]  # Extract model name from path
            )
            
            all_results.extend(batch_results)
            
            logger.info(f"Processed batch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx-1})")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            # Continue with next batch
            continue
        
        # Save results periodically
        if (batch_idx + 1) % save_every_n_batches == 0 or batch_idx == num_batches - 1:
            start_id = config.dataset_start + batch_save_counter * save_every_n_batches * batch_size
            end_id = config.dataset_start + len(all_results)
            save_path = os.path.join(config.output_dir, f"batch_results_{start_id}_{end_id}.pkl")
            
            with open(save_path, "wb") as f:
                pickle.dump(all_results, f)
            
            logger.info(f"Saved {len(all_results)} results to {save_path}")
            
            # Reset for next batch save
            all_results = []
            batch_save_counter += 1

    logger.info("Batch inference completed 🔥!")


if __name__ == "__main__":
    main()
