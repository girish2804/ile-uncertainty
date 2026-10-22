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
import click
import logging
import json
import gc
import torch

import numpy as np
from datasets import load_dataset

from vllm import LLM, SamplingParams
import bayesian_transformer

# _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if _ROOT not in sys.path:
#     sys.path.insert(0, _ROOT)
from utils.config import Config

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_question_id_from_sample(sample):
    """Extract question_id from sample using the same logic as greedy_response_all"""
    if "unique_id" in sample:
        raw_id = sample["unique_id"]
    elif "metadata" in sample and "source_index" in sample["metadata"]:
        raw_id = sample["metadata"]["source_index"]
    else:
        return None
    
    question_id = str(raw_id).replace("/", "_").replace(".json", "")
    return question_id


def get_processed_question_ids(output_dir):
    """Scan output directory and return set of already processed question_ids"""
    processed_ids = set()
    
    if not os.path.exists(output_dir):
        return processed_ids
    
    try:
        for filename in os.listdir(output_dir):
            # Skip error files
            if filename.endswith("_error.json"):
                continue
            
            # Check for valid JSON files
            if filename.endswith(".json"):
                question_id = filename[:-5]  # Remove .json extension
                
                # Verify file is valid (not empty and contains outputs)
                file_path = os.path.join(output_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Check if file has valid outputs (not empty or error)
                        if isinstance(data, dict) and 'outputs' in data:
                            outputs = data.get('outputs', [])
                            # Only consider it processed if it has outputs and no error flag
                            if outputs and not data.get('error', False):
                                processed_ids.add(question_id)
                except (json.JSONDecodeError, IOError) as e:
                    logger.debug(f"Skipping invalid file {filename}: {e}")
                    continue
    except OSError as e:
        logger.warning(f"Error scanning output directory {output_dir}: {e}")
    
    return processed_ids


def filter_processed_samples(dataset, output_dir):
    """Filter out samples that have already been processed"""
    processed_ids = get_processed_question_ids(output_dir)
    
    if not processed_ids:
        logger.info("No previously processed samples found. Processing all samples.")
        return dataset
    
    logger.info(f"Found {len(processed_ids)} already processed samples in output directory")
    
    def should_process(sample):
        question_id = get_question_id_from_sample(sample)
        if question_id is None:
            # If we can't extract ID, process it (better safe than sorry)
            return True
        return question_id not in processed_ids
    
    # Filter dataset
    original_len = len(dataset)
    dataset = dataset.filter(should_process)
    filtered_len = len(dataset)
    skipped = original_len - filtered_len
    
    logger.info(f"Filtered dataset: {original_len} -> {filtered_len} samples ({skipped} already processed, {filtered_len} remaining)")
    
    return dataset


def extract_essential_data(outputs):
    """Extract only essential data needed for evaluation to reduce file size"""
    essential_outputs = []

    for output in outputs:
        # Extract uncertainty and logprob data
        try:
            unc_list = output.uncertainties
            ll_list = output.logprobs

            # Extract token-level data
            au_list = [list(d.values())[0].aleatoric_uncertainty for d in unc_list]
            tu_list = [list(d.values())[0].total_uncertainty for d in unc_list]
            eu_list = [list(d.values())[0].epistemic_uncertainty for d in unc_list]
            ll_list_total = [list(d.values())[0].logprob for d in ll_list]

            # Calculate aggregated values (same as extract_unc function)
            au_arr = np.array(au_list)
            tu_arr = np.array(tu_list)
            eu_arr = np.array(eu_list)
            ll_arr = np.array(ll_list_total)

            # Store essential data only
            essential_data = {
                'text': output.text,
                'uncertainties': {
                    'nau': float(au_arr.mean()),
                    'ntu': float(tu_arr.mean()),
                    'neu': float(eu_arr.mean()),
                    'au': float(au_arr.sum()),
                    'tu': float(tu_arr.sum()),
                    'eu': float(eu_arr.sum())
                },
                'logprobs': {
                    'nll': float(ll_arr.mean()),
                    'll': float(ll_arr.sum())
                }
            }
            essential_outputs.append(essential_data)

        except Exception as e:
            logger.warning(f"Error extracting data from output: {e}")
            # Save minimal data if extraction fails
            essential_data = {
                'text': output.text if hasattr(output, 'text') else "",
                'uncertainties': {},
                'logprobs': {},
                'perplexity': 0.0
            }
            essential_outputs.append(essential_data)

    return essential_outputs    

def greedy_response_all(sample, config, llm, sampling_params):
    question = sample.get('problem', sample.get('question'))
    if question is None:
        raise KeyError("Sample missing 'problem' or 'question' field required for prompting.")
    if "unique_id" in sample:
        raw_id = sample["unique_id"]
    elif "metadata" in sample and "source_index" in sample["metadata"]:
        raw_id = sample["metadata"]["source_index"]
    else:
        raise KeyError("Sample missing 'unique_id' or 'source_index' field required for identification.")

    question_id = str(raw_id).replace("/", "_").replace(".json", "")
    tokenizer = llm.get_tokenizer()
    system = [
        {
            "role": "system",
            "content": config.system_prompt,
        }
    ]
 
    prompt = tokenizer.apply_chat_template(
        system + [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sample['formatted_prompt'] = prompt
    
    # Handle large particle counts by splitting into batches
    max_particles_per_batch = 32
    total_particles = sampling_params.n
    
    try:
        if total_particles <= max_particles_per_batch:
            # Process normally for small particle counts
            res = llm.generate(prompt, sampling_params)
        else:
            # Process in batches for large particle counts
            logger.info(f"Question {question_id}: Processing {total_particles} particles in batches of {max_particles_per_batch}")
            all_outputs = []
            
            num_batches = (total_particles + max_particles_per_batch - 1) // max_particles_per_batch
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * max_particles_per_batch
                end_idx = min(start_idx + max_particles_per_batch, total_particles)
                batch_size = end_idx - start_idx
                
                logger.debug(f"Question {question_id}: Processing batch {batch_idx + 1}/{num_batches} with {batch_size} particles")
                
                # Create sampling params for this batch
                batch_sampling_params = SamplingParams(
                    n=batch_size,
                    temperature=sampling_params.temperature,
                    logprobs=sampling_params.logprobs,
                    max_tokens=sampling_params.max_tokens,
                    stop_token_ids=sampling_params.stop_token_ids,
                )
                
                try:
                    # Generate for this batch
                    batch_res = llm.generate(prompt, batch_sampling_params)
                    
                    # Extract outputs from this batch
                    if batch_res and len(batch_res) > 0:
                        all_outputs.extend(batch_res[0].outputs)
                    
                    # Clear GPU cache periodically to prevent memory buildup
                    if batch_idx % 4 == 0:  # Clear every 4 batches
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx + 1} for question {question_id}: {str(e)}")
                    # Continue with next batch instead of failing completely
                    continue
            
            # Create a combined result object
            if all_outputs:
                # Create a mock result object with all outputs combined
                from types import SimpleNamespace
                res = [SimpleNamespace(outputs=all_outputs, prompt=prompt)]
                logger.info(f"Question {question_id}: Successfully generated {len(all_outputs)} responses")
            else:
                logger.warning(f"No outputs generated for question_id: {question_id}")
                res = []
        
        # Save results in JSON format with essential data only
        if res and len(res) > 0 and hasattr(res[0], 'outputs'):
            # Extract essential data from outputs
            essential_outputs = extract_essential_data(res[0].outputs)
            essential_res = {
                'outputs': essential_outputs,
                'prompt': res[0].prompt if hasattr(res[0], 'prompt') else ""
            }
        else:
            essential_res = {'outputs': [], 'prompt': ""}

        output_file = f"{config.output_dir}/{question_id}.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(essential_res, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing question {question_id}: {str(e)}")
        # Save empty result to indicate processing was attempted
        with open(f"{config.output_dir}/{question_id}_error.json", "w", encoding='utf-8') as f:
            json.dump({'outputs': [], 'prompt': "", 'error': True}, f)
        raise e
    
    return sample

    # breakpoint()

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
    "--n-particles",
    default=1,
    type=int,
    help="Number of generations.",
    show_default=True,
)
@click.option(
    "--seed",
    default=96,
    type=int,
    help="Random seed for reproducibility.",
    show_default=True,
)

def main(
    dataset_start: int,
    dataset_end: int,
    output_dir: str,
    model_path: str,
    n_particles: int,
    seed: int,
    dataset_path: str,
    ):  
    # Setup logging based on particle count
    log_level = logging.DEBUG if n_particles > 64 else logging.INFO
    logging.getLogger().setLevel(log_level)
    
    # parser = H4ArgumentParser(Config)
    # config = parser.parse()
    config = Config()
    config.dataset_start = dataset_start
    config.dataset_end = dataset_end
    config.output_dir = output_dir
    config.model_path = model_path
        
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)

    # Log system information
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        logger.info(f"GPU Memory Available: {gpu_memory:.1f} GB")
        
        # Estimate memory requirements
        estimated_memory_per_particle = 0.1  # GB, rough estimate
        estimated_total_memory = n_particles * estimated_memory_per_particle
        logger.info(f"Estimated memory requirement: {estimated_total_memory:.1f} GB")
        
        if estimated_total_memory > gpu_memory * 0.8:  # 80% threshold
            logger.warning(f"High memory usage expected. Batch processing will be used.")

    num_gpus = 1
    logger.info(f"Initializing model with seed: {seed}")
    
    # Clear GPU cache before model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    llm = LLM(
        model=config.model_path,
        # max_num_batched_tokens=2048,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        seed=seed,
        tensor_parallel_size=num_gpus,
        enforce_eager=True,
    )


    if dataset_path is not None:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        # dataset = get_dataset(config)
        raise ValueError("Dataset path is required")

    print("Length of dataset:", len(dataset))
    
    # Log particle processing strategy
    if n_particles > 16:
        logger.info(f"Large particle count detected ({n_particles}). Will process in batches of 16 to avoid GPU memory issues.")
        logger.info(f"Expected total batches per sample: {(n_particles + 15) // 16}")
    else:
        logger.info(f"Processing {n_particles} particles in single batch.")

    sampling_params = SamplingParams(
        n=n_particles,
        temperature=0.8,
        logprobs=1,
        max_tokens=config.max_tokens,
        stop_token_ids=(
            [151645, 151643]
            if "qwen2" in config.model_path.lower()
            else None
    ),
    )
    
    dataset = dataset.select(range(config.dataset_start, config.dataset_end))
    
    # Filter out already processed samples to enable resume functionality
    dataset = filter_processed_samples(dataset, config.output_dir)
    
    if len(dataset) == 0:
        logger.info("All samples have already been processed. Nothing to do!")
        return
    
    logger.info(f"Starting processing of {len(dataset)} samples with {n_particles} particles each")
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if start_time:
        start_time.record()
    
    dataset = dataset.map(
        greedy_response_all,
        batched=False,
        batch_size=20,
        fn_kwargs={"config": config, "llm": llm, "sampling_params": sampling_params},
        desc="Running search",
        load_from_cache_file=False,
    )
    
    if end_time and start_time:
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Average time per sample: {elapsed_time / len(dataset):.2f} seconds")
        
        if n_particles > 16:
            total_batches = len(dataset) * ((n_particles + 15) // 16)
            logger.info(f"Total batches processed: {total_batches}")
            logger.info(f"Average time per batch: {elapsed_time / total_batches:.2f} seconds")
    
    # Final GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


    # save_steps = 50
    # splits = [dataset.select(range(i, i+save_steps)) for i in range(0, len(dataset), save_steps)]

    # print number of splits and length of each split
    # print("Number of splits:", len(splits))
    # print("Length of each split:", [len(split) for split in splits])

    # for i, dataset in enumerate(splits):
    #     print("--------"*20)
    #     print("Processing batch:", i)
    #     dataset = dataset.map(
    #         greedy_response,
    #         batched=False,
    #         batch_size=20,
    #         fn_kwargs={"config": config, "llm": llm,},
    #         desc="Running search",
    #         load_from_cache_file=False,
    #     )
    #     sampling_params = SamplingParams(
    #         n=n_particles,
    #         temperature=0.8,
    #         max_tokens=config.max_tokens,
    #         stop_token_ids=(
    #             [151645, 151643]
    #             if "qwen2" in config.model_path.lower()
    #             else None
    #         ),
    #     )

    #     res_lists = llm.generate(list(dataset['formatted_prompt']), sampling_params)  
    #     # response_texts = [res.outputs[0].text for res in res_lists] 
    #     # we save all responses
    #     response_texts = []
    #     for res in res_lists:
    #         response_texts.append([output.text for output in res.outputs])
    #     dataset = dataset.add_column('greedy_response', response_texts)

    #     dataset.to_json(f"{config.output_dir}/{config.dataset_name}_batch_{i}.jsonl", orient="records", lines=True)
    # save_dataset(dataset, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()

