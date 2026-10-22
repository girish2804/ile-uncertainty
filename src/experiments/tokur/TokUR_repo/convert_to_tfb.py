"""
Convert a base HuggingFace model to a TFB (Training-Free Bayesian) model.

This script computes SVD basis vectors for target attention layers (q_proj, v_proj)
and saves them alongside the original weights, producing a TFB model that can be
loaded directly by the bayesian_transformer package and vLLM.

Usage:
    python convert_to_tfb.py \
        --model-path /path/to/base-model \
        --output-path /path/to/output-tfb-model \
        --architecture llama \
        --rank 8
"""

import argparse
import json
import os
import shutil

import safetensors
import torch
from safetensors.torch import save_file


ARCHITECTURE_TO_MODEL_TYPE = {
    "llama": "tfb_llama",
    "qwen2": "tfb_qwen2",
}

ARCHITECTURE_TO_VLLM_CLASS = {
    "llama": "VllmTFBLLamaForCausalLM",
    "qwen2": "VllmTFBQwen2ForCausalLM",
}

TARGET_MODULES = ("q_proj", "v_proj")


def load_safetensors(path):
    """Load all safetensor files from a model directory into a single dict."""
    tensors = {}
    tensor_files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not tensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {path}")
    for filename in tensor_files:
        filepath = os.path.join(path, filename)
        with safetensors.safe_open(filepath, framework="pt") as f:
            for k in f.keys():
                if k in tensors:
                    raise ValueError(f"Duplicate key {k} found in {filename}")
                tensors[k] = f.get_tensor(k)
    return tensors, tensor_files


def compute_basis_vectors(model_tensors, rank, bayes_noise="right"):
    """Compute SVD basis vectors for target modules (q_proj, v_proj).

    For bayes_noise='right', stores left singular vectors (U[:, :rank]).
    For bayes_noise='left', stores right singular vectors (V[:, :rank]).
    """
    basis_vectors = {}
    basis_idx = list(range(rank))

    for key, weight in model_tensors.items():
        # Only process target module weights
        is_target = any(f"{mod}.weight" in key for mod in TARGET_MODULES)
        if not is_target:
            continue

        print(f"  Computing SVD for {key} (shape: {weight.shape})")
        old_dtype = weight.dtype
        U, _, V = torch.svd(weight.float())

        if bayes_noise == "right":
            bv = U[:, basis_idx].to(old_dtype)
        else:
            bv = V[:, basis_idx].to(old_dtype)

        bv_key = key.replace(".weight", ".basis_vectors")
        basis_vectors[bv_key] = bv

    return basis_vectors


def save_single_file_model(model_tensors, basis_vectors, output_path):
    """Save combined model as a single safetensors file."""
    combined = {**model_tensors, **basis_vectors}
    output_file = os.path.join(output_path, "model.safetensors")
    print(f"  Saving to {output_file}")
    save_file(combined, output_file, metadata={"format": "safetensors", "framework": "pt"})


def save_sharded_model(model_tensors, basis_vectors, output_path, source_path, source_tensor_files):
    """Save model in sharded format, adding basis vectors as an extra shard."""
    # Copy original shards
    for filename in source_tensor_files:
        src = os.path.join(source_path, filename)
        dst = os.path.join(output_path, filename)
        print(f"  Copying {filename}")
        shutil.copy2(src, dst)

    # Determine the new shard name
    num_shards = len(source_tensor_files) + 1
    new_shard_name = f"model-{num_shards:05d}-of-{num_shards:05d}.safetensors"

    # Rename existing shards in filenames (update the "of-NNNNN" part)
    for old_filename in source_tensor_files:
        old_path = os.path.join(output_path, old_filename)
        new_filename = old_filename.replace(
            f"-of-{len(source_tensor_files):05d}",
            f"-of-{num_shards:05d}",
        )
        new_path = os.path.join(output_path, new_filename)
        if old_path != new_path:
            os.rename(old_path, new_path)

    # Save basis vectors as the new shard
    new_shard_path = os.path.join(output_path, new_shard_name)
    print(f"  Saving basis vectors to {new_shard_name}")
    save_file(basis_vectors, new_shard_path, metadata={"format": "safetensors", "framework": "pt"})

    # Update the index file
    index_path = os.path.join(source_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)

    # Update existing shard references
    for key in index_data["weight_map"]:
        old_val = index_data["weight_map"][key]
        index_data["weight_map"][key] = old_val.replace(
            f"-of-{len(source_tensor_files):05d}",
            f"-of-{num_shards:05d}",
        )

    # Add basis vector entries
    for key in basis_vectors:
        index_data["weight_map"][key] = new_shard_name

    output_index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(output_index_path, "w") as f:
        json.dump(index_data, f, indent=4)


def create_tfb_config(source_path, output_path, architecture, rank, bayes_noise):
    """Create TFB config.json by extending the base model's config."""
    config_path = os.path.join(source_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    config["model_type"] = ARCHITECTURE_TO_MODEL_TYPE[architecture]
    config["architectures"] = [ARCHITECTURE_TO_VLLM_CLASS[architecture]]
    config["basis_idx"] = list(range(rank))
    config["bayes_noise"] = bayes_noise
    config["bayes_sigma"] = 0.1
    config["sample"] = True
    config["num_samples"] = 10
    config["lowrank"] = True
    config["init_basis_vectors"] = True
    config["use_tfb_for_generation"] = False

    output_config_path = os.path.join(output_path, "config.json")
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json with model_type={config['model_type']}")


def copy_tokenizer_files(source_path, output_path):
    """Copy tokenizer and other non-weight files from the source model."""
    skip_extensions = {".safetensors", ".bin", ".pt", ".pth"}
    skip_files = {"config.json", "model.safetensors.index.json"}

    for filename in os.listdir(source_path):
        if filename in skip_files:
            continue
        if any(filename.endswith(ext) for ext in skip_extensions):
            continue
        src = os.path.join(source_path, filename)
        if os.path.isfile(src):
            dst = os.path.join(output_path, filename)
            print(f"  Copying {filename}")
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a base HuggingFace model to a TFB model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the base HuggingFace model directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the converted TFB model",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=list(ARCHITECTURE_TO_MODEL_TYPE.keys()),
        help="Model architecture (llama or qwen2)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="Rank for low-rank basis vectors (default: 8)",
    )
    parser.add_argument(
        "--bayes-noise",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Noise direction: 'right' uses left singular vectors (U), "
             "'left' uses right singular vectors (V) (default: right)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    os.makedirs(args.output_path, exist_ok=True)

    # Step 1: Load model weights
    print("Loading model weights...")
    model_tensors, tensor_files = load_safetensors(args.model_path)
    print(f"  Loaded {len(model_tensors)} tensors from {len(tensor_files)} file(s)")

    # Step 2: Compute basis vectors
    print("Computing SVD basis vectors...")
    basis_vectors = compute_basis_vectors(
        model_tensors, rank=args.rank, bayes_noise=args.bayes_noise
    )
    print(f"  Computed {len(basis_vectors)} basis vector tensors")

    # Step 3: Save weights
    is_sharded = len(tensor_files) > 1
    print("Saving TFB model weights...")
    if is_sharded:
        save_sharded_model(
            model_tensors, basis_vectors, args.output_path,
            args.model_path, tensor_files,
        )
    else:
        save_single_file_model(model_tensors, basis_vectors, args.output_path)

    # Step 4: Create TFB config
    print("Creating TFB config...")
    create_tfb_config(
        args.model_path, args.output_path, args.architecture,
        args.rank, args.bayes_noise,
    )

    # Step 5: Copy tokenizer and other files
    print("Copying tokenizer files...")
    copy_tokenizer_files(args.model_path, args.output_path)

    print(f"\nDone! TFB model saved to {args.output_path}")


if __name__ == "__main__":
    main()
