#!/usr/bin/env python3
"""Merge LoRA adapters into the base Nemotron model.

After SFT or RL fine-tuning on VESSL, the output is a set of LoRA adapter
weights.  This script merges them back into the full model so TensorRT-LLM
can compile a single, optimised engine.

Runs ON the GPU server (VESSL A100), not locally.

Usage:
    python merge_adapters.py \
        --base_model nvidia/llama-3.3-nemotron-super-49b-v1 \
        --adapter_path /output/sft-adapters \
        --output_dir /models/nemotron-merged \
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("merge_adapters")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into base Nemotron model"
    )
    parser.add_argument("--base_model", required=True, help="HF model ID or local path")
    parser.add_argument("--adapter_path", required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--output_dir", required=True, help="Where to save merged model")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Load base model
    logger.info("Loading base model: %s", args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Load and merge LoRA adapters
    logger.info("Loading LoRA adapters from: %s", args.adapter_path)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    logger.info("Merging adapters into base model...")
    model = model.merge_and_unload()

    # Save merged model
    logger.info("Saving merged model to: %s", args.output_dir)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done. Merged model ready for TensorRT-LLM compilation.")


if __name__ == "__main__":
    main()
