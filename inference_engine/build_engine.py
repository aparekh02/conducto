#!/usr/bin/env python3
"""Compile Nemotron into a TensorRT-LLM engine via AutoDeploy.

AutoDeploy handles:
  - Automatic graph optimisation and layer fusion
  - KV-cache paging for long-context (up to 1M tokens)
  - FP8 / INT4-AWQ / NVFP4 quantisation
  - Multi-GPU tensor parallelism (TP=1,2,4,8)
  - In-flight batching configuration
  - CUDA Graph capture settings
  - Overlap scheduling for compute + communication

Multi-GPU + long-context guide:
  - 1 GPU  (A100-80GB):  up to 131K context with FP8
  - 2 GPUs (A100-80GB):  up to 262K context with FP8
  - 4 GPUs (A100-80GB):  up to 1M context with NVFP4
  - H100s give ~2x throughput per GPU vs A100

Runs ON the GPU server (VESSL A100/H100), not locally.

Usage:
    # Standard build (single A100, 131K context)
    python build_engine.py \
        --model_dir /models/nemotron-merged \
        --output_dir /engines/nemotron-trtllm \
        --quantization fp8

    # Multi-GPU build (4× A100, 1M context)
    python build_engine.py \
        --model_dir /models/nemotron-merged \
        --output_dir /engines/nemotron-trtllm-1m \
        --quantization nvfp4 \
        --tp_size 4 \
        --max_input_len 1048576 \
        --max_output_len 4096 \
        --max_batch_size 4

    # High-throughput build (batch-oriented)
    python build_engine.py \
        --model_dir /models/nemotron-merged \
        --output_dir /engines/nemotron-trtllm-batch \
        --max_batch_size 32 \
        --max_num_tokens 65536
"""

from __future__ import annotations

import argparse
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("build_engine")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile Nemotron to TensorRT-LLM engine with AutoDeploy"
    )
    parser.add_argument("--model_dir", required=True, help="Merged model directory")
    parser.add_argument("--output_dir", required=True, help="Engine output directory")

    # Quantisation
    parser.add_argument(
        "--quantization", default="fp8",
        choices=["none", "fp8", "int4_awq", "nvfp4"],
        help=(
            "Quantisation mode. "
            "fp8: best latency on A100/H100. "
            "nvfp4: max compression for 1M-token context. "
            "int4_awq: alternative 4-bit with activation-aware weights."
        ),
    )

    # Context and batching
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument(
        "--max_input_len", type=int, default=131072,
        help="Max input sequence length (131072=128K, 1048576=1M)",
    )
    parser.add_argument("--max_output_len", type=int, default=4096)
    parser.add_argument(
        "--max_num_tokens", type=int, default=32768,
        help="Max tokens across all in-flight requests (controls KV-cache budget)",
    )

    # Multi-GPU
    parser.add_argument(
        "--tp_size", type=int, default=1,
        help="Tensor parallel size (number of GPUs). "
             "TP=2 halves per-GPU memory, TP=4 enables 1M context on A100s",
    )
    parser.add_argument(
        "--pp_size", type=int, default=1,
        help="Pipeline parallel size (for very large models across many GPUs)",
    )

    # Optimisation flags
    parser.add_argument(
        "--enable_cuda_graphs", action="store_true", default=True,
        help="Capture CUDA Graphs for lower kernel launch latency (default: on)",
    )
    parser.add_argument(
        "--paged_kv_cache", action="store_true", default=True,
        help="Use paged KV-cache for efficient long-context memory (default: on)",
    )

    args = parser.parse_args()

    from tensorrt_llm import LLM, BuildConfig
    from tensorrt_llm.llmapi import QuantConfig

    # ── Quantisation config ──────────────────────────────
    quant_config = None
    if args.quantization == "fp8":
        quant_config = QuantConfig(quant_algo="fp8")
    elif args.quantization == "int4_awq":
        quant_config = QuantConfig(quant_algo="W4A16_AWQ")
    elif args.quantization == "nvfp4":
        quant_config = QuantConfig(quant_algo="NVFP4")

    # ── Build config ─────────────────────────────────────
    max_seq_len = args.max_input_len + args.max_output_len

    build_config = BuildConfig(
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_seq_len=max_seq_len,
        max_num_tokens=args.max_num_tokens,
    )

    logger.info("=" * 60)
    logger.info("TensorRT-LLM AutoDeploy Engine Build")
    logger.info("=" * 60)
    logger.info("  Model:           %s", args.model_dir)
    logger.info("  Output:          %s", args.output_dir)
    logger.info("  Quantisation:    %s", args.quantization)
    logger.info("  TP size:         %d GPU(s)", args.tp_size)
    logger.info("  PP size:         %d", args.pp_size)
    logger.info("  Max batch:       %d", args.max_batch_size)
    logger.info("  Max input len:   %d tokens", args.max_input_len)
    logger.info("  Max output len:  %d tokens", args.max_output_len)
    logger.info("  Max seq len:     %d tokens", max_seq_len)
    logger.info("  Max num tokens:  %d (in-flight budget)", args.max_num_tokens)
    logger.info("  CUDA Graphs:     %s", args.enable_cuda_graphs)
    logger.info("  Paged KV-cache:  %s", args.paged_kv_cache)

    if args.max_input_len >= 524288:
        logger.info("  ** Long-context build (>512K) — ensure TP >= 2 and NVFP4 **")
    if args.max_input_len >= 1048576:
        logger.info("  ** 1M-token build — recommend TP=4 with NVFP4 on A100-80GB **")

    logger.info("=" * 60)

    t0 = time.time()

    # AutoDeploy: single call handles everything
    #   - Converts HF model → TRT-LLM graph
    #   - Applies quantisation (FP8/NVFP4/INT4)
    #   - Configures paged KV-cache for long context
    #   - Sets up tensor parallelism across GPUs
    #   - Optimises with layer fusion, CUDA Graphs, overlap scheduling
    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=args.tp_size,
        pipeline_parallel_size=args.pp_size,
        quant_config=quant_config,
        build_config=build_config,
        enable_cuda_graph=args.enable_cuda_graphs,
    )

    llm.save(args.output_dir)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Engine compiled in %.1f minutes", elapsed / 60)
    logger.info("Output: %s", args.output_dir)
    logger.info(
        "Serve with: python serve.py --engine_dir %s --tp_size %d --max_context_len %d",
        args.output_dir, args.tp_size, max_seq_len,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
