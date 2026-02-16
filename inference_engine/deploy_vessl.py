#!/usr/bin/env python3
"""End-to-end deployment: merge → build → serve on VESSL workspace via SSH.

Orchestrates the full pipeline from LoRA adapters to a live inference
endpoint running on your VESSL A100:

  1. rsync scripts + model artifacts to the workspace
  2. (Optional) Merge LoRA adapters into base Nemotron model
  3. Compile TensorRT-LLM engine via AutoDeploy
  4. Start the inference server (serve.py) on the workspace

Run this from your Mac — it SSHes into the VESSL workspace.

Usage:
    # Full pipeline (adapters → live server)
    python deploy_vessl.py \
        --adapter_path ./output/sft-adapters \
        --quantization fp8

    # Skip merge (already have merged model on the workspace)
    python deploy_vessl.py --skip_merge --skip_build

    # Just start the server (engine already built)
    python deploy_vessl.py --serve_only
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("deploy_vessl")

SCRIPTS_DIR = Path(__file__).parent


def _get_ssh_config() -> tuple[str, str, int, str]:
    """Load SSH config from the agent .env."""
    sys.path.insert(0, str(SCRIPTS_DIR.parent))
    from dotenv import load_dotenv
    load_dotenv(SCRIPTS_DIR.parent / "nemotron_agent" / ".env")

    host = os.getenv("VESSL_SSH_HOST", "")
    key = os.getenv("VESSL_SSH_KEY_PATH", "~/.ssh/vessl_key")
    port = int(os.getenv("VESSL_SSH_PORT", "22"))
    remote_dir = os.getenv("VESSL_WORKSPACE_DIR", "/root/conducto")
    if not host:
        raise RuntimeError("VESSL_SSH_HOST not set in nemotron_agent/.env")
    return host, str(Path(key).expanduser()), port, remote_dir


def ssh_run(host: str, key: str, command: str, port: int = 22) -> None:
    cmd = [
        "ssh", "-i", key,
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        host, command,
    ]
    logger.info("SSH: %s", command)
    subprocess.run(cmd, check=True)


def rsync_to(host: str, key: str, local: str, remote: str, port: int = 22) -> None:
    cmd = [
        "rsync", "-avz",
        "-e", f"ssh -i {key} -p {port} -o StrictHostKeyChecking=no",
        local, f"{host}:{remote}",
    ]
    logger.info("Upload: %s → %s", local, remote)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy TRT-LLM engine to VESSL workspace")
    parser.add_argument("--base_model", default="nvidia/llama-3.3-nemotron-super-49b-v1")
    parser.add_argument("--adapter_path", help="Local path to LoRA adapters")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--serve_only", action="store_true", help="Just start the server")
    parser.add_argument("--quantization", default="fp8", choices=["none", "fp8", "int4_awq", "nvfp4"])
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_input_len", type=int, default=131072)
    parser.add_argument("--max_output_len", type=int, default=4096)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    host, key, port, remote_dir = _get_ssh_config()
    engine_dir = f"{remote_dir}/engines/nemotron-trtllm"
    model_dir = f"{remote_dir}/models/nemotron-merged"

    # Upload all inference scripts
    logger.info("Uploading inference engine scripts...")
    ssh_run(host, key, f"mkdir -p {remote_dir}/scripts {remote_dir}/models {remote_dir}/engines", port=port)
    for script in ["merge_adapters.py", "build_engine.py", "serve.py"]:
        rsync_to(host, key, str(SCRIPTS_DIR / script), f"{remote_dir}/scripts/", port=port)

    if args.serve_only:
        logger.info("Starting inference server...")
        ssh_run(host, key, (
            f"cd {remote_dir} && nohup python scripts/serve.py "
            f"--engine_dir {engine_dir} "
            f"--port {args.port} "
            f"--tp_size {args.tp_size} "
            f"--max_context_len {args.max_input_len + args.max_output_len} "
            f"> serve.log 2>&1 &"
        ), port=port)
        _print_done(host, args.port)
        return

    # Step 1: Merge adapters
    if not args.skip_merge:
        if not args.adapter_path:
            parser.error("--adapter_path required unless --skip_merge")
        logger.info("Step 1/3: Uploading and merging LoRA adapters...")
        rsync_to(host, key, args.adapter_path + "/", f"{remote_dir}/adapters/", port=port)
        ssh_run(host, key, (
            f"pip install -q peft transformers && "
            f"cd {remote_dir} && python scripts/merge_adapters.py "
            f"--base_model {args.base_model} "
            f"--adapter_path {remote_dir}/adapters "
            f"--output_dir {model_dir}"
        ), port=port)
        logger.info("Merge complete")
    else:
        logger.info("Step 1/3: Skipping merge")

    # Step 2: Build TRT-LLM engine
    if not args.skip_build:
        logger.info("Step 2/3: Building TensorRT-LLM engine...")
        ssh_run(host, key, (
            f"cd {remote_dir} && python scripts/build_engine.py "
            f"--model_dir {model_dir} "
            f"--output_dir {engine_dir} "
            f"--quantization {args.quantization} "
            f"--max_batch_size {args.max_batch_size} "
            f"--tp_size {args.tp_size} "
            f"--max_input_len {args.max_input_len} "
            f"--max_output_len {args.max_output_len}"
        ), port=port)
        logger.info("Engine build complete")
    else:
        logger.info("Step 2/3: Skipping build")

    # Step 3: Start server
    logger.info("Step 3/3: Starting inference server...")
    max_ctx = args.max_input_len + args.max_output_len
    ssh_run(host, key, (
        f"cd {remote_dir} && nohup python scripts/serve.py "
        f"--engine_dir {engine_dir} "
        f"--port {args.port} "
        f"--tp_size {args.tp_size} "
        f"--max_context_len {max_ctx} "
        f"> serve.log 2>&1 &"
    ), port=port)
    _print_done(host, args.port)


def _print_done(host: str, port: int) -> None:
    logger.info("=" * 60)
    logger.info("Inference server starting on %s:%d", host, port)
    logger.info("")
    logger.info("Update your .env to use it:")
    logger.info("  NVIDIA_NIM_BASE_URL=http://%s:%d/v1", host.split("@")[-1], port)
    logger.info("")
    logger.info("Check health:  curl http://%s:%d/health", host.split("@")[-1], port)
    logger.info("View logs:     ssh -i <key> %s 'tail -f %s/serve.log'", host, "/root/conducto")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
