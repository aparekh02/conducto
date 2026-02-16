#!/usr/bin/env python3
"""Manage the inference server running on the VESSL workspace via SSH.

Simple commands to start, stop, check status, and view logs of the
TRT-LLM inference server on your A100 workspace.

Usage:
    python vessl_serve.py start   --engine_dir /root/conducto/engines/nemotron-trtllm
    python vessl_serve.py stop
    python vessl_serve.py status
    python vessl_serve.py logs
    python vessl_serve.py health
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
logger = logging.getLogger("vessl_serve")


def _get_ssh_config() -> tuple[str, str, int, str]:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / "nemotron_agent" / ".env")

    host = os.getenv("VESSL_SSH_HOST", "")
    key = os.getenv("VESSL_SSH_KEY_PATH", "~/.ssh/vessl_key")
    port = int(os.getenv("VESSL_SSH_PORT", "22"))
    remote_dir = os.getenv("VESSL_WORKSPACE_DIR", "/root/conducto")
    if not host:
        raise RuntimeError("VESSL_SSH_HOST not set in nemotron_agent/.env")
    return host, str(Path(key).expanduser()), port, remote_dir


def ssh_run(host: str, key: str, command: str, port: int = 22, capture: bool = False) -> str:
    cmd = [
        "ssh", "-i", key,
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        host, command,
    ]
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout
    subprocess.run(cmd, check=False)
    return ""


def cmd_start(args: argparse.Namespace) -> None:
    host, key, port, remote_dir = _get_ssh_config()
    engine_dir = args.engine_dir or f"{remote_dir}/engines/nemotron-trtllm"

    # Kill any existing server
    ssh_run(host, key, "pkill -f 'python.*serve.py' || true", port=port)

    logger.info("Starting serve.py on %s (port %d, tp=%d)...", host, args.port, args.tp_size)
    ssh_run(host, key, (
        f"cd {remote_dir} && nohup python scripts/serve.py "
        f"--engine_dir {engine_dir} "
        f"--port {args.port} "
        f"--tp_size {args.tp_size} "
        f"--max_context_len {args.max_context_len} "
        f"> serve.log 2>&1 &"
    ), port=port)
    ip = host.split("@")[-1]
    logger.info("Server starting at http://%s:%d/v1", ip, args.port)


def cmd_stop(args: argparse.Namespace) -> None:
    host, key, port, _ = _get_ssh_config()
    logger.info("Stopping inference server on %s...", host)
    ssh_run(host, key, "pkill -f 'python.*serve.py' || true", port=port)
    logger.info("Stopped")


def cmd_status(args: argparse.Namespace) -> None:
    host, key, port, _ = _get_ssh_config()
    out = ssh_run(host, key, "ps aux | grep 'serve.py' | grep -v grep || echo 'Not running'", port=port, capture=True)
    print(out.strip())

    out = ssh_run(host, key, "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable'", port=port, capture=True)
    print(f"GPU: {out.strip()}")


def cmd_logs(args: argparse.Namespace) -> None:
    host, key, port, remote_dir = _get_ssh_config()
    n = args.lines
    out = ssh_run(host, key, f"tail -n {n} {remote_dir}/serve.log 2>/dev/null || echo 'No logs yet'", port=port, capture=True)
    print(out)


def cmd_health(args: argparse.Namespace) -> None:
    host, key, port, _ = _get_ssh_config()
    ip = host.split("@")[-1]
    out = ssh_run(host, key, f"curl -s http://localhost:{args.port}/health 2>/dev/null || echo 'Server not responding'", port=port, capture=True)
    print(out.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage VESSL inference server")
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="Start the inference server")
    p_start.add_argument("--engine_dir")
    p_start.add_argument("--port", type=int, default=8000)
    p_start.add_argument("--tp_size", type=int, default=1)
    p_start.add_argument("--max_context_len", type=int, default=131072)

    sub.add_parser("stop", help="Stop the inference server")
    sub.add_parser("status", help="Check if server is running + GPU usage")

    p_logs = sub.add_parser("logs", help="View server logs")
    p_logs.add_argument("--lines", type=int, default=50)

    p_health = sub.add_parser("health", help="Check /health endpoint")
    p_health.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "logs": cmd_logs,
        "health": cmd_health,
    }[args.command](args)


if __name__ == "__main__":
    main()
