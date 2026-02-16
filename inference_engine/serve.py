#!/usr/bin/env python3
"""Fast inference server — TensorRT-LLM engine with multi-GPU + long-context.

Features:
  - Multi-GPU tensor parallelism (TP=1,2,4 across A100/H100)
  - Up to 1M-token context windows with paged KV-cache
  - In-flight batching: new requests start while others are still generating
  - CUDA Graphs: captures and replays GPU kernels for minimal launch overhead
  - Overlap scheduling: compute + KV-cache transfers run in parallel
  - /health endpoint for VESSL Serve health-check routing
  - /v1/chat/completions endpoint (NIM-compatible)
  - /metrics endpoint for GPU utilisation + throughput stats

Runs ON the GPU server (VESSL A100/H100), not locally.

Usage:
    python serve.py \
        --engine_dir /engines/nemotron-trtllm \
        --tp_size 1 \
        --max_context_len 131072 \
        --port 8000
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
import uuid
from collections import deque
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("trtllm_serve")

app = FastAPI(title="Conducto TensorRT-LLM Inference", version="2.0")

# Globals set at startup
_LLM = None
_ENGINE_CONFIG: dict[str, Any] = {}
_METRICS = {
    "requests_total": 0,
    "tokens_generated": 0,
    "total_latency_ms": 0.0,
    "errors": 0,
    "active_requests": 0,
    "recent_latencies_ms": deque(maxlen=100),
}
_METRICS_LOCK = threading.Lock()


# ── Request / Response models ────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "nemotron-trtllm"
    messages: list[ChatMessage]
    temperature: float = 0.6
    max_tokens: int = 4096
    top_p: float = 0.9


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo


# ── Endpoints ────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check for VESSL Serve routing — returns 200 only when ready."""
    if _LLM is None:
        raise HTTPException(503, "Engine not loaded")
    return {
        "status": "ok",
        "engine": _ENGINE_CONFIG,
        "active_requests": _METRICS["active_requests"],
    }


@app.get("/metrics")
def metrics():
    """Throughput and latency stats for monitoring / autoscaling decisions."""
    with _METRICS_LOCK:
        recent = list(_METRICS["recent_latencies_ms"])
    avg_latency = sum(recent) / len(recent) if recent else 0
    p99_latency = sorted(recent)[int(len(recent) * 0.99)] if recent else 0
    total_tokens = _METRICS["tokens_generated"]
    total_time_s = _METRICS["total_latency_ms"] / 1000 if _METRICS["total_latency_ms"] > 0 else 1

    return {
        "requests_total": _METRICS["requests_total"],
        "tokens_generated": total_tokens,
        "errors": _METRICS["errors"],
        "active_requests": _METRICS["active_requests"],
        "avg_latency_ms": round(avg_latency, 1),
        "p99_latency_ms": round(p99_latency, 1),
        "throughput_tok_per_s": round(total_tokens / total_time_s, 1),
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> ChatResponse:
    if _LLM is None:
        raise HTTPException(503, "Engine not loaded")

    from tensorrt_llm import SamplingParams

    with _METRICS_LOCK:
        _METRICS["requests_total"] += 1
        _METRICS["active_requests"] += 1

    # Build prompt from messages
    prompt_parts = []
    for msg in req.messages:
        prompt_parts.append(f"<|{msg.role}|>\n{msg.content}")
    prompt_parts.append("<|assistant|>\n")
    prompt = "\n".join(prompt_parts)

    sampling = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )

    try:
        t0 = time.perf_counter()
        # TRT-LLM handles in-flight batching internally — concurrent
        # requests are batched automatically by the executor
        outputs = _LLM.generate([prompt], sampling_params=sampling)
        latency_ms = (time.perf_counter() - t0) * 1000

        generated_text = outputs[0].outputs[0].text
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(outputs[0].outputs[0].token_ids)

        with _METRICS_LOCK:
            _METRICS["tokens_generated"] += completion_tokens
            _METRICS["total_latency_ms"] += latency_ms
            _METRICS["recent_latencies_ms"].append(latency_ms)
            _METRICS["active_requests"] -= 1

        tok_per_s = completion_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
        logger.info(
            "Generated %d tokens in %.0fms (%.0f tok/s)  prompt=%d ctx=%d/%s",
            completion_tokens,
            latency_ms,
            tok_per_s,
            prompt_tokens,
            prompt_tokens + completion_tokens,
            _ENGINE_CONFIG.get("max_context_len", "?"),
        )

        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=generated_text),
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        with _METRICS_LOCK:
            _METRICS["errors"] += 1
            _METRICS["active_requests"] -= 1
        logger.error("Generation failed: %s", e)
        raise HTTPException(500, f"Generation failed: {e}")


# ── Engine loading ───────────────────────────────────────

def load_engine(
    engine_dir: str,
    tp_size: int = 1,
    max_context_len: int = 131072,
    enable_cuda_graphs: bool = True,
) -> None:
    """Load TRT-LLM engine with multi-GPU and long-context config."""
    global _LLM, _ENGINE_CONFIG
    from tensorrt_llm import LLM

    _ENGINE_CONFIG = {
        "engine_dir": engine_dir,
        "tp_size": tp_size,
        "max_context_len": max_context_len,
        "cuda_graphs": enable_cuda_graphs,
    }

    logger.info("Loading TensorRT-LLM engine")
    logger.info("  Engine dir:      %s", engine_dir)
    logger.info("  TP size:         %d GPU(s)", tp_size)
    logger.info("  Max context:     %d tokens", max_context_len)
    logger.info("  CUDA Graphs:     %s", enable_cuda_graphs)
    logger.info("  In-flight batch: enabled (automatic)")
    logger.info("  Paged KV-cache:  enabled (automatic)")

    t0 = time.time()

    # TensorRT-LLM LLM class handles:
    #   - Tensor parallelism across tp_size GPUs via NCCL
    #   - Paged KV-cache for efficient long-context memory
    #   - In-flight batching via the executor
    #   - CUDA Graph capture for kernel launch optimization
    #   - Overlap scheduling for compute + memory transfers
    _LLM = LLM(
        model=engine_dir,
        tensor_parallel_size=tp_size,
        max_seq_len=max_context_len,
        enable_cuda_graph=enable_cuda_graphs,
    )

    elapsed = time.time() - t0
    logger.info("Engine loaded in %.1fs — ready to serve", elapsed)


# ── Entrypoint ───────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TensorRT-LLM inference server")
    parser.add_argument("--engine_dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--tp_size", type=int, default=1,
        help="Tensor parallel size (must match engine build)",
    )
    parser.add_argument(
        "--max_context_len", type=int, default=131072,
        help="Max context length in tokens (up to 1048576 for 1M)",
    )
    parser.add_argument(
        "--no_cuda_graphs", action="store_true",
        help="Disable CUDA Graph capture (debug only)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Uvicorn worker count (typically 1 — TRT-LLM handles parallelism internally)",
    )
    args = parser.parse_args()

    load_engine(
        engine_dir=args.engine_dir,
        tp_size=args.tp_size,
        max_context_len=args.max_context_len,
        enable_cuda_graphs=not args.no_cuda_graphs,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
