#!/usr/bin/env python3
"""Inference + online-training server — base model + LoRA adapters via PEFT.

Loads the fine-tuned model with QLoRA and serves NIM-compatible
chat completions.  Also exposes a /train endpoint that does rapid
in-process gradient updates on the LoRA weights WITHOUT restarting.

Runs ON the GPU server (VESSL A100), not locally.

Usage:
    python peft_serve.py \
        --model_id nvidia/Nemotron-Mini-4B-Instruct \
        --adapter_path /root/conducto/output/sft-adapters \
        --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("peft_serve")

app = FastAPI(title="Conducto PEFT Inference + Online Training")

_MODEL = None
_TOKENIZER = None
_MODEL_INFO: dict = {}
_OPTIMIZER = None          # persistent AdamW for online LoRA updates
_TRAIN_STEPS: int = 0      # total gradient steps applied
_LOCK = threading.Lock()   # serialize train vs inference


# ── Request / Response models ────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "nemotron-finetuned"
    messages: list[ChatMessage]
    temperature: float = 0.6
    max_tokens: int = 4096


class TrainRequest(BaseModel):
    """Batch of traces for online LoRA gradient updates."""
    traces: list[dict]       # each: {"messages": [...], "raw_response": "..."}
    learning_rate: float = 1e-5
    max_steps: int = 0       # 0 = auto (1 step per trace)
    max_length: int = 1024


# ── Endpoints ────────────────────────────────────────────

@app.get("/health")
def health():
    if _MODEL is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "status": "ok",
        "model": _MODEL_INFO,
        "train_steps": _TRAIN_STEPS,
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if _MODEL is None:
        raise HTTPException(503, "Model not loaded")

    with _LOCK:
        _MODEL.eval()

        # Build prompt using the model's native chat template
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = _TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = _TOKENIZER(prompt, return_tensors="pt").to(_MODEL.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = _MODEL.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=max(req.temperature, 0.01),
                do_sample=req.temperature > 0,
                pad_token_id=_TOKENIZER.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

    new_tokens = outputs[0][prompt_tokens:]
    text = _TOKENIZER.decode(new_tokens, skip_special_tokens=True)
    completion_tokens = len(new_tokens)

    tok_s = completion_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
    logger.info(
        "Generated %d tokens in %.0fms (%.0f tok/s)  prompt=%d",
        completion_tokens, latency_ms, tok_s, prompt_tokens,
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ── Online training endpoint ─────────────────────────────

@app.post("/train")
def train_online(req: TrainRequest):
    """Do rapid LoRA gradient updates from recent traces.

    This runs a few gradient steps on the LoRA parameters only
    (base model stays frozen in 4-bit).  The optimizer persists
    between calls for momentum continuity.

    Typical call: 3-10 traces, 1 step per trace → finishes in ~2-5s.
    """
    global _OPTIMIZER, _TRAIN_STEPS

    if _MODEL is None:
        raise HTTPException(503, "Model not loaded")
    if not req.traces:
        return {"status": "ok", "steps": 0, "loss": 0.0}

    with _LOCK:
        # Switch to train mode and enable gradients on LoRA params
        _MODEL.train()
        for name, param in _MODEL.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        # ── Ensure optimizer exists ──
        if _OPTIMIZER is None:
            lora_params = [p for p in _MODEL.parameters() if p.requires_grad]
            if not lora_params:
                _MODEL.eval()
                return {"status": "error", "error": "No trainable LoRA parameters found"}
            _OPTIMIZER = torch.optim.AdamW(lora_params, lr=req.learning_rate)
            logger.info(
                "Created optimizer for %d LoRA parameters (%d tensors)",
                sum(p.numel() for p in lora_params),
                len(lora_params),
            )
        else:
            # Update learning rate if changed
            for pg in _OPTIMIZER.param_groups:
                pg["lr"] = req.learning_rate

        # ── Tokenize all traces into training examples ──
        examples = []
        for trace in req.traces:
            messages = trace.get("messages", [])
            raw_response = trace.get("raw_response", "")
            if not messages or not raw_response:
                continue

            # Build the full conversation including the assistant response
            full_messages = messages + [{"role": "assistant", "content": raw_response}]
            text = _TOKENIZER.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False,
            )
            tokens = _TOKENIZER(
                text,
                truncation=True,
                max_length=req.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            examples.append(tokens)

        if not examples:
            _MODEL.eval()
            return {"status": "ok", "steps": 0, "loss": 0.0, "reason": "no valid traces"}

        # ── Gradient steps ──
        max_steps = req.max_steps if req.max_steps > 0 else len(examples)
        total_loss = 0.0
        steps_done = 0

        t0 = time.perf_counter()
        for i in range(max_steps):
            ex = examples[i % len(examples)]
            input_ids = ex["input_ids"].to(_MODEL.device)
            attention_mask = ex["attention_mask"].to(_MODEL.device)
            labels = input_ids.clone()

            outputs = _MODEL(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            _OPTIMIZER.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in _MODEL.parameters() if p.requires_grad], 1.0
            )
            _OPTIMIZER.step()

            total_loss += loss.item()
            steps_done += 1
            _TRAIN_STEPS += 1

        train_ms = (time.perf_counter() - t0) * 1000
        avg_loss = total_loss / steps_done if steps_done > 0 else 0.0

        _MODEL.eval()

    logger.info(
        "Online train: %d steps in %.0fms, avg_loss=%.4f, total_steps=%d",
        steps_done, train_ms, avg_loss, _TRAIN_STEPS,
    )

    return {
        "status": "ok",
        "steps": steps_done,
        "loss": avg_loss,
        "total_steps": _TRAIN_STEPS,
        "train_ms": train_ms,
        "traces_used": len(examples),
    }


# ── Reload endpoint — swap adapters without full restart ─

@app.post("/reload")
def reload_adapters(adapter_path: str):
    """Hot-reload new LoRA adapters without restarting the server."""
    global _MODEL, _OPTIMIZER, _TRAIN_STEPS
    if _MODEL is None:
        raise HTTPException(503, "Base model not loaded")

    with _LOCK:
        logger.info("Reloading adapters from: %s", adapter_path)
        try:
            if hasattr(_MODEL, "unload"):
                base = _MODEL.unload()
            else:
                base = _MODEL

            _MODEL = PeftModel.from_pretrained(base, adapter_path)
            _MODEL.eval()
            # Reset optimizer since we have new weights
            _OPTIMIZER = None
            _TRAIN_STEPS = 0
            _MODEL_INFO["adapter_path"] = adapter_path
            _MODEL_INFO["reloaded_at"] = time.time()
            logger.info("Adapters reloaded successfully")
            return {"status": "ok", "adapter_path": adapter_path}
        except Exception as e:
            logger.error("Adapter reload failed: %s", e)
            raise HTTPException(500, f"Reload failed: {e}")


# ── Save checkpoint ──────────────────────────────────────

@app.post("/save")
def save_checkpoint(path: str = "/root/conducto/output/online-adapters"):
    """Save current LoRA weights to disk."""
    if _MODEL is None:
        raise HTTPException(503, "Model not loaded")

    with _LOCK:
        _MODEL.save_pretrained(path)
        _TOKENIZER.save_pretrained(path)

    logger.info("Checkpoint saved to %s (after %d online steps)", path, _TRAIN_STEPS)
    return {"status": "ok", "path": path, "train_steps": _TRAIN_STEPS}


# ── Model loading ────────────────────────────────────────

def load_model(model_id: str, adapter_path: str | None = None) -> None:
    global _MODEL, _TOKENIZER, _MODEL_INFO

    logger.info("Loading base model: %s", model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    _TOKENIZER = AutoTokenizer.from_pretrained(model_id)
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if adapter_path:
        logger.info("Loading LoRA adapters from: %s", adapter_path)
        _MODEL = PeftModel.from_pretrained(base_model, adapter_path)
        logger.info("Adapters loaded — fine-tuned model ready")
    else:
        # No pre-trained adapters — create fresh LoRA adapters for online training
        logger.info("No adapters provided — creating fresh LoRA for online training")
        base_model = prepare_model_for_kbit_training(base_model)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        _MODEL = get_peft_model(base_model, lora_config)
        _MODEL.print_trainable_parameters()

    _MODEL.eval()
    _MODEL_INFO = {
        "model_id": model_id,
        "adapter_path": adapter_path,
        "quantization": "4bit-nf4",
        "loaded_at": time.time(),
        "online_training": True,
    }
    logger.info("Model ready for inference + online training")


# ── Entrypoint ───────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PEFT inference + online training server")
    parser.add_argument("--model_id", default="nvidia/Nemotron-Mini-4B-Instruct")
    parser.add_argument("--adapter_path", default=None, help="Path to LoRA adapters")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.model_id, args.adapter_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
