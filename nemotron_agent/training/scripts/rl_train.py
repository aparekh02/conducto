#!/usr/bin/env python3
"""RL fine-tuning script — runs ON the VESSL A100.

Supports two modes:
  --method dpo   Direct Preference Optimisation (needs chosen/rejected pairs)
  --method reward   Reward-weighted regression on (prompt, response, reward)

Usage (called by VESSLRunner, not locally):
    python rl_train.py \
        --model_id nvidia/llama-3.3-nemotron-super-49b-v1 \
        --data_path /data/dpo_train.jsonl \
        --output_dir /output/rl-adapters \
        --method dpo --epochs 1 --learning_rate 5e-6 --lora_rank 16
"""

from __future__ import annotations

import argparse
import json
import logging

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("rl_train")


# ── Data loaders ─────────────────────────────────────────

def _messages_to_text(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        parts.append(f"<|{msg['role']}|>\n{msg['content']}")
    return "\n".join(parts)


def load_dpo_dataset(path: str) -> Dataset:
    """Load DPO JSONL: {prompt, chosen, rejected}."""
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records.append({
                "prompt": _messages_to_text(rec["prompt"]),
                "chosen": rec["chosen"],
                "rejected": rec["rejected"],
            })
    return Dataset.from_list(records)


def load_reward_dataset(path: str) -> Dataset:
    """Load reward JSONL: {prompt, response, reward}.
    Converts to DPO pairs by splitting on median reward.
    """
    raw = []
    with open(path) as f:
        for line in f:
            raw.append(json.loads(line))

    raw.sort(key=lambda x: x["reward"], reverse=True)
    mid = len(raw) // 2
    good = raw[:mid]
    bad = raw[mid:]

    records = []
    for g, b in zip(good, bad):
        records.append({
            "prompt": _messages_to_text(g["prompt"]),
            "chosen": g["response"],
            "rejected": b["response"],
        })
    return Dataset.from_list(records)


# ── Main ─────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", choices=["dpo", "reward"], default="dpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    logger.info("Loading model %s for %s training", args.model_id, args.method)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Load dataset
    if args.method == "dpo":
        dataset = load_dpo_dataset(args.data_path)
    else:
        dataset = load_reward_dataset(args.data_path)

    logger.info("Dataset size: %d pairs", len(dataset))

    # DPO training config
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        beta=0.1,
    )

    # DPO needs a reference model — use the same base (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting %s training — %d epochs", args.method.upper(), args.epochs)
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("RL adapters saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
