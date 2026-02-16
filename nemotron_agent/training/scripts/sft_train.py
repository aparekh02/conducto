#!/usr/bin/env python3
"""Supervised fine-tuning script — runs ON the VESSL A100.

Takes the SFT JSONL (prompt/completion pairs from good grid decisions)
and fine-tunes with QLoRA adapters using transformers Trainer + peft.

Usage (called by VESSLRunner, not locally):
    python sft_train.py \
        --model_id nvidia/Nemotron-Mini-4B-Instruct \
        --data_path /data/sft_train.jsonl \
        --output_dir /output/sft-adapters \
        --epochs 1 --learning_rate 2e-5 --lora_rank 8
"""

from __future__ import annotations

import argparse
import json
import logging

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("sft_train")


def load_and_tokenize(path: str, tokenizer, max_length: int) -> Dataset:
    """Load the SFT JSONL and tokenize for causal LM training."""
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            # Use the tokenizer's native chat template for correct formatting
            text = tokenizer.apply_chat_template(
                rec["messages"], tokenize=False, add_generation_prompt=False,
            )
            records.append({"text": text})

    dataset = Dataset.from_list(records)

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    logger.info("Loading model %s", args.model_id)

    # 4-bit quantisation (QLoRA)
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
    model = prepare_model_for_kbit_training(model)

    # LoRA adapters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and tokenize data
    logger.info("Loading dataset from %s", args.data_path)
    dataset = load_and_tokenize(args.data_path, tokenizer, args.max_seq_length)
    logger.info("Dataset size: %d samples", len(dataset))

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info("Starting SFT training — %d epochs, %d samples", args.epochs, len(dataset))
    trainer.train()

    # Save LoRA adapters
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("SFT adapters saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
