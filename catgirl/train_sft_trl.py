#!/usr/bin/env python3
"""Train catgirl SFT LoRA with TRL SFTTrainer."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pick_supported(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    supported = set(inspect.signature(cls).parameters)
    return {key: value for key, value in values.items() if key in supported}


def get_value(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any = None) -> Any:
    value = getattr(args, name)
    return config.get(name, default) if value is None else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRL SFT LoRA training for the catgirl model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-name-or-path", dest="model_name_or_path", default=None)
    parser.add_argument("--train-file", dest="train_file", default=None)
    parser.add_argument("--eval-file", dest="eval_file", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--max-length", dest="max_length", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", dest="per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", dest="per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", dest="gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--num-train-epochs", dest="num_train_epochs", type=float, default=None)
    parser.add_argument("--lora-r", dest="lora_r", type=int, default=None)
    parser.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", dest="lora_dropout", type=float, default=None)
    parser.add_argument("--logging-steps", dest="logging_steps", type=int, default=None)
    parser.add_argument("--save-steps", dest="save_steps", type=int, default=None)
    parser.add_argument("--save-total-limit", dest="save_total_limit", type=int, default=None)
    parser.add_argument("--report-to", dest="report_to", default=None)
    parser.add_argument("--run-name", dest="run_name", default=None)
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=None)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--packing", action="store_true", default=None)
    parser.add_argument("--no-packing", dest="packing", action="store_false")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)

    model_name_or_path = get_value(args, config, "model_name_or_path")
    train_file = get_value(args, config, "train_file", "data/catgirl_sft.jsonl")
    eval_file = get_value(args, config, "eval_file", "data/catgirl_eval.jsonl")
    output_dir = get_value(args, config, "output_dir", "saves/qwen35-4b/sft_lora")
    if not model_name_or_path:
        raise ValueError("--model-name-or-path is required")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = config["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    data_files = {"train": train_file}
    if eval_file and Path(eval_file).exists():
        data_files["eval"] = eval_file
    dataset = load_dataset("json", data_files=data_files)

    lora_config = LoraConfig(
        r=int(get_value(args, config, "lora_r", 16)),
        lora_alpha=int(get_value(args, config, "lora_alpha", 32)),
        lora_dropout=float(get_value(args, config, "lora_dropout", 0.05)),
        target_modules=config.get("lora_target_modules", "all-linear"),
        task_type=TaskType.CAUSAL_LM,
    )

    training_args = {
        "output_dir": output_dir,
        "max_length": int(get_value(args, config, "max_length", 2048)),
        "per_device_train_batch_size": int(get_value(args, config, "per_device_train_batch_size", 8)),
        "per_device_eval_batch_size": int(get_value(args, config, "per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(get_value(args, config, "gradient_accumulation_steps", 4)),
        "learning_rate": float(get_value(args, config, "learning_rate", 5e-5)),
        "num_train_epochs": float(get_value(args, config, "num_train_epochs", 1.0)),
        "lr_scheduler_type": config.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": float(config.get("warmup_ratio", 0.03)),
        "bf16": bool(config.get("bf16", True)),
        "logging_steps": int(get_value(args, config, "logging_steps", 10)),
        "save_steps": int(get_value(args, config, "save_steps", 200)),
        "save_total_limit": int(get_value(args, config, "save_total_limit", 2)),
        "packing": bool(get_value(args, config, "packing", True)),
        "gradient_checkpointing": bool(get_value(args, config, "gradient_checkpointing", False)),
        "report_to": get_value(args, config, "report_to", "swanlab"),
        "run_name": get_value(args, config, "run_name", "qwen35-4b-catgirl-sft-trl"),
        "eval_strategy": "steps" if "eval" in dataset else "no",
        "eval_steps": int(config.get("eval_steps", 200)),
    }
    sft_config = SFTConfig(**pick_supported(SFTConfig, training_args))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"] if "eval" in dataset else None,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SFT LoRA saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
