#!/usr/bin/env python3
"""Train catgirl GRPO LoRA with TRL and optional vLLM generation."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from catgirl_reward import get_reward_funcs


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
    parser = argparse.ArgumentParser(description="TRL GRPO training for catgirl LoRA.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-name-or-path", dest="model_name_or_path", default=None)
    parser.add_argument("--train-file", dest="train_file", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--debug-sample-size", dest="debug_sample_size", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", dest="per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", dest="gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--num-train-epochs", dest="num_train_epochs", type=float, default=None)
    parser.add_argument("--num-generations", dest="num_generations", type=int, default=None)
    parser.add_argument("--max-prompt-length", dest="max_prompt_length", type=int, default=None)
    parser.add_argument("--max-completion-length", dest="max_completion_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", dest="top_p", type=float, default=None)
    parser.add_argument("--lora-r", dest="lora_r", type=int, default=None)
    parser.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", dest="lora_dropout", type=float, default=None)
    parser.add_argument("--logging-steps", dest="logging_steps", type=int, default=None)
    parser.add_argument("--save-steps", dest="save_steps", type=int, default=None)
    parser.add_argument("--save-total-limit", dest="save_total_limit", type=int, default=None)
    parser.add_argument("--report-to", dest="report_to", default=None)
    parser.add_argument("--run-name", dest="run_name", default=None)
    parser.add_argument("--use-vllm", dest="use_vllm", action="store_true", default=None)
    parser.add_argument("--no-vllm", dest="use_vllm", action="store_false")
    parser.add_argument("--vllm-mode", dest="vllm_mode", choices=["server", "colocate"], default=None)
    parser.add_argument("--vllm-server-base-url", dest="vllm_server_base_url", default=None)
    parser.add_argument("--vllm-server-host", dest="vllm_server_host", default=None)
    parser.add_argument("--vllm-server-port", dest="vllm_server_port", type=int, default=None)
    parser.add_argument("--vllm-group-port", dest="vllm_group_port", type=int, default=None)
    parser.add_argument("--vllm-gpu-memory-utilization", dest="vllm_gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--vllm-tensor-parallel-size", dest="vllm_tensor_parallel_size", type=int, default=None)
    parser.add_argument("--vllm-enable-sleep-mode", dest="vllm_enable_sleep_mode", action="store_true", default=None)
    parser.add_argument("--no-vllm-enable-sleep-mode", dest="vllm_enable_sleep_mode", action="store_false")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)

    model_name_or_path = get_value(args, config, "model_name_or_path")
    train_file = get_value(args, config, "train_file", "data/catgirl_grpo.jsonl")
    output_dir = get_value(args, config, "output_dir", "saves/qwen35-4b/grpo_lora")
    if not model_name_or_path:
        raise ValueError("--model-name-or-path is required. Use the merged SFT model path.")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = load_dataset("json", data_files={"train": train_file})["train"]
    debug_sample_size = int(get_value(args, config, "debug_sample_size", 0))
    if debug_sample_size > 0:
        dataset = dataset.shuffle(seed=int(config.get("seed", 42))).select(range(min(debug_sample_size, len(dataset))))
    print(f"Using {len(dataset)} GRPO prompts")

    lora_config = LoraConfig(
        r=int(get_value(args, config, "lora_r", 16)),
        lora_alpha=int(get_value(args, config, "lora_alpha", 32)),
        lora_dropout=float(get_value(args, config, "lora_dropout", 0.05)),
        target_modules=config.get("lora_target_modules", "all-linear"),
        task_type=TaskType.CAUSAL_LM,
    )

    vllm_mode = get_value(args, config, "vllm_mode", "colocate")
    use_vllm = bool(get_value(args, config, "use_vllm", True))
    vllm_server_host = get_value(args, config, "vllm_server_host", "0.0.0.0")
    vllm_server_port = int(get_value(args, config, "vllm_server_port", 8000))
    vllm_group_port = int(get_value(args, config, "vllm_group_port", 51216))
    if use_vllm and vllm_mode == "server":
        os.environ.setdefault("MASTER_ADDR", str(vllm_server_host))
        os.environ.setdefault("MASTER_PORT", str(vllm_group_port))

    grpo_args = {
        "output_dir": output_dir,
        "num_train_epochs": float(get_value(args, config, "num_train_epochs", 1.0)),
        "per_device_train_batch_size": int(get_value(args, config, "per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(get_value(args, config, "gradient_accumulation_steps", 4)),
        "learning_rate": float(get_value(args, config, "learning_rate", 1.5e-6)),
        "lr_scheduler_type": config.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": float(config.get("warmup_ratio", 0.03)),
        "logging_steps": int(get_value(args, config, "logging_steps", 5)),
        "save_steps": int(get_value(args, config, "save_steps", 100)),
        "save_total_limit": int(get_value(args, config, "save_total_limit", 2)),
        "num_generations": int(get_value(args, config, "num_generations", 4)),
        "max_prompt_length": int(get_value(args, config, "max_prompt_length", 1024)),
        "max_completion_length": int(get_value(args, config, "max_completion_length", 320)),
        "temperature": float(get_value(args, config, "temperature", 0.8)),
        "top_p": float(get_value(args, config, "top_p", 0.9)),
        "bf16": bool(config.get("bf16", False)),
        "fp16": bool(config.get("fp16", True)),
        "beta": float(config.get("beta", 0.0)),
        "loss_type": config.get("loss_type", "dr_grpo"),
        "mask_truncated_completions": bool(config.get("mask_truncated_completions", True)),
        "report_to": get_value(args, config, "report_to", "swanlab"),
        "run_name": get_value(args, config, "run_name", "qwen35-4b-catgirl-grpo-vllm"),
        "use_vllm": use_vllm,
        "vllm_mode": vllm_mode,
        "chat_template_kwargs": config.get("chat_template_kwargs", {"enable_thinking": False}),
        "reward_weights": config.get("reward_weights", [1.0, 1.0, 1.0, 1.0]),
    }
    if use_vllm and vllm_mode == "server":
        grpo_args.update(
            {
                "vllm_server_host": vllm_server_host,
                "vllm_server_port": vllm_server_port,
                "vllm_group_port": vllm_group_port,
                "vllm_server_base_url": get_value(args, config, "vllm_server_base_url", None),
            }
        )
    elif use_vllm and vllm_mode == "colocate":
        grpo_args.update(
            {
                "vllm_gpu_memory_utilization": float(get_value(args, config, "vllm_gpu_memory_utilization", 0.28)),
                "vllm_tensor_parallel_size": int(get_value(args, config, "vllm_tensor_parallel_size", 1)),
                "vllm_enable_sleep_mode": bool(get_value(args, config, "vllm_enable_sleep_mode", True)),
            }
        )
    training_args = GRPOConfig(**pick_supported(GRPOConfig, grpo_args))

    trainer = GRPOTrainer(
        model=model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=get_reward_funcs(),
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"GRPO LoRA saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
