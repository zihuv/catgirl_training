#!/usr/bin/env python3
"""Merge the SFT LoRA adapter into a standalone model for vLLM rollout."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a catgirl SFT LoRA into the base model.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--sft-lora", default="saves/qwen35-4b/sft_lora")
    parser.add_argument("--output-dir", default="saves/qwen35-4b/sft_merged")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.sft_lora, adapter_name="sft", is_trainable=False)
    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged SFT model saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
