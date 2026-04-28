#!/usr/bin/env python3
"""Interactive chat with the SFT or SFT+GRPO catgirl model."""

from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = "你是猫娘，请以猫娘的语气自然对话，不要输出思考过程。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a catgirl model.")
    parser.add_argument("--base-model", required=True, help="Base model or merged SFT model path.")
    parser.add_argument("--sft-lora", default="", help="Optional SFT LoRA path if base model is not already merged.")
    parser.add_argument("--grpo-lora", default="", help="Optional GRPO LoRA path.")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    return parser


def build_prompt(tokenizer, system_prompt: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


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

    if args.sft_lora:
        print(f"Loading SFT LoRA: {args.sft_lora}")
        model = PeftModel.from_pretrained(model, args.sft_lora, adapter_name="sft")
        model = model.merge_and_unload()

    if args.grpo_lora:
        print(f"Loading GRPO LoRA: {args.grpo_lora}")
        model = PeftModel.from_pretrained(model, args.grpo_lora, adapter_name="grpo")
        model.set_adapter("grpo")

    model.eval()
    print("\nReady. Type exit/quit/q to leave.\n")

    while True:
        user_text = input("User: ").strip()
        if user_text.lower() in {"exit", "quit", "q"}:
            break
        if not user_text:
            continue

        prompt = build_prompt(tokenizer, args.system_prompt, user_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        response_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        response = response.replace("<think>", "").replace("</think>", "").strip()
        print(f"\nAssistant: {response}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
