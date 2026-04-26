import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


base = "/mnt/workspace/models/Qwen3.5-4B"
sft = "/mnt/workspace/LLaMA-Factory/saves/qwen35-4b/lora/catgirl_sft"
grpo_candidates = [
    "/mnt/workspace/grpo_train/catgirl_grpo_lora_output/final/grpo",
    "/mnt/workspace/grpo_train/catgirl_grpo_lora_output/final",
]
grpo = next((path for path in grpo_candidates if os.path.exists(path)), grpo_candidates[0])


def build_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "你是猫娘，请以猫娘的语气自然对话，不要输出思考过程。",
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main():
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载 base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("加载并合并 SFT LoRA...")
    model = PeftModel.from_pretrained(
        model,
        sft,
        adapter_name="sft",
    )
    model = model.merge_and_unload()

    print("加载 GRPO LoRA...")
    model = PeftModel.from_pretrained(
        model,
        grpo,
        adapter_name="grpo",
    )
    model.set_adapter("grpo")
    model.eval()

    print("\n模型加载完成。输入 exit 退出。\n")

    while True:
        user_text = input("User: ").strip()

        if user_text.lower() in {"exit", "quit", "q"}:
            break

        if not user_text:
            continue

        prompt = build_prompt(tokenizer, user_text)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                repetition_penalty=1.08,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        ).strip()

        response = response.replace("<think>", "").replace("</think>", "").strip()

        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
