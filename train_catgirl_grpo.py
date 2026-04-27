import os
import json
import inspect
import torch
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from catgirl_reward import catgirl_reward_func



base_model_path = "/root/LLaMA-Factory/models/Qwen3.5-4B"
sft_lora_path = "/root/LLaMA-Factory/saves/qwen35-4b/lora/catgirl_sft"
grpo_json_path = "/root/LLaMA-Factory/data/catgirl_grpo.json"

# =========================
# 1. SwanLab 配置
# =========================

os.environ["SWANLAB_PROJECT"] = "catgirl-grpo"


# =========================
# 2. 加载 tokenizer
# =========================

print("加载 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


# =========================
# 3. 加载 base model
# =========================

print("加载 base model...")

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)


# =========================
# 4. 加载 SFT LoRA，并合并到 base
# =========================

print("加载 SFT LoRA，并合并到 base...")

model = PeftModel.from_pretrained(
    model,
    sft_lora_path,
    adapter_name="sft",
    is_trainable=False,
)
model = model.merge_and_unload()


# =========================
# 5. 在 base+SFT 上添加新的 GRPO LoRA
# =========================

print("添加 GRPO LoRA...")

grpo_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    target_modules="all-linear",
)

model = get_peft_model(model, grpo_lora_config, adapter_name="grpo")
model.set_adapter("grpo")
model.train()

# TRL + PEFT 兼容修复
model.warnings_issued = {}
if hasattr(model, "base_model"):
    model.base_model.warnings_issued = {}
if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
    model.base_model.model.warnings_issued = {}

model.print_trainable_parameters()


# =========================
# 6. 加载 GRPO 数据
# =========================

print("加载 GRPO 数据...")

with open(grpo_json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

def build_prompt(user_text: str) -> str:
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


records = []

for item in raw_data:
    for conv in item["conversations"]:
        if conv["from"] == "human":
            records.append(
                {
                    "prompt": build_prompt(conv["value"]),
                    "reference_output": item.get("reference_output", ""),
                    "type": item.get("type", ""),
                }
            )
            break

dataset = Dataset.from_list(records)

print(f"Loaded {len(dataset)} GRPO prompts")


# =========================
# 7. Debug 小样本
# =========================
debug_sample_size = int(os.getenv("DEBUG_SAMPLE_SIZE", "0"))
if debug_sample_size > 0:
    dataset = dataset.shuffle(seed=42).select(range(min(debug_sample_size, len(dataset))))

print(f"Using {len(dataset)} prompts for this run")


# =========================
# 8. 训练中抽样生成到 SwanLab
# =========================

sample_every_steps = int(os.getenv("SAMPLE_EVERY_STEPS", "50"))
sample_prompt_count = int(os.getenv("SAMPLE_PROMPT_COUNT", "3"))
sample_max_new_tokens = int(os.getenv("SAMPLE_MAX_NEW_TOKENS", "220"))
sample_seed = int(os.getenv("SAMPLE_SEED", "2026"))


def _input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


class SwanLabGenerationSampleCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        samples,
        every_steps: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        self.tokenizer = tokenizer
        self.samples = list(samples)
        self.every_steps = every_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if (
            model is None
            or self.every_steps <= 0
            or state.global_step == 0
            or state.global_step % self.every_steps != 0
            or not getattr(state, "is_world_process_zero", True)
        ):
            return control

        was_training = model.training
        model.eval()
        rows = []

        try:
            for index, sample in enumerate(self.samples, start=1):
                prompt = sample["prompt"]
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {key: value.to(_input_device(model)) for key, value in inputs.items()}

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                prompt_len = inputs["input_ids"].shape[-1]
                completion = self.tokenizer.decode(
                    output_ids[0][prompt_len:],
                    skip_special_tokens=True,
                ).strip()

                rows.append(
                    {
                        "index": index,
                        "prompt": prompt,
                        "completion": completion,
                        "reference_output": sample.get("reference_output", ""),
                        "type": sample.get("type", ""),
                    }
                )
        finally:
            if was_training:
                model.train()

        report = [f"GRPO generation samples at step {state.global_step}"]
        for row in rows:
            report.append(
                "\n".join(
                    [
                        f"\n--- sample {row['index']} | type={row['type']} ---",
                        "[prompt]",
                        row["prompt"],
                        "[completion]",
                        row["completion"],
                        "[reference]",
                        row["reference_output"],
                    ]
                )
            )

        text = "\n".join(report)
        print(text)

        try:
            import swanlab

            swanlab.log(
                {
                    "samples/grpo_generations": swanlab.Text(
                        text,
                        caption=f"step {state.global_step}",
                    )
                },
                step=state.global_step,
            )
        except Exception as exc:
            print(f"SwanLab sample logging skipped: {exc}")

        return control


sample_dataset = []
if sample_every_steps > 0 and sample_prompt_count > 0 and len(dataset) > 0:
    sample_dataset = dataset.shuffle(seed=sample_seed).select(
        range(min(sample_prompt_count, len(dataset)))
    )
    print(
        "SwanLab generation sampling enabled: "
        f"{len(sample_dataset)} prompts every {sample_every_steps} steps"
    )


# =========================
# 9. GRPO 参数
# =========================
output_dir = "./catgirl_grpo_lora_output"
use_vllm = os.getenv("USE_VLLM", "0") == "1"
vllm_mode = os.getenv("VLLM_MODE", "colocate")
vllm_gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.30"))
vllm_enable_sleep_mode = os.getenv("VLLM_ENABLE_SLEEP_MODE", "1") == "1"

grpo_config_kwargs = {
    "output_dir": output_dir,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-6,
    "logging_steps": 1,
    "save_steps": 50,
    "save_total_limit": 2,
    "num_generations": 4,
    "max_completion_length": 320,
    "temperature": 0.8,
    "top_p": 0.9,
    "bf16": True,
    "report_to": "swanlab",
    "run_name": "qwen35-4b-catgirl-sft-merged-grpo-lora",
}

if use_vllm:
    grpo_config_kwargs.update(
        {
            "use_vllm": True,
            "vllm_mode": vllm_mode,
            "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
            "vllm_enable_sleep_mode": vllm_enable_sleep_mode,
        }
    )

supported_grpo_args = set(inspect.signature(GRPOConfig).parameters)
grpo_config_kwargs = {
    key: value for key, value in grpo_config_kwargs.items()
    if key in supported_grpo_args
}

training_args = GRPOConfig(**grpo_config_kwargs)


# =========================
# 10. 创建 Trainer
# =========================

print("创建 GRPOTrainer...")

callbacks = []
if sample_dataset:
    callbacks.append(
        SwanLabGenerationSampleCallback(
            tokenizer=tokenizer,
            samples=sample_dataset,
            every_steps=sample_every_steps,
            max_new_tokens=sample_max_new_tokens,
            temperature=grpo_config_kwargs.get("temperature", 0.8),
            top_p=grpo_config_kwargs.get("top_p", 0.9),
        )
    )

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=catgirl_reward_func,
    processing_class=tokenizer,
    callbacks=callbacks,
)


# =========================
# 11. 开始训练
# =========================

print("开始 GRPO 训练...")

trainer.train()


# =========================
# 12. 保存 GRPO LoRA
# =========================

final_dir = os.path.join(output_dir, "final")

print(f"保存 GRPO LoRA 到: {final_dir}")

model.save_pretrained(final_dir, selected_adapters=["grpo"])
tokenizer.save_pretrained(final_dir)

print("GRPO 训练完成")
