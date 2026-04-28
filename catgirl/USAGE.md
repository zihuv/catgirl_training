# Catgirl TRL Training User Guide

这是单卡版本的使用手册。流程是：

1. 准备 JSONL 数据
2. 用 TRL 做 SFT LoRA
3. 合并 SFT LoRA，得到 vLLM 可加载的 SFT 模型
4. 用 TRL GRPOTrainer + vLLM colocate 训练 GRPO LoRA
5. 试聊验证 SFT 或 SFT+GRPO 效果

## 目录结构

```text
catgirl/
  NekoQA-30K.json
  prepare_catgirl_data.py
  train_sft_trl.py
  merge_sft_lora.py
  train_grpo_trl_vllm.py
  catgirl_reward.py
  chat_catgirl.py
  configs/
    sft_qwen35_4b.json
    grpo_qwen35_4b_vllm.json
  data/
    catgirl_sft.jsonl
    catgirl_grpo.jsonl
    catgirl_eval.jsonl
  saves/
    qwen35-4b/
      sft_lora/
      sft_merged/
      grpo_lora/
```

## 环境安装

建议新建干净环境：

```bash
conda create -n catgirl python=3.11 -y
conda activate catgirl
```

安装 PyTorch 时按你的云端 CUDA/驱动版本选择官方命令。下面只是 CUDA 12.1 示例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Qwen3.5：

```bash
pip install "transformers>=5.2.0" "datasets>=4.0.0" "accelerate>=1.10.0" peft swanlab modelscope
pip install -U trl
uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

如果没有安装 `uv`：

```bash
pip install uv
```

安装后检查：

```bash
python -c "import torch, trl, vllm; print(torch.__version__, trl.__version__, vllm.__version__)"
python -c "from trl import SFTTrainer, GRPOTrainer; print('TRL ok')"
```


## 准备模型路径

把基础模型放到一个固定目录，例如：

```text
/root/models/Qwen3.5-4B
```

然后修改：

```text
configs/sft_qwen35_4b.json
```

把 `model_name_or_path` 改成你的实际模型路径。


## 准备数据

在 `catgirl/` 目录下执行：

```bash
python prepare_catgirl_data.py \
  --input NekoQA-30K.json \
  --output-dir data \
  --sft-size 20000 \
  --grpo-size 1000 \
  --eval-size 500
```

输出文件：

```text
data/catgirl_sft.jsonl
data/catgirl_grpo.jsonl
data/catgirl_eval.jsonl
```

如果想保留所有短任务样本，不做弱任务样本过滤：

```bash
python prepare_catgirl_data.py \
  --input NekoQA-30K.json \
  --output-dir data \
  --sft-size 20000 \
  --grpo-size 1000 \
  --eval-size 500 \
  --keep-weak-task-samples
```

## 训练 SFT

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 python train_sft_trl.py \
  --config configs/sft_qwen35_4b.json
```

如果报：

```text
Your setup doesn't support bf16/gpu
```

使用当前默认配置即可，它已经改成 `fp16`：

```json
"bf16": false,
"fp16": true,
"model_dtype": "float16"
```

如果你的显卡和 PyTorch 明确支持 bf16，再把它改回：

```json
"bf16": true,
"fp16": false,
"model_dtype": "bfloat16"
```

如果显存够，可以在 `configs/sft_qwen35_4b.json` 里调大：

```text
per_device_train_batch_size
```

SFT 输出：

```text
saves/qwen35-4b/sft_lora
```

## 合并 SFT LoRA

vLLM 建议加载合并后的 SFT 模型：

```bash
python merge_sft_lora.py \
  --base-model /root/models/Qwen3.5-4B \
  --sft-lora saves/qwen35-4b/sft_lora \
  --output-dir saves/qwen35-4b/sft_merged
```

输出：

```text
saves/qwen35-4b/sft_merged
```

## 训练 GRPO

单卡默认使用 vLLM colocate，也就是训练和 vLLM 生成共用同一张显卡。不需要单独启动 `trl vllm-serve`。

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch train_grpo_trl_vllm.py \
  --config configs/grpo_qwen35_4b_vllm.json
```

先小样本试跑：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch train_grpo_trl_vllm.py \
  --config configs/grpo_qwen35_4b_vllm.json \
  --debug-sample-size 20
```

再跑 100 条观察：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch train_grpo_trl_vllm.py \
  --config configs/grpo_qwen35_4b_vllm.json \
  --debug-sample-size 100
```

如果 100 条效果稳定，再跑全量。

GRPO 输出：

```text
saves/qwen35-4b/grpo_lora
```

如果单卡 colocate 显存不足，先把配置里的显存占用调低：

```json
"vllm_gpu_memory_utilization": 0.22
```

如果仍然 OOM，再临时关掉 vLLM 兜底：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch train_grpo_trl_vllm.py \
  --config configs/grpo_qwen35_4b_vllm.json \
  --no-vllm \
  --debug-sample-size 20
```

关掉 vLLM 会慢很多，但可以确认训练逻辑和 reward 没问题。

## 试聊

只测 SFT：

```bash
python chat_catgirl.py \
  --base-model saves/qwen35-4b/sft_merged
```

测 SFT + GRPO：

```bash
python chat_catgirl.py \
  --base-model saves/qwen35-4b/sft_merged \
  --grpo-lora saves/qwen35-4b/grpo_lora
```

如果没有合并 SFT，也可以这样测：

```bash
python chat_catgirl.py \
  --base-model /root/models/Qwen3.5-4B \
  --sft-lora saves/qwen35-4b/sft_lora \
  --grpo-lora saves/qwen35-4b/grpo_lora
```

## 推荐评测 Prompt

每次 SFT/GRPO 后，固定测这些：

```text
你好
你是谁
解释一下什么是过拟合
帮我写一个 Python 快排
算一下 37 * 48
我今天焦虑得睡不着怎么办
不要撒娇，认真回答
你是人工智能吗？
我租房被二房东坑了，怎么要水电明细？
```

重点看四件事：

```text
猫娘风格是否明显
任务有没有答准
动作和口癖是否过度
有没有输出 <think> 或自称 AI/Qwen
```

## 常用调参

SFT 想更快：

```text
per_device_train_batch_size 调大
gradient_accumulation_steps 调小
packing 保持 true
```

GRPO 想更快：

```text
优先确认 vLLM colocate 正常工作
logging_steps 从 5 调到 10
max_completion_length 从 320 调到 256
num_generations 从 4 调到 3
```

单卡 vLLM colocate 想更稳：

```text
vllm_gpu_memory_utilization 先用 0.28
OOM 就降到 0.22
稳定后可试 0.32 或 0.35
per_device_train_batch_size 默认 1
显存很宽裕再试 2
```

GRPO 猫娘味不够：

```text
先不要加 epoch
优先增加 GRPO 数据里的 ACG文化、日常闲聊、心理疗愈比例
再考虑提高 catgirl_reward.py 里的风格奖励权重
```

GRPO 任务能力下降：

```text
先停全量训练
用 --debug-sample-size 100 重新验证
降低 max_completion_length
提高任务类数据比例
检查 reward/task_completion_reward 和 reference_reward 的日志
```

## 推荐运行顺序

```bash
python prepare_catgirl_data.py --input NekoQA-30K.json --output-dir data --sft-size 20000 --grpo-size 1000 --eval-size 500

CUDA_VISIBLE_DEVICES=0 python train_sft_trl.py --config configs/sft_qwen35_4b.json

python merge_sft_lora.py --base-model /root/models/Qwen3.5-4B --sft-lora saves/qwen35-4b/sft_lora --output-dir saves/qwen35-4b/sft_merged

CUDA_VISIBLE_DEVICES=0 accelerate launch train_grpo_trl_vllm.py --config configs/grpo_qwen35_4b_vllm.json --debug-sample-size 20

python chat_catgirl.py --base-model saves/qwen35-4b/sft_merged --grpo-lora saves/qwen35-4b/grpo_lora
```
