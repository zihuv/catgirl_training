import json
import os
from collections import defaultdict
import random

# 设置随机种子，保证每次划分一致
random.seed(42)

# ---------- 路径配置 ----------
INPUT_FILE = "/mnt/workspace/.cache/modelscope/datasets/AI-ModelScope/NekoQA-30K/NekoQA-30K.json"          # 原始数据
OUTPUT_DIR = "./data"                     # 输出目录
SFT_SIZE = 6000
GRPO_SIZE = 600
TEST_SIZE = 500
GRPO_MIN_REFERENCE_CHARS = 120
GRPO_MAX_REFERENCE_CHARS = 700
SYSTEM_PROMPT = "你是猫娘，请以猫娘的语气自然对话，不要输出思考过程。"

# ---------- 1. 加载数据 ----------
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)             # 原始数据是一个 list

print(f"✅ 加载原始数据，共 {len(raw_data)} 条")

# ---------- 2. 按 type 分层 ----------
type_to_samples = defaultdict(list)
for sample in raw_data:
    t = sample['type']
    type_to_samples[t].append(sample)

print(f"📊 类型分布：{ {k: len(v) for k, v in type_to_samples.items()} }")

# ---------- 3. 分层抽样函数 ----------
def stratified_sample_by_type(total_size, exclude_ids=None, candidate_filter=None):
    """按候选池内的类型比例抽样，可排除已使用样本。"""
    exclude_ids = exclude_ids or set()
    candidates_by_type = defaultdict(list)
    for sample in raw_data:
        if id(sample) in exclude_ids:
            continue
        if candidate_filter is not None and not candidate_filter(sample):
            continue
        candidates_by_type[sample["type"]].append(sample)

    candidate_total = sum(len(samples) for samples in candidates_by_type.values())
    if candidate_total < total_size:
        raise ValueError(f"候选样本不足：需要 {total_size} 条，只有 {candidate_total} 条")

    proportions = {t: len(samples) / candidate_total for t, samples in candidates_by_type.items()}
    samples_needed = {t: int(round(total_size * proportion)) for t, proportion in proportions.items()}

    # 修正四舍五入带来的总和偏差
    diff = total_size - sum(samples_needed.values())
    sorted_types = sorted(proportions, key=proportions.get, reverse=True)
    if diff > 0:
        for i in range(diff):
            samples_needed[sorted_types[i % len(sorted_types)]] += 1
    elif diff < 0:
        for t in sorted_types:
            if diff == 0:
                break
            if samples_needed[t] > 0:
                samples_needed[t] -= 1
                diff += 1

    selected = []
    for t, need in samples_needed.items():
        selected.extend(random.sample(candidates_by_type[t], need))
    random.shuffle(selected)
    return selected

def is_grpo_candidate(sample):
    """GRPO 用中等长度参考答案；超长答案留给 SFT 学风格，不拿来做长度偏好。"""
    output_len = len(sample["output"])
    return GRPO_MIN_REFERENCE_CHARS <= output_len <= GRPO_MAX_REFERENCE_CHARS

# SFT 占大头，用来稳住猫娘人格；GRPO 单独从中等长度样本中抽取。
sft_pool = stratified_sample_by_type(SFT_SIZE)
used_ids = set(id(x) for x in sft_pool)
grpo_pool = stratified_sample_by_type(GRPO_SIZE, exclude_ids=used_ids, candidate_filter=is_grpo_candidate)
used_ids.update(id(x) for x in grpo_pool)
test_pool = stratified_sample_by_type(TEST_SIZE, exclude_ids=used_ids)

print(f"✅ 抽样完成：SFT {len(sft_pool)} | GRPO {len(grpo_pool)} | Test {len(test_pool)}")

# 剩余未使用的数据（留作备用）
used_ids.update(id(x) for x in test_pool)
remain_pool = [x for x in raw_data if id(x) not in used_ids]
print(f"📦 剩余备用数据：{len(remain_pool)} 条")

# ---------- 4. 转换为 ShareGPT 格式 ----------
def to_sharegpt_sft(sample):
    """SFT 格式：human + gpt 两个 turn"""
    return {
        "system": SYSTEM_PROMPT,
        "conversations": [
            {"from": "human", "value": sample["instruction"]},
            {"from": "gpt", "value": sample["output"]}
        ]
    }

def to_sharegpt_grpo(sample):
    """GRPO 格式：human turn + 参考答案元数据，训练时可用于奖励塑形"""
    return {
        "system": SYSTEM_PROMPT,
        "conversations": [
            {"from": "human", "value": sample["instruction"]}
        ],
        "reference_output": sample["output"],
        "type": sample["type"]
    }

def to_sharegpt_test(sample):
    """测试集保留完整对话（同 SFT 格式）"""
    return to_sharegpt_sft(sample)

# ---------- 5. 保存为 JSON 文件 ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "catgirl_sft.json"), 'w', encoding='utf-8') as f:
    json.dump([to_sharegpt_sft(s) for s in sft_pool], f, ensure_ascii=False, indent=2)

with open(os.path.join(OUTPUT_DIR, "catgirl_grpo.json"), 'w', encoding='utf-8') as f:
    json.dump([to_sharegpt_grpo(s) for s in grpo_pool], f, ensure_ascii=False, indent=2)

with open(os.path.join(OUTPUT_DIR, "test.json"), 'w', encoding='utf-8') as f:
    json.dump([to_sharegpt_test(s) for s in test_pool], f, ensure_ascii=False, indent=2)

print(f"💾 已保存至 {OUTPUT_DIR}/")
print(f"   - catgirl_sft.json   ({SFT_SIZE}条，ShareGPT 格式)")
print(f"   - catgirl_grpo.json  ({GRPO_SIZE}条，human turn + reference_output/type)")
print(f"   - test.json          ({TEST_SIZE}条，ShareGPT 格式)")
