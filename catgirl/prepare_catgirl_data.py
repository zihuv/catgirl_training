#!/usr/bin/env python3
"""Prepare NekoQA catgirl data for TRL SFT and GRPO."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = "你是猫娘，请以猫娘的语气自然对话，不要输出思考过程。"
TASK_TYPES = {"代码编程", "数学计算", "生活技巧", "心理疗愈", "职场辅导"}
TASK_SIGNAL_WORDS = (
    "步骤",
    "建议",
    "可以",
    "首先",
    "然后",
    "因为",
    "所以",
    "代码",
    "函数",
    "```",
    "计算",
    "结果",
    "注意",
    "证据",
    "联系",
    "记录",
)


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level JSON list: {path}")
    return [item for item in data if isinstance(item, dict)]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_complete_record(record: dict[str, Any]) -> bool:
    return bool(record.get("instruction")) and bool(record.get("output")) and bool(record.get("type"))


def has_task_signal(record: dict[str, Any]) -> bool:
    """Keep strong RP samples, but avoid task samples that dodge the task completely."""
    sample_type = str(record.get("type", ""))
    if sample_type not in TASK_TYPES:
        return True
    output = str(record.get("output", ""))
    if len(output) >= 180:
        return True
    return any(word in output for word in TASK_SIGNAL_WORDS)


def stratified_sample(
    records: list[dict[str, Any]],
    total_size: int,
    rng: random.Random,
    exclude_ids: set[int] | None = None,
    min_output_chars: int | None = None,
    max_output_chars: int | None = None,
) -> list[dict[str, Any]]:
    exclude_ids = exclude_ids or set()
    candidates_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        if id(record) in exclude_ids:
            continue
        output_len = len(str(record.get("output", "")))
        if min_output_chars is not None and output_len < min_output_chars:
            continue
        if max_output_chars is not None and output_len > max_output_chars:
            continue
        candidates_by_type[str(record["type"])].append(record)

    candidate_total = sum(len(items) for items in candidates_by_type.values())
    if candidate_total < total_size:
        raise ValueError(f"Not enough candidates: need {total_size}, found {candidate_total}")

    proportions = {
        type_name: len(items) / candidate_total
        for type_name, items in candidates_by_type.items()
    }
    needs = {
        type_name: min(len(candidates_by_type[type_name]), round(total_size * proportion))
        for type_name, proportion in proportions.items()
    }

    diff = total_size - sum(needs.values())
    sorted_types = sorted(proportions, key=proportions.get, reverse=True)
    while diff != 0:
        changed = False
        for type_name in sorted_types:
            if diff > 0 and needs[type_name] < len(candidates_by_type[type_name]):
                needs[type_name] += 1
                diff -= 1
                changed = True
            elif diff < 0 and needs[type_name] > 0:
                needs[type_name] -= 1
                diff += 1
                changed = True
            if diff == 0:
                break
        if not changed:
            break

    selected: list[dict[str, Any]] = []
    for type_name, need in needs.items():
        selected.extend(rng.sample(candidates_by_type[type_name], need))
    rng.shuffle(selected)
    return selected


def to_sft_record(record: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(record["instruction"])},
            {"role": "assistant", "content": str(record["output"])},
        ],
        "type": str(record["type"]),
    }


def to_grpo_record(record: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(record["instruction"])},
        ],
        "reference_output": str(record["output"]),
        "type": str(record["type"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare TRL JSONL files for catgirl training.")
    parser.add_argument("--input", default="NekoQA-30K.json", help="Path to NekoQA JSON array.")
    parser.add_argument("--output-dir", default="data", help="Directory for generated JSONL files.")
    parser.add_argument("--sft-size", type=int, default=20000)
    parser.add_argument("--grpo-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--grpo-min-reference-chars", type=int, default=120)
    parser.add_argument("--grpo-max-reference-chars", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument(
        "--keep-weak-task-samples",
        action="store_true",
        help="Keep short task samples that appear to dodge the user's task.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    records = [record for record in load_records(input_path) if is_complete_record(record)]
    if not args.keep_weak_task_samples:
        records = [record for record in records if has_task_signal(record)]

    sft_pool = stratified_sample(records, args.sft_size, rng)
    used_ids = {id(record) for record in sft_pool}
    grpo_pool = stratified_sample(
        records,
        args.grpo_size,
        rng,
        exclude_ids=used_ids,
        min_output_chars=args.grpo_min_reference_chars,
        max_output_chars=args.grpo_max_reference_chars,
    )
    used_ids.update(id(record) for record in grpo_pool)
    eval_pool = stratified_sample(records, args.eval_size, rng, exclude_ids=used_ids)

    write_jsonl(output_dir / "catgirl_sft.jsonl", [to_sft_record(record, args.system_prompt) for record in sft_pool])
    write_jsonl(output_dir / "catgirl_grpo.jsonl", [to_grpo_record(record, args.system_prompt) for record in grpo_pool])
    write_jsonl(output_dir / "catgirl_eval.jsonl", [to_sft_record(record, args.system_prompt) for record in eval_pool])

    print(f"Input records: {len(records)}")
    print(f"SFT: {len(sft_pool)} -> {output_dir / 'catgirl_sft.jsonl'}")
    print(f"GRPO: {len(grpo_pool)} -> {output_dir / 'catgirl_grpo.jsonl'}")
    print(f"Eval: {len(eval_pool)} -> {output_dir / 'catgirl_eval.jsonl'}")
    print("SFT type distribution:", dict(Counter(record["type"] for record in sft_pool).most_common()))
    print("GRPO type distribution:", dict(Counter(record["type"] for record in grpo_pool).most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
