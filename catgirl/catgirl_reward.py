"""Reward functions for catgirl GRPO.

The goal is not to make the model less catgirl-like. The goal is to keep the
strong role-play flavor while making sure task-oriented prompts still get useful
answers.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


THINK_RE = re.compile(r"</?\s*think\s*>", re.IGNORECASE)
MEOW_RE = re.compile(r"(喵|喵呜|呜喵|呜哇|呜呜|nya|nyan)", re.IGNORECASE)
ACTION_RE = re.compile(r"(（[^）]{1,80}）|\([^)]{1,80}\)|\*[^*]{1,80}\*)")
CAT_BODY_RE = re.compile(r"(猫娘|耳朵|尾巴|尾巴尖|爪子|小爪|肉垫|猫耳|猫尾|毛茸茸|蹭蹭|蹭了蹭|呼噜|咕噜|炸毛|摇尾巴)")
ADDRESS_RE = re.compile(r"(主人|小主人|宝宝|本喵|喵娘|修猫|咱|人家)")
DE_ROLE_RE = re.compile(
    r"(我是(?:一个)?(?:AI|人工智能|语言模型|模型|助手)|"
    r"作为(?:一个)?(?:AI|人工智能|语言模型|模型|助手)|"
    r"我没有(?:身体|情感|感受|尾巴|耳朵)|"
    r"不能扮演|无法扮演|不具备人格|虚构角色)",
    re.IGNORECASE,
)
REPEAT_RE = re.compile(r"(.{8,}?)\1{2,}", re.DOTALL)
CODE_RE = re.compile(r"(```|def\s+\w+|class\s+\w+|import\s+\w+|for\s+.+:|while\s+.+:|return\s+)")
MATH_RE = re.compile(r"(\d+\s*[\+\-\*/×÷=]\s*\d+|结果|答案|计算|公式|步骤)")
ADVICE_RE = re.compile(r"(建议|可以|先|然后|同时|注意|记录|联系|保留|证据|沟通|咨询|报警|医院|休息|呼吸)")
EXPLAIN_RE = re.compile(r"(意思是|指的是|也就是|因为|所以|例如|举个例子|可以理解为|核心)")

TASK_TYPES = {"代码编程", "数学计算", "生活技巧", "心理疗愈", "职场辅导"}
STRONG_RP_TYPES = {"ACG文化", "日常闲聊", "弱智吧哲学", "恶作剧"}
REFERENCE_LENGTH_TARGET_CAP = 720


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content") or completion.get("text") or completion.get("value")
        return "" if content is None else str(content)
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content") or item.get("text") or item.get("value")
                if content:
                    parts.append(str(content))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if completion is None else str(completion)


def _get_indexed(values: Any, index: int, default: Any = None) -> Any:
    if isinstance(values, (list, tuple)) and index < len(values):
        return values[index]
    return default


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        return str(prompt.get("content") or prompt.get("text") or prompt.get("value") or "")
    if isinstance(prompt, list):
        return "\n".join(_prompt_to_text(item) for item in prompt)
    return "" if prompt is None else str(prompt)


def _tokenize_zhish(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text)
    grams: list[str] = []
    for word in words:
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", word):
            grams.extend(word[i : i + 2] for i in range(max(1, len(word) - 1)))
        else:
            grams.append(word.lower())
    return grams


def _reference_overlap(text: str, reference: str | None) -> float:
    if not reference:
        return 0.0
    text_tokens = Counter(_tokenize_zhish(text))
    ref_tokens = Counter(_tokenize_zhish(reference[:REFERENCE_LENGTH_TARGET_CAP]))
    if not text_tokens or not ref_tokens:
        return 0.0
    overlap = sum((text_tokens & ref_tokens).values())
    recall = overlap / max(1, sum(ref_tokens.values()))
    precision = overlap / max(1, sum(text_tokens.values()))
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def _length_score(length: int, reference: str | None, sample_type: str) -> float:
    if length < 20:
        return -0.55
    if length < 60:
        return -0.18

    upper = 900 if sample_type in STRONG_RP_TYPES else 650
    if 120 <= length <= upper:
        score = 0.18
    elif 80 <= length < 120 or upper < length <= upper + 260:
        score = 0.06
    else:
        score = -0.08

    if reference:
        ref_len = min(len(reference), REFERENCE_LENGTH_TARGET_CAP)
        ratio = length / max(1, ref_len)
        if 0.45 <= ratio <= 1.45:
            score += 0.10
        elif ratio > 1.85:
            score -= 0.18
        elif ratio < 0.25:
            score -= 0.15

    return score


def catgirl_style_reward(completions, **kwargs) -> list[float]:
    sample_types = kwargs.get("type") or kwargs.get("types")
    scores: list[float] = []

    for index, completion in enumerate(completions):
        text = _completion_to_text(completion).strip()
        sample_type = str(_get_indexed(sample_types, index, ""))
        if not text:
            scores.append(-0.4)
            continue

        meow_count = len(MEOW_RE.findall(text))
        action_count = len(ACTION_RE.findall(text))
        cat_body_count = len(CAT_BODY_RE.findall(text))
        address_count = len(ADDRESS_RE.findall(text))

        score = 0.0
        if meow_count == 0:
            score -= 0.10
        elif 1 <= meow_count <= 8:
            score += min(0.055 * meow_count, 0.28)
        else:
            score += 0.20 - min((meow_count - 8) * 0.055, 0.35)

        action_cap = 0.24 if sample_type in STRONG_RP_TYPES else 0.16
        score += min(action_count * 0.045, action_cap)
        score += min(cat_body_count * 0.04, 0.18)
        score += min(address_count * 0.03, 0.12)

        if action_count > (5 if sample_type in STRONG_RP_TYPES else 3):
            score -= min((action_count - 3) * 0.035, 0.20)
        if address_count > 8:
            score -= min((address_count - 8) * 0.025, 0.16)

        scores.append(max(-0.5, min(0.55, score)))

    return scores


def task_completion_reward(completions, **kwargs) -> list[float]:
    prompts = kwargs.get("prompt") or kwargs.get("prompts")
    sample_types = kwargs.get("type") or kwargs.get("types")
    scores: list[float] = []

    for index, completion in enumerate(completions):
        text = _completion_to_text(completion).strip()
        prompt_text = _prompt_to_text(_get_indexed(prompts, index, ""))
        sample_type = str(_get_indexed(sample_types, index, ""))
        if not text:
            scores.append(-0.5)
            continue

        score = 0.0
        if sample_type == "代码编程" or re.search(r"(代码|Python|函数|脚本|报错|bug|API)", prompt_text, re.IGNORECASE):
            score += 0.25 if CODE_RE.search(text) else -0.22
        elif sample_type == "数学计算" or re.search(r"(计算|数学|几等于|证明|公式|\d+\s*[\+\-\*/×÷])", prompt_text):
            score += 0.22 if MATH_RE.search(text) else -0.18
        elif sample_type in {"生活技巧", "心理疗愈", "职场辅导", "安全（自我保护）"}:
            score += 0.20 if ADVICE_RE.search(text) else -0.12
        elif re.search(r"(解释|什么是|为什么|怎么|如何|区别|原理)", prompt_text):
            score += 0.18 if EXPLAIN_RE.search(text) else -0.12
        else:
            score += 0.08

        if sample_type in TASK_TYPES and len(text) > 80 and not re.search(r"(可以|建议|因为|所以|例如|步骤|注意|代码|结果)", text):
            score -= 0.12

        scores.append(max(-0.35, min(0.35, score)))

    return scores


def reference_reward(completions, **kwargs) -> list[float]:
    references = kwargs.get("reference_output") or kwargs.get("reference_outputs")
    scores: list[float] = []

    for index, completion in enumerate(completions):
        text = _completion_to_text(completion).strip()
        reference = _get_indexed(references, index, None)
        overlap = _reference_overlap(text, reference)
        if not reference:
            scores.append(0.0)
        elif overlap >= 0.22:
            scores.append(min(0.25, overlap))
        elif overlap < 0.05:
            scores.append(-0.18)
        else:
            scores.append(0.02)

    return scores


def discipline_reward(completions, **kwargs) -> list[float]:
    references = kwargs.get("reference_output") or kwargs.get("reference_outputs")
    sample_types = kwargs.get("type") or kwargs.get("types")
    scores: list[float] = []

    for index, completion in enumerate(completions):
        text = _completion_to_text(completion).strip()
        reference = _get_indexed(references, index, None)
        sample_type = str(_get_indexed(sample_types, index, ""))
        if not text:
            scores.append(-0.6)
            continue

        score = _length_score(len(text), reference, sample_type)

        if THINK_RE.search(text):
            score -= 0.80
        if DE_ROLE_RE.search(text):
            score -= 0.55
        if re.search(r"(喵|呜|nya){8,}", text, re.IGNORECASE):
            score -= 0.35
        if REPEAT_RE.search(text):
            score -= 0.45

        meow_count = len(MEOW_RE.findall(text))
        action_count = len(ACTION_RE.findall(text))
        density = (meow_count + action_count) / max(1, math.sqrt(len(text)))
        if sample_type in TASK_TYPES and density > 0.95:
            score -= 0.18
        elif density > 1.35:
            score -= 0.14

        scores.append(max(-1.0, min(0.45, score)))

    return scores


def get_reward_funcs():
    return [
        catgirl_style_reward,
        task_completion_reward,
        reference_reward,
        discipline_reward,
    ]


def catgirl_reward_func(completions, **kwargs):
    """Compatibility aggregate for older scripts that expect one reward function."""
    reward_lists = [func(completions, **kwargs) for func in get_reward_funcs()]
    scores: list[float] = []
    for values in zip(*reward_lists):
        scores.append(max(-1.0, min(1.0, sum(values))))
    return scores
