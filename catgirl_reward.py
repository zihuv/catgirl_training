import re
from typing import Any


THINK_RE = re.compile(r"</?\s*think\s*>", re.IGNORECASE)
MEOW_RE = re.compile(r"(喵|喵呜|呜喵|呜哇|呜呜|nya|nyan)", re.IGNORECASE)
ACTION_RE = re.compile(r"(（[^）]{1,60}）|\([^)]{1,60}\)|\*[^*]{1,60}\*)")
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
REFERENCE_LENGTH_TARGET_CAP = 520


def _completion_to_text(completion: Any) -> str:
    """Handle both plain-string and chat-message completions from TRL."""
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


def _length_reward(length: int, reference: str | None) -> float:
    if length < 20:
        return -0.80
    if length < 60:
        return -0.40
    if length < 100:
        return 0.05
    if 140 <= length <= 700:
        base = 0.28
    elif 100 <= length < 140 or 700 < length <= 950:
        base = 0.15
    elif length > 1200:
        base = -0.25
    else:
        base = 0.0

    if not reference:
        return base

    ref_len = min(len(reference), REFERENCE_LENGTH_TARGET_CAP)
    if ref_len <= 0:
        return base

    ratio = length / ref_len
    if 0.55 <= ratio <= 1.35:
        base += 0.12
    elif ratio < 0.30:
        base -= 0.18

    return base


def catgirl_reward_func(completions, **kwargs):
    reference_outputs = kwargs.get("reference_output") or kwargs.get("reference_outputs")
    scores = []

    for index, completion in enumerate(completions):
        text = _completion_to_text(completion).strip()
        reference = _get_indexed(reference_outputs, index, None)

        if not text:
            scores.append(-1.0)
            continue

        score = 0.0
        length = len(text)

        # 1. 回复长度：猫娘风格需要一点铺陈，但不要灌水。
        score += _length_reward(length, reference)

        # 2. 猫娘语气：以加分为主，避免把口癖变成硬性格式。
        meow_count = len(MEOW_RE.findall(text))
        if meow_count == 0:
            score -= 0.15
        elif 1 <= meow_count <= 8:
            score += min(0.07 * meow_count, 0.28)
        else:
            score += 0.14
            score -= min((meow_count - 8) * 0.04, 0.25)

        # 3. 角色动作和猫娘身体感：自然出现就奖励，不强迫每句都带。
        action_count = len(ACTION_RE.findall(text))
        score += min(action_count * 0.06, 0.18)

        cat_body_count = len(CAT_BODY_RE.findall(text))
        score += min(cat_body_count * 0.05, 0.18)

        address_count = len(ADDRESS_RE.findall(text))
        score += min(address_count * 0.04, 0.12)

        # 4. Qwen thinking 和出戏表达会直接破坏人格。
        if THINK_RE.search(text):
            score -= 0.90

        if DE_ROLE_RE.search(text):
            score -= 0.70

        # 5. 反复横跳和硬刷口癖。
        if re.search(r"(喵|呜|nya){8,}", text, re.IGNORECASE):
            score -= 0.45

        if REPEAT_RE.search(text):
            score -= 0.55

        scores.append(max(-1.0, min(1.0, score)))

    return scores
