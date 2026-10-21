"""裁判引擎：DeepSeek OpenAI 兼容接口 + 结构化 JSON 多维度评分。"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from ..config import DimensionSpec, JudgeConfig
from .schemas import JudgeInput, JudgeVerdict, ScoreResult


class JudgeError(RuntimeError):
    """裁判调用或解析失败。"""


def _final_answer_evidence(
    sample_id: str,
    input_text: str,
    output_text: str,
    observations: list[dict[str, Any]],
    cfg: JudgeConfig,
) -> str:
    return f"【问题】\n{input_text}\n\n【回答】\n{output_text}"


def _preview(value: Any, max_len: int = 300) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[:max_len] + "…"


def _summarize_observation(obs: dict[str, Any]) -> str:
    return (
        f"- [{obs.get('type', '-')}] {obs.get('name', '-')}"
        f" model={obs.get('model') or '-'}"
        f" latency={obs.get('latency')}"
        f"\n  input : {_preview(obs.get('input'))}"
        f"\n  output: {_preview(obs.get('output'))}"
    )


def _trajectory_evidence(
    sample_id: str,
    input_text: str,
    output_text: str,
    observations: list[dict[str, Any]],
    cfg: JudgeConfig,
) -> str:
    base = _final_answer_evidence(sample_id, input_text, output_text, observations, cfg)
    if not observations:
        return base
    lines = [_summarize_observation(obs) for obs in observations]
    return base + "\n\n【执行轨迹】\n" + "\n".join(lines)


EVIDENCE_BUILDERS: dict[str, Callable[..., str]] = {
    "final_answer": _final_answer_evidence,
    "trajectory": _trajectory_evidence,
}


def register_evidence_builder(name: str, fn: Callable[..., str]) -> None:
    """注册自定义 evidence builder，供配置里的 ``evidence`` 键引用。"""
    EVIDENCE_BUILDERS[name] = fn


def build_judge_input(
    sample_id: str,
    input_text: str,
    output_text: str,
    *,
    trace_id: str | None = None,
    observations: list[dict[str, Any]] | None = None,
    cfg: JudgeConfig | None = None,
) -> JudgeInput:
    """按配置的 evidence 方式组装裁判输入。"""
    cfg = cfg or JudgeConfig()
    builder = EVIDENCE_BUILDERS.get(cfg.evidence)
    if builder is None:
        raise ValueError(
            f"未知的 evidence builder: {cfg.evidence}，可选: {sorted(EVIDENCE_BUILDERS)}"
        )
    evidence = builder(
        sample_id, input_text, output_text, list(observations or []), cfg
    )
    return JudgeInput(
        sample_id=sample_id,
        input=input_text,
        output=output_text,
        trace_id=trace_id,
        evidence=evidence,
    )


def _dimension_rubric(dimensions: list[DimensionSpec]) -> str:
    lines = []
    for index, dim in enumerate(dimensions, 1):
        steps = (
            "\n".join(f"  - {s}" for s in dim.steps) if dim.steps else "（无额外步骤）"
        )
        lines.append(f"{index}. {dim.name}（权重 {dim.weight}）: {dim.criteria}\n{steps}")
    return "\n".join(lines)


def _single_call_messages(
    inp: JudgeInput, dimensions: list[DimensionSpec], cfg: JudgeConfig
) -> list[dict[str, str]]:
    system = (
        "你是一名严格的评测裁判。请根据问题和回答，从给定的多个维度打分。\n"
        "每个维度的分数必须是 0 到 1 之间的小数（0 最差，1 最好）。\n"
        "你必须只输出一个 JSON 对象，格式为：\n"
        '{"scores":[{"dimension":"<维度名>","score":0.75,"reason":"<用中文简述理由>"}]}\n'
        "scores 数组必须覆盖给出的每一个维度，不得遗漏或新增。"
    )
    user = (
        f"{inp.evidence}\n\n【评分维度】\n{_dimension_rubric(dimensions)}\n\n"
        "请用中文为每个维度给出分数与理由，输出 JSON。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _dimension_messages(
    inp: JudgeInput, dim: DimensionSpec, cfg: JudgeConfig
) -> list[dict[str, str]]:
    system = (
        "你是一名严格的评测裁判。请针对单个评分维度打分。\n"
        "分数必须是 0 到 1 之间的小数（0 最差，1 最好）。\n"
        '只输出 JSON：{"score":0.75,"reason":"<用中文简述理由>"}'
    )
    steps = "\n".join(f"  - {s}" for s in dim.steps) if dim.steps else "（无额外步骤）"
    user = (
        f"{inp.evidence}\n\n【评分维度】{dim.name}: {dim.criteria}\n{steps}\n\n"
        "请用中文给出该维度的分数与理由，输出 JSON。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def _validate_scores(
    raw: Any, dimensions: list[DimensionSpec], cfg: JudgeConfig
) -> JudgeVerdict:
    if not isinstance(raw, dict):
        raise JudgeError(f"裁判输出不是 JSON 对象: {_preview(raw, 120)}")
    items = raw.get("scores")
    if not isinstance(items, list) or not items:
        raise JudgeError("裁判 JSON 缺少 scores 数组")

    by_dim: dict[str, tuple[float, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("dimension", "")).strip().lower()
        if not name:
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError) as exc:
            raise JudgeError(f"维度 {name} 的分数无法解析: {item.get('score')!r}") from exc
        by_dim[name] = (score, str(item.get("reason", "") or ""))

    scores: list[ScoreResult] = []
    total_weight = 0.0
    weighted = 0.0
    for dim in dimensions:
        key = dim.name.strip().lower()
        if key not in by_dim:
            raise JudgeError(f"裁判缺少维度 {dim.name} 的分数")
        raw_score, reason = by_dim[key]
        score = _clamp(raw_score)
        threshold = dim.threshold if dim.threshold is not None else cfg.threshold
        scores.append(
            ScoreResult(
                dimension=dim.name,
                score=round(score, 4),
                reason=reason,
                passed=score >= threshold,
            )
        )
        weighted += dim.weight * score
        total_weight += dim.weight

    overall = weighted / total_weight if total_weight else 0.0
    overall = round(overall, 4)
    return JudgeVerdict(scores=scores, overall=overall, passed=overall >= cfg.threshold)


_throttle_lock = threading.Lock()
_last_call_at = [0.0]


def _throttle(interval: float) -> None:
    if interval <= 0:
        return
    with _throttle_lock:
        now = time.monotonic()
        wait = _last_call_at[0] + interval - now
        if wait > 0:
            time.sleep(wait)
        _last_call_at[0] = time.monotonic()


def _make_client(cfg: JudgeConfig):
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise JudgeError(f"缺少裁判 API Key 环境变量 {cfg.api_key_env}")
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=cfg.base_url)


def _parse_content(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"裁判输出不是合法 JSON: {_preview(text, 120)}") from exc
    if not isinstance(data, dict):
        raise JudgeError("裁判输出顶层不是 JSON 对象")
    return data


def _judge_single_call(client: Any, inp: JudgeInput, dims: list[DimensionSpec], cfg: JudgeConfig) -> JudgeVerdict:
    _throttle(cfg.throttle_seconds)
    response = client.chat.completions.create(
        model=cfg.model,
        messages=_single_call_messages(inp, dims, cfg),
        response_format={"type": "json_object"},
        temperature=cfg.temperature,
    )
    content = response.choices[0].message.content
    return _validate_scores(_parse_content(content), dims, cfg)


def _judge_per_dimension(client: Any, inp: JudgeInput, dims: list[DimensionSpec], cfg: JudgeConfig) -> JudgeVerdict:
    scores: list[ScoreResult] = []
    weighted = 0.0
    total_weight = 0.0
    for dim in dims:
        _throttle(cfg.throttle_seconds)
        response = client.chat.completions.create(
            model=cfg.model,
            messages=_dimension_messages(inp, dim, cfg),
            response_format={"type": "json_object"},
            temperature=cfg.temperature,
        )
        data = _parse_content(response.choices[0].message.content)
        try:
            raw_score = float(data.get("score"))
        except (TypeError, ValueError) as exc:
            raise JudgeError(f"维度 {dim.name} 的分数无法解析") from exc
        score = _clamp(raw_score)
        reason = str(data.get("reason", "") or "")
        threshold = dim.threshold if dim.threshold is not None else cfg.threshold
        scores.append(
            ScoreResult(
                dimension=dim.name,
                score=round(score, 4),
                reason=reason,
                passed=score >= threshold,
            )
        )
        weighted += dim.weight * score
        total_weight += dim.weight
    overall = round(weighted / total_weight, 4) if total_weight else 0.0
    return JudgeVerdict(scores=scores, overall=overall, passed=overall >= cfg.threshold)


def _judge_one_with_retry(
    client: Any, inp: JudgeInput, dims: list[DimensionSpec], cfg: JudgeConfig
) -> JudgeVerdict:
    last_error: Exception | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            if cfg.scoring == "per_dimension":
                return _judge_per_dimension(client, inp, dims, cfg)
            return _judge_single_call(client, inp, dims, cfg)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < cfg.max_retries:
                time.sleep(2.0 * attempt)

    reason = f"评测失败: {last_error}"
    scores = [
        ScoreResult(dimension=dim.name, score=0.0, reason=reason, passed=False)
        for dim in dims
    ]
    return JudgeVerdict(scores=scores, overall=0.0, passed=False)


def judge_batch(
    inputs: list[JudgeInput],
    cfg: JudgeConfig | None = None,
) -> list[JudgeVerdict]:
    """并发对多条 JudgeInput 打分，返回与输入顺序一致的 JudgeVerdict 列表。"""
    cfg = cfg or JudgeConfig()
    if not inputs:
        return []
    dims = cfg.dimensions
    if not dims:
        raise JudgeError("没有配置任何评分维度")

    client = _make_client(cfg)
    results: list[JudgeVerdict | None] = [None] * len(inputs)
    workers = max(1, cfg.max_concurrent)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_index = {
            pool.submit(_judge_one_with_retry, client, inp, dims, cfg): index
            for index, inp in enumerate(inputs)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    return [r for r in results if r is not None]
