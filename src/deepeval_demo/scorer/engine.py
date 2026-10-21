"""确定性 scorer 引擎：内置 exact / contains / regex / numeric，可注册自定义。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from ..config import ScorerSpec

ScorerFn = Callable[[str, str, dict[str, Any]], float]


@dataclass
class ScorerResult:
    """单个确定性 scorer 的评分结果。"""

    name: str
    score: float
    passed: bool
    detail: str = ""


SCORERS: dict[str, ScorerFn] = {}


def register_scorer(name: str, fn: ScorerFn) -> None:
    """注册自定义确定性 scorer。"""
    SCORERS[name] = fn


def _norm_text(text: str, params: dict[str, Any]) -> str:
    value = text or ""
    if not params.get("case_sensitive", False):
        value = value.lower()
    if params.get("normalize_whitespace", True):
        value = " ".join(value.split())
    if params.get("strip_punctuation", False):
        value = re.sub(r"[^\w\s]", "", value)
    return value


def _norm_number(text: str) -> str:
    return re.sub(r"[,\s]", "", (text or ""))


def _exact(output: str, target: str, params: dict[str, Any]) -> float:
    return 1.0 if _norm_text(output, params) == _norm_text(target, params) else 0.0


def _contains(output: str, target: str, params: dict[str, Any]) -> float:
    out = _norm_text(output, params)
    tgt = _norm_text(target, params)
    if params.get("reverse", False):
        out, tgt = tgt, out
    if not tgt:
        return 0.0
    return 1.0 if tgt in out else 0.0


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text or "")
    if not match:
        return None
    if match.lastindex:
        return match.group(match.lastindex)
    return match.group(0)


def _regex(output: str, target: str, params: dict[str, Any]) -> float:
    pattern = params.get("pattern")
    if not pattern:
        raise ValueError("regex scorer 需要 params.pattern")
    out_match = _first_match(pattern, output)
    if out_match is None:
        return 0.0
    if params.get("extract_from_target", False):
        tgt_match = _first_match(pattern, target)
        if tgt_match is None:
            return 0.0
        return 1.0 if _norm_number(out_match) == _norm_number(tgt_match) else 0.0
    expected = params.get("expected")
    if expected is not None:
        return 1.0 if _norm_number(out_match) == _norm_number(str(expected)) else 0.0
    return 1.0 if _norm_number(out_match) in _norm_number(target or "") else 0.0


def _first_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text or "")
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _numeric(output: str, target: str, params: dict[str, Any]) -> float:
    tolerance = float(params.get("tolerance", 1e-9))
    out_num = _first_number(output)
    tgt_num = _first_number(target)
    if out_num is None or tgt_num is None:
        return 0.0
    return 1.0 if abs(out_num - tgt_num) <= tolerance else 0.0


register_scorer("exact", _exact)
register_scorer("contains", _contains)
register_scorer("regex", _regex)
register_scorer("numeric", _numeric)


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def score_one(spec: ScorerSpec, output: str, target: str | None) -> ScorerResult:
    """用单个 scorer 对输出打分。"""
    fn = SCORERS.get(spec.type)
    if fn is None:
        raise ValueError(
            f"未知的 scorer 类型: {spec.type}，可选: {sorted(SCORERS)}"
        )
    score = _clamp(float(fn(output or "", target or "", dict(spec.params))))
    score = round(score, 4)
    return ScorerResult(
        name=spec.name,
        score=score,
        passed=score >= spec.threshold,
    )


def run_scorers(
    specs: list[ScorerSpec], output: str, target: str | None
) -> list[ScorerResult]:
    """对一条输出跑全部确定性 scorer，返回与 specs 顺序一致的结果列表。"""
    return [score_one(spec, output, target) for spec in specs]


def reduce_epoch_scores(
    epoch_scores: list[list[ScorerResult]],
    specs: list[ScorerSpec],
    reducer: str,
    *,
    pass_k: int | None = None,
) -> dict[str, dict[str, Any]]:
    """把每个 epoch 的确定性分数归并成每个 scorer 的汇总指标。"""
    if not specs or not epoch_scores:
        return {}
    epochs = len(epoch_scores)
    k = pass_k or epochs
    k = max(1, min(k, epochs))

    out: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(specs):
        values = [epoch[index].score for epoch in epoch_scores]
        best = max(values)
        mean = sum(values) / len(values)
        pass_at_1 = values[0] >= spec.threshold
        pass_at_k = any(v >= spec.threshold for v in values[:k])
        if reducer == "average":
            value = mean
            passed = mean >= spec.threshold
        else:
            value = best
            passed = any(v >= spec.threshold for v in values[:k])
        out[spec.name] = {
            "reducer": reducer,
            "value": round(value, 4),
            "mean": round(mean, 4),
            "best": round(best, 4),
            "passed": passed,
            "pass_at_1": pass_at_1,
            "pass_at_k": pass_at_k,
            "k": k,
        }
    return out
