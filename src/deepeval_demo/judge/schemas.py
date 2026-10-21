"""裁判模块的数据类型。"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import DimensionSpec  # noqa: F401  (re-export 方便使用)


@dataclass
class JudgeInput:
    """喂给裁判的一条输入。"""

    sample_id: str
    input: str
    output: str
    trace_id: str | None = None
    evidence: str = ""


@dataclass
class ScoreResult:
    """单个维度的评分结果。"""

    dimension: str
    score: float
    reason: str
    passed: bool


@dataclass
class JudgeVerdict:
    """一条样本的完整裁判结论。"""

    scores: list[ScoreResult] = field(default_factory=list)
    overall: float = 0.0
    passed: bool = False
