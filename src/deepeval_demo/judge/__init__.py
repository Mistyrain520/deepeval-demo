"""裁判模块：用 DeepSeek 作为 LLM 裁判，按多维度结构化打分。"""

from .schemas import JudgeInput, JudgeVerdict, ScoreResult
from .dimensions import default_dimensions
from .engine import (
    EVIDENCE_BUILDERS,
    build_judge_input,
    judge_batch,
    register_evidence_builder,
)

__all__ = [
    "JudgeInput",
    "JudgeVerdict",
    "ScoreResult",
    "default_dimensions",
    "EVIDENCE_BUILDERS",
    "build_judge_input",
    "judge_batch",
    "register_evidence_builder",
]
