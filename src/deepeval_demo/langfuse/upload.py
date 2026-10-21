"""把裁判分数上传回 Langfuse。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..judge import JudgeVerdict


@dataclass
class ScoreRecord:
    name: str
    value: float
    uploaded: bool
    error: str | None = None


def upload_scores(
    client: Any,
    trace_id: str,
    verdict: JudgeVerdict,
    run_meta: dict[str, Any] | None = None,
    *,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> list[ScoreRecord]:
    """每维度一条 ``judge_<dim>`` 分数，外加一条 ``judge_overall`` 加权总分。"""
    run_meta = dict(run_meta or {})
    entries: list[tuple[str, float, str]] = [
        (f"judge_{s.dimension}", s.score, s.reason) for s in verdict.scores
    ]
    overall_reason = "加权总分: " + "; ".join(
        f"{s.dimension}={s.score:.3f}" for s in verdict.scores
    )
    entries.append(("judge_overall", verdict.overall, overall_reason))

    records: list[ScoreRecord] = []
    for name, value, comment in entries:
        uploaded = False
        error: str | None = None
        for attempt in range(1, max_retries + 1):
            try:
                client.create_score(
                    name=name,
                    value=float(value),
                    trace_id=trace_id,
                    data_type="NUMERIC",
                    comment=comment,
                    metadata=dict(run_meta),
                )
                uploaded = True
                error = None
                break
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                if attempt < max_retries:
                    time.sleep(backoff_seconds * attempt)
        records.append(
            ScoreRecord(name=name, value=float(value), uploaded=uploaded, error=error)
        )
    return records


def upload_scorer_scores(
    client: Any,
    trace_id: str,
    deterministic_stats: dict[str, dict[str, Any]],
    run_meta: dict[str, Any] | None = None,
    *,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> list[ScoreRecord]:
    """把确定性 scorer 的归并结果上传为 ``scorer_<name>`` 数值分数。"""
    run_meta = dict(run_meta or {})
    records: list[ScoreRecord] = []
    for name, stats in deterministic_stats.items():
        value = float(stats.get("value", 0.0))
        comment = (
            f"reducer={stats.get('reducer', 'pass_at_k')} k={stats.get('k', 1)} "
            f"mean={stats.get('mean', 0):.3f} pass@1={stats.get('pass_at_1')} "
            f"pass@k={stats.get('pass_at_k')}"
        )
        uploaded = False
        error: str | None = None
        for attempt in range(1, max_retries + 1):
            try:
                client.create_score(
                    name=f"scorer_{name}",
                    value=value,
                    trace_id=trace_id,
                    data_type="NUMERIC",
                    comment=comment,
                    metadata=dict(run_meta),
                )
                uploaded = True
                error = None
                break
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                if attempt < max_retries:
                    time.sleep(backoff_seconds * attempt)
        records.append(
            ScoreRecord(name=f"scorer_{name}", value=value, uploaded=uploaded, error=error)
        )
    return records
