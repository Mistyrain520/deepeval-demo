"""评测报告：汇总结果并写入 JSON。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .log import get_logger, log_event

logger = get_logger(__name__)


@dataclass
class RunReport:
    """一次评测的汇总报告。"""

    run_id: str
    name: str
    generated_at: str
    total_samples: int
    total_traces: int
    judge_model: str
    threshold: float
    uploaded: bool
    overall_average: float
    pass_rate: float
    dimension_averages: dict[str, float]
    results: list[dict[str, Any]] = field(default_factory=list)
    epochs: int = 1
    epoch_reducer: str = "pass_at_k"
    deterministic_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    agent_model: str | None = None
    codex_version: str | None = None
    dataset_hash: str | None = None
    seed: int | None = None
    held_rate: float = 0.0
    mean_rounds: float = 0.0


def write_report(report: RunReport, run_dir: str | Path) -> Path:
    """把报告写到 run_dir/report.json，返回路径。"""
    from dataclasses import asdict

    started = time.perf_counter()
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log_event(
        logger,
        "INFO",
        "report.write.done",
        path=str(path),
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return path


def assemble_report(
    *,
    run_id: str,
    name: str,
    judge_model: str,
    threshold: float,
    uploaded: bool,
    results: list[dict[str, Any]],
    verdicts: list[Any],
    epochs: int = 1,
    epoch_reducer: str = "pass_at_k",
    agent_model: str | None = None,
    codex_version: str | None = None,
    dataset_hash: str | None = None,
    seed: int | None = None,
    total_traces: int = 0,
) -> RunReport:
    """由逐样本结果 + verdict 列表汇总成 RunReport（单一真相源）。"""
    generated_at = datetime.now(timezone.utc).isoformat()
    overall_values = [v.overall for v in verdicts]
    overall_average = (
        sum(overall_values) / len(overall_values) if overall_values else 0.0
    )
    passed_count = sum(1 for v in verdicts if v.passed)
    pass_rate = passed_count / len(verdicts) if verdicts else 0.0

    dimension_averages: dict[str, float] = {}
    if verdicts:
        for dim_name in (s.dimension for s in verdicts[0].scores):
            values = [
                next((s.score for s in v.scores if s.dimension == dim_name), 0.0)
                for v in verdicts
            ]
            dimension_averages[dim_name] = (
                sum(values) / len(values) if values else 0.0
            )

    det_agg: dict[str, dict[str, float | int]] = {}
    for result in results:
        for scorer_name, stats in (result.get("deterministic") or {}).items():
            agg = det_agg.setdefault(
                scorer_name,
                {
                    "mean": 0.0,
                    "pass_rate": 0.0,
                    "pass_at_1_rate": 0.0,
                    "pass_at_k_rate": 0.0,
                    "count": 0,
                },
            )
            agg["count"] = int(agg["count"]) + 1
            agg["mean"] = float(agg["mean"]) + float(stats.get("mean", 0.0))
            if stats.get("passed"):
                agg["pass_rate"] = float(agg["pass_rate"]) + 1.0
            if stats.get("pass_at_1"):
                agg["pass_at_1_rate"] = float(agg["pass_at_1_rate"]) + 1.0
            if stats.get("pass_at_k"):
                agg["pass_at_k_rate"] = float(agg["pass_at_k_rate"]) + 1.0
    for agg in det_agg.values():
        count = int(agg.pop("count")) or 1
        agg["mean"] = round(float(agg["mean"]) / count, 4)
        agg["pass_rate"] = round(float(agg["pass_rate"]) / count, 4)
        agg["pass_at_1_rate"] = round(float(agg["pass_at_1_rate"]) / count, 4)
        agg["pass_at_k_rate"] = round(float(agg["pass_at_k_rate"]) / count, 4)

    rounds_values = [
        int(r["rounds_used"])
        for r in results
        if r.get("rounds_used") is not None
    ]
    held_values = [
        bool(r["held"])
        for r in results
        if r.get("held") is not None
    ]
    mean_rounds = (
        sum(rounds_values) / len(rounds_values) if rounds_values else 0.0
    )
    held_rate = (
        sum(held_values) / len(held_values) if held_values else 0.0
    )

    return RunReport(
        run_id=run_id,
        name=name,
        generated_at=generated_at,
        total_samples=len(results),
        total_traces=total_traces,
        judge_model=judge_model,
        threshold=threshold,
        uploaded=uploaded,
        overall_average=round(overall_average, 4),
        pass_rate=round(pass_rate, 4),
        dimension_averages={
            k: round(v, 4) for k, v in dimension_averages.items()
        },
        results=results,
        epochs=epochs,
        epoch_reducer=epoch_reducer,
        deterministic_stats={
            k: {kk: vv for kk, vv in v.items()} for k, v in det_agg.items()
        },
        agent_model=agent_model,
        codex_version=codex_version,
        dataset_hash=dataset_hash,
        seed=seed,
        held_rate=round(held_rate, 4),
        mean_rounds=round(mean_rounds, 4),
    )


def print_summary(report: RunReport) -> None:
    log_event(
        logger,
        "INFO",
        "summary",
        run_id=report.run_id,
        total_samples=report.total_samples,
        total_traces=report.total_traces,
        pass_rate=report.pass_rate,
        overall_average=report.overall_average,
        dimension_averages=report.dimension_averages,
        deterministic_stats=report.deterministic_stats,
        held_rate=report.held_rate,
        mean_rounds=report.mean_rounds,
    )
