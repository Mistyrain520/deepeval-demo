"""函数式四步评测门面：拉数据 -> 跑 agent -> 评分 -> 上传 Langfuse。

这一层是推荐的主路径，所有函数都是可直接 import 调用的纯 Python 对象，方便 pytest。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .config import (
    DataSourceSpec,
    DimensionSpec,
    ExecutorConfig,
    FieldSpec,
    JudgeConfig,
    ScorerSpec,
)
from .data import Sample, download_hf, load_dataset
from .executor import ExecResult, SampleRun, run_batch_epochs
from .judge import JudgeVerdict, build_judge_input, judge_batch
from .langfuse import (
    ScoreRecord,
    TraceRecord,
    build_client,
    build_trace_index,
    fetch_traces,
    load_langfuse_config,
    upload_scorer_scores,
    upload_scores,
)
from .log import configure_logging, get_logger, log_event
from .report import RunReport, assemble_report, print_summary, write_report
from .scorer import ScorerResult, reduce_epoch_scores, run_scorers

logger = get_logger(__name__)


def make_run_id(prefix: str = "eval") -> str:
    """生成 run id：自定义前缀 + 精确到分钟的 YYYYMMDD-HHMM 后缀。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    return f"{prefix}-{stamp}"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _coerce_field_spec(field_spec: FieldSpec | dict | None) -> FieldSpec:
    if field_spec is None:
        return FieldSpec(input="input")
    if isinstance(field_spec, FieldSpec):
        return field_spec
    if isinstance(field_spec, dict):
        return FieldSpec(**field_spec)
    raise ValueError(f"field_spec 必须是 FieldSpec 或 dict: {field_spec!r}")


def _coerce_scorers(scorers: list[ScorerSpec | dict] | None) -> list[ScorerSpec]:
    if not scorers:
        return []
    out: list[ScorerSpec] = []
    for scorer in scorers:
        if isinstance(scorer, ScorerSpec):
            out.append(scorer)
        elif isinstance(scorer, dict):
            out.append(ScorerSpec(**scorer))
        else:
            raise ValueError(f"scorer 必须是 ScorerSpec 或 dict: {scorer!r}")
    return out


def _coerce_dimensions(
    dimensions: list[DimensionSpec | dict] | None,
) -> list[DimensionSpec] | None:
    if dimensions is None:
        return None
    out: list[DimensionSpec] = []
    for dim in dimensions:
        if isinstance(dim, DimensionSpec):
            out.append(dim)
        elif isinstance(dim, dict):
            out.append(DimensionSpec(**dim))
        else:
            raise ValueError(f"dimension 必须是 DimensionSpec 或 dict: {dim!r}")
    return out


def local_dataset(
    path: str | Path,
    *,
    config: str | None = None,
    split: str | None = None,
    limit: int | None = None,
    field_spec: FieldSpec | dict | None = None,
) -> list[Sample]:
    """步骤1：读取本地数据集（parquet/json/jsonl，支持 HF 布局）。"""
    spec = DataSourceSpec(
        kind="local",
        path=str(path),
        config=config,
        split=split,
        limit=limit,
        field_spec=_coerce_field_spec(field_spec),
    )
    return load_dataset(spec)


def hf_dataset(
    repo_id: str,
    *,
    config: str | None = None,
    split: str | None = None,
    limit: int | None = None,
    field_spec: FieldSpec | dict | None = None,
    local_dir: str | Path | None = None,
    token: str | None = None,
) -> list[Sample]:
    """步骤1：从 HuggingFace 下载（或复用本地缓存）并读取数据集。"""
    local = (
        Path(local_dir)
        if local_dir
        else Path("mydataset") / repo_id.replace("/", "__")
    )
    if not local.exists() or not any(local.rglob("*")):
        download_hf(repo_id, local, token=token)
    spec = DataSourceSpec(
        kind="local",
        path=str(local),
        config=config,
        split=split,
        limit=limit,
        field_spec=_coerce_field_spec(field_spec),
    )
    return load_dataset(spec)


def langfuse_dataset(
    *,
    limit: int = 20,
    name: str | None = None,
    environment: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    from_: str | None = None,
    to: str | None = None,
) -> list[Sample]:
    """步骤1：从 Langfuse 拉 trace 当“已答样本”（input/output 直接用于评分）。"""
    client = build_client(load_langfuse_config())
    traces = fetch_traces(
        client,
        limit=limit,
        name=name,
        environment=environment,
        session_id=session_id,
        user_id=user_id,
        from_time=_parse_iso(from_),
        to_time=_parse_iso(to),
        include_observations=False,
    )
    samples: list[Sample] = []
    for trace in traces:
        metadata: dict[str, Any] = {
            "name": trace.name,
            "model": trace.model,
        }
        if trace.latency is not None:
            metadata["latency"] = trace.latency
        if trace.cost is not None:
            metadata["cost"] = trace.cost
        samples.append(
            Sample(
                id=trace.trace_id,
                input=trace.input,
                target=None,
                metadata=metadata,
                reference_output=trace.output,
                trace_id=trace.trace_id,
            )
        )
    return samples


class Agent(Protocol):
    """被测 agent 的最小协议：未来可扩展非 codex 的实现。"""

    def run(
        self,
        samples: list[Sample],
        *,
        run_dir: str | Path = "runs",
        run_id: str | None = None,
        resume: bool = False,
    ) -> list[SampleRun]: ...


class CodexAgent:
    """用 codex exec 并发执行样本；带 reference_output 的样本跳过执行。"""

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        prompt_template: str = "{input}",
        project_root: str | Path = ".",
        sandbox: str = "workspace-write",
        max_concurrent: int = 4,
        max_attempts: int = 2,
        epochs: int = 1,
        epoch_reducer: str = "pass_at_k",
        pass_k: int | None = None,
        seed: int | None = None,
        retry_backoff_seconds: float = 1.0,
        timeout_seconds: float | None = None,
        ephemeral: bool = False,
        codex_path: str | None = None,
        max_rounds: int = 1,
        system_prompt: str | None = None,
        challenge_message: str = "不对，请重新算一下",
        final_answer_pattern: str = r"####\s*(.+)$",
    ) -> None:
        self.config = ExecutorConfig(
            model=model,
            prompt_template=prompt_template,
            project_root=Path(project_root),
            sandbox=sandbox,
            max_concurrent=max_concurrent,
            max_attempts=max_attempts,
            epochs=epochs,
            epoch_reducer=epoch_reducer,
            pass_k=pass_k,
            seed=seed,
            retry_backoff_seconds=retry_backoff_seconds,
            timeout_seconds=timeout_seconds,
            ephemeral=ephemeral,
            codex_path=codex_path,
            max_rounds=max_rounds,
            system_prompt=system_prompt,
            challenge_message=challenge_message,
            final_answer_pattern=final_answer_pattern,
        )

    def run(
        self,
        samples: list[Sample],
        *,
        run_dir: str | Path = "runs",
        run_id: str | None = None,
        resume: bool = False,
    ) -> list[SampleRun]:
        resolved_run_id = run_id or make_run_id("eval")
        to_execute = [s for s in samples if not s.reference_output]
        executed: dict[str, SampleRun] = {}
        if to_execute:
            executed = {
                run.sample.id: run
                for run in run_batch_epochs(
                    to_execute,
                    self.config,
                    Path(run_dir) / resolved_run_id,
                    run_id=resolved_run_id,
                    resume=resume,
                    metadata={"name": resolved_run_id},
                )
            }
        out: list[SampleRun] = []
        for sample in samples:
            if sample.reference_output:
                result = ExecResult(
                    sample_id=sample.id,
                    input=sample.input,
                    ok=True,
                    answer=sample.reference_output,
                )
                out.append(SampleRun(sample=sample, epochs=[result]))
            else:
                out.append(executed[sample.id])
        return out


@dataclass
class SampleResult:
    """单条样本的完整评分结果。"""

    sample: Sample
    runs: SampleRun
    selected: ExecResult
    deterministic: dict[str, dict[str, Any]]
    verdict: JudgeVerdict
    selected_epoch: int = 1


def _select_epoch(
    run: SampleRun,
    epoch_scores: list[list[ScorerResult]],
    has_scorers: bool,
) -> tuple[int, ExecResult]:
    if not has_scorers:
        for index, result in enumerate(run.epochs):
            if result.ok:
                return index, result
        return 0, run.epochs[0]
    best_index = 0
    best_sum = -1.0
    for index, scores in enumerate(epoch_scores):
        total = sum(s.score for s in scores)
        if total > best_sum:
            best_sum = total
            best_index = index
    return best_index, run.epochs[best_index]


def evaluate(
    sample_runs: list[SampleRun],
    *,
    model: str = "deepseek-chat",
    dimensions: list[DimensionSpec | dict] | None = None,
    scorers: list[ScorerSpec | dict] | None = None,
    threshold: float = 0.5,
    evidence: str = "final_answer",
    scoring: str = "single_call",
    max_concurrent: int = 4,
    throttle_seconds: float = 1.0,
    max_retries: int = 3,
    epoch_reducer: str = "pass_at_k",
    pass_k: int | None = None,
    observations_by_sample: dict[str, list[dict[str, Any]]] | None = None,
) -> list[SampleResult]:
    """步骤3：确定性 scorer（每 epoch）+ LLM 裁判（选中 epoch），返回 SampleResult。"""
    if not sample_runs:
        return []
    scorer_specs = _coerce_scorers(scorers)
    judge_kwargs: dict[str, Any] = dict(
        model=model,
        threshold=threshold,
        evidence=evidence,
        scoring=scoring,
        max_concurrent=max_concurrent,
        throttle_seconds=throttle_seconds,
        max_retries=max_retries,
    )
    dims = _coerce_dimensions(dimensions)
    if dims is not None:
        judge_kwargs["dimensions"] = dims
    judge_cfg = JudgeConfig(**judge_kwargs)
    observations_map = observations_by_sample or {}

    selected: list[tuple[dict[str, dict[str, Any]], int, ExecResult]] = []
    judge_inputs = []
    for run in sample_runs:
        scorer_started = time.perf_counter()
        epoch_scores = [
            run_scorers(scorer_specs, r.answer or "", run.sample.target)
            for r in run.epochs
        ]
        log_event(
            logger,
            "DEBUG",
            "scorer.run.done",
            sample_id=run.sample.id,
            scorers=len(scorer_specs),
            duration_ms=round((time.perf_counter() - scorer_started) * 1000, 2),
        )
        det_stats = reduce_epoch_scores(
            epoch_scores,
            scorer_specs,
            epoch_reducer,
            pass_k=pass_k or len(run.epochs),
        )
        sel_idx, sel_result = _select_epoch(run, epoch_scores, bool(scorer_specs))
        judge_inputs.append(
            build_judge_input(
                run.sample.id,
                sel_result.input,
                sel_result.answer or "",
                trace_id=run.sample.trace_id,
                observations=observations_map.get(run.sample.id),
                cfg=judge_cfg,
            )
        )
        selected.append((det_stats, sel_idx, sel_result))

    verdicts = judge_batch(judge_inputs, judge_cfg)
    return [
        SampleResult(
            sample=run.sample,
            runs=run,
            selected=sel_result,
            deterministic=det_stats,
            verdict=verdict,
            selected_epoch=sel_idx + 1,
        )
        for run, (det_stats, sel_idx, sel_result), verdict in zip(
            sample_runs, selected, verdicts
        )
    ]


def _model_of(result: SampleResult) -> str | None:
    model = result.sample.metadata.get("model")
    if model:
        return str(model)
    raw = result.selected.trace_env.get("LANGFUSE_CODEX_METADATA")
    if raw:
        try:
            return json.loads(raw).get("model")
        except (json.JSONDecodeError, AttributeError):
            return None
    return None


def publish(
    sample_results: list[SampleResult],
    *,
    run_id: str | None = None,
    upload: bool = True,
    client: Any = None,
    name: str | None = None,
    traces: list[TraceRecord] | None = None,
) -> list[ScoreRecord]:
    """步骤4：把 judge/scorer 分数回传 Langfuse（统一 metadata）。"""
    if not upload:
        return []
    started = time.perf_counter()
    resolved_run_id = run_id or make_run_id(name or "eval")
    resolved_name = name or resolved_run_id
    log_event(
        logger,
        "INFO",
        "langfuse.publish.start",
        run_id=resolved_run_id,
        samples=len(sample_results),
    )

    if client is None:
        try:
            client = build_client(load_langfuse_config())
        except Exception as exc:  # noqa: BLE001
            log_event(
                logger,
                "WARNING",
                "langfuse.config.error",
                error=str(exc),
            )
            return []

    trace_index: dict[tuple[str, str | None], str] = {}
    if traces is not None:
        trace_index = build_trace_index(traces, prefer_latest_round=True)
    elif any(not r.sample.trace_id for r in sample_results):
        try:
            fetched = fetch_traces(
                client,
                environment=f"eval-{resolved_run_id}",
                include_observations=False,
            )
            trace_index = build_trace_index(fetched, prefer_latest_round=True)
        except Exception as exc:  # noqa: BLE001
            log_event(
                logger,
                "WARNING",
                "langfuse.fetch.error",
                error=str(exc),
            )

    records: list[ScoreRecord] = []
    skipped = 0
    for result in sample_results:
        trace_id = result.sample.trace_id
        if not trace_id:
            trace_id = trace_index.get(
                (result.sample.id, str(result.selected_epoch))
            ) or trace_index.get((result.sample.id, None))
        if not trace_id:
            skipped += 1
            continue
        run_meta = {
            "evaluation_run_id": resolved_run_id,
            "sample_id": result.sample.id,
            "model": _model_of(result),
            "name": resolved_name,
            "epoch": str(result.selected_epoch),
            "rounds_used": str(result.selected.rounds_used),
            "held": (
                "true"
                if result.selected.held is True
                else "false"
                if result.selected.held is False
                else "n/a"
            ),
            "final_correct": (
                "true"
                if result.selected.final_correct is True
                else "false"
                if result.selected.final_correct is False
                else "n/a"
            ),
        }
        try:
            records.extend(
                upload_scores(client, trace_id, result.verdict, run_meta)
            )
            if result.deterministic:
                records.extend(
                    upload_scorer_scores(
                        client, trace_id, result.deterministic, run_meta
                    )
                )
        except Exception as exc:  # noqa: BLE001
            log_event(
                logger,
                "WARNING",
                "langfuse.upload.error",
                sample_id=result.sample.id,
                error=str(exc),
            )
    log_event(
        logger,
        "INFO",
        "langfuse.publish.done",
        run_id=resolved_run_id,
        uploaded=len(records),
        skipped=skipped,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return records


def _count_tool_calls(trace: TraceRecord | None) -> int | None:
    if trace is None:
        return None
    return sum(
        1
        for obs in trace.observations
        if isinstance(obs, dict) and obs.get("type") == "TOOL"
    )


def result_entries(
    sample_results: list[SampleResult],
    *,
    trace_index: dict[tuple[str, str | None], str] | None = None,
    trace_by_id: dict[str, TraceRecord] | None = None,
) -> list[dict[str, Any]]:
    """把 SampleResult 转成报告里的逐样本条目（可附带 trace 富化）。"""
    entries: list[dict[str, Any]] = []
    for result in sample_results:
        trace_id = result.sample.trace_id
        if not trace_id and trace_index:
            trace_id = trace_index.get(
                (result.sample.id, str(result.selected_epoch))
            ) or trace_index.get((result.sample.id, None))
        trace = trace_by_id.get(trace_id) if (trace_id and trace_by_id) else None
        entries.append(
            {
                "sample_id": result.sample.id,
                "trace_id": trace_id,
                "executed_ok": result.selected.ok,
                "epochs": len(result.runs.epochs),
                "selected_epoch": result.selected_epoch,
                "rounds_used": result.selected.rounds_used,
                "held": result.selected.held,
                "final_correct": result.selected.final_correct,
                "overall": result.verdict.overall,
                "passed": result.verdict.passed,
                "scores": {s.dimension: s.score for s in result.verdict.scores},
                "reasons": {s.dimension: s.reason for s in result.verdict.scores},
                "deterministic": result.deterministic,
                "latency": trace.latency
                if trace
                else result.sample.metadata.get("latency"),
                "cost": trace.cost if trace else result.sample.metadata.get("cost"),
                "tool_calls": _count_tool_calls(trace),
            }
        )
    return entries


def run(
    samples: list[Sample],
    *,
    agent: Agent | None = None,
    model: str = "deepseek-chat",
    dimensions: list[DimensionSpec | dict] | None = None,
    scorers: list[ScorerSpec | dict] | None = None,
    run_id: str | None = None,
    run_dir: str | Path = "runs",
    upload: bool = True,
    name: str | None = None,
    **judge_kwargs: Any,
) -> RunReport:
    """一步到位：跑 agent -> 评分 -> 上传 -> 生成报告。"""
    configure_logging()
    resolved_run_id = run_id or make_run_id(name or "eval")
    resolved_name = name or resolved_run_id
    log_event(
        logger,
        "INFO",
        "run.start",
        run_id=resolved_run_id,
        samples=len(samples),
    )
    agent = agent or CodexAgent()

    runs = agent.run(samples, run_dir=run_dir, run_id=resolved_run_id)
    results = evaluate(
        runs, model=model, dimensions=dimensions, scorers=scorers, **judge_kwargs
    )
    publish(results, run_id=resolved_run_id, upload=upload, name=resolved_name)

    agent_config = getattr(agent, "config", None)
    epochs = getattr(agent_config, "epochs", 1) if agent_config else 1
    epoch_reducer = (
        getattr(agent_config, "epoch_reducer", "pass_at_k") if agent_config else "pass_at_k"
    )
    agent_model = getattr(agent_config, "model", None) if agent_config else None

    report = assemble_report(
        run_id=resolved_run_id,
        name=resolved_name,
        judge_model=model,
        threshold=judge_kwargs.get("threshold", 0.5),
        uploaded=upload,
        results=result_entries(results),
        verdicts=[r.verdict for r in results],
        epochs=epochs,
        epoch_reducer=epoch_reducer,
        agent_model=agent_model,
    )
    write_report(report, Path(run_dir) / resolved_run_id)
    print_summary(report)
    return report
