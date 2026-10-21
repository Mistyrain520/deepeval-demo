"""yaml/config 驱动的批跑入口：把 EvalConfig 映射到 api 门面，彻底统一。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .api import CodexAgent, evaluate, make_run_id, publish, result_entries
from .config import EvalConfig
from .data import Sample, load_dataset
from .langfuse import (
    LangfuseConfigError,
    TraceRecord,
    build_client,
    build_trace_index,
    fetch_traces,
    load_langfuse_config,
)
from .report import RunReport, assemble_report, print_summary, write_report


def _codex_version() -> str | None:
    try:
        proc = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        if proc.returncode == 0:
            return (proc.stdout or "").strip() or None
    except Exception:  # noqa: BLE001
        return None
    return None


def _dataset_hash(samples: list[Sample]) -> str:
    payload = json.dumps(
        [[s.id, s.input, s.target] for s in samples],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _fetch_traces(config: EvalConfig, environment: str):
    lf_client = None
    traces: list[TraceRecord] = []
    lf_error: str | None = None
    try:
        lf_client = build_client(load_langfuse_config())
        traces = fetch_traces(
            lf_client,
            environment=environment,
            limit=config.langfuse.fetch_limit,
            include_observations=True,
        )
    except LangfuseConfigError as exc:
        lf_error = str(exc)
    except Exception as exc:  # noqa: BLE001
        lf_error = str(exc)
    return lf_client, traces, lf_error


def run_eval(config: EvalConfig) -> RunReport:
    """执行完整评测流程，返回 RunReport（同时落盘 run_dir/report.json）。"""
    run_id = config.run.run_id or make_run_id(config.name)
    run_dir = Path(config.run.run_dir)

    samples = load_dataset(config.dataset)
    dataset_hash = _dataset_hash(samples)
    codex_version = _codex_version()

    agent = CodexAgent(
        model=config.agent.model,
        prompt_template=config.agent.prompt_template,
        project_root=config.agent.project_root,
        sandbox=config.agent.sandbox,
        max_concurrent=config.agent.max_concurrent,
        max_attempts=config.agent.max_attempts,
        epochs=config.agent.epochs,
        epoch_reducer=config.agent.epoch_reducer,
        pass_k=config.agent.pass_k,
        seed=config.agent.seed,
        retry_backoff_seconds=config.agent.retry_backoff_seconds,
        timeout_seconds=config.agent.timeout_seconds,
        ephemeral=config.agent.ephemeral,
        codex_path=config.agent.codex_path,
        max_rounds=config.agent.max_rounds,
        system_prompt=config.agent.system_prompt,
        challenge_message=config.agent.challenge_message,
        final_answer_pattern=config.agent.final_answer_pattern,
    )
    runs = agent.run(
        samples,
        run_dir=run_dir,
        run_id=run_id,
        resume=config.run.resume,
    )

    environment = f"{config.langfuse.environment_prefix}-{run_id}"
    lf_client, traces, lf_error = _fetch_traces(config, environment)
    if lf_error:
        print(f"[langfuse] {lf_error}")
    trace_index = build_trace_index(traces, prefer_latest_round=True)
    trace_by_id = {t.trace_id: t for t in traces}
    observations_by_sample: dict[str, list[dict]] = {}
    for trace in traces:
        sample_id = trace.metadata.get("sample_id")
        if sample_id:
            observations_by_sample.setdefault(str(sample_id), trace.observations)

    results = evaluate(
        runs,
        model=config.judge.model,
        dimensions=config.judge.dimensions,
        scorers=config.scorers,
        threshold=config.judge.threshold,
        evidence=config.judge.evidence,
        scoring=config.judge.scoring,
        max_concurrent=config.judge.max_concurrent,
        throttle_seconds=config.judge.throttle_seconds,
        max_retries=config.judge.max_retries,
        epoch_reducer=config.agent.epoch_reducer,
        pass_k=config.agent.pass_k,
        observations_by_sample=observations_by_sample,
    )

    publish(
        results,
        run_id=run_id,
        upload=config.langfuse.upload,
        client=lf_client,
        name=config.name,
        traces=traces,
    )

    entries = result_entries(
        results, trace_index=trace_index, trace_by_id=trace_by_id
    )
    report = assemble_report(
        run_id=run_id,
        name=config.name,
        judge_model=config.judge.model,
        threshold=config.judge.threshold,
        uploaded=config.langfuse.upload,
        results=entries,
        verdicts=[r.verdict for r in results],
        epochs=max(1, config.agent.epochs),
        epoch_reducer=config.agent.epoch_reducer,
        agent_model=config.agent.model,
        codex_version=codex_version,
        dataset_hash=dataset_hash,
        seed=config.agent.seed,
        total_traces=len(traces),
    )
    write_report(report, run_dir / run_id)
    print_summary(report)
    return report
