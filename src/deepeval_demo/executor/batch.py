"""批量执行：asyncio 并发、状态落盘、断点续跑与失败重试。"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import ExecutorConfig
from ..data import Sample
from ..log import get_logger, log_event
from .runner import ExecResult, SampleRun, run_one, run_one_multi_turn

logger = get_logger(__name__)

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(run_dir: Path) -> Path:
    return run_dir / "state.jsonl"


def _load_state(run_dir: Path) -> dict[str, dict]:
    path = _state_path(run_dir)
    state: dict[str, dict] = {}
    if not path.exists():
        return state
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("sample_id"):
            state[obj["sample_id"]] = obj
    return state


def _append_state(run_dir: Path, entry: dict) -> None:
    with _state_path(run_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_done_answer(state: dict, sample_id: str) -> tuple[bool, str | None, Path | None]:
    entry = state.get(sample_id) or {}
    answer_path = entry.get("answer_path")
    if answer_path and Path(answer_path).exists():
        answer = Path(answer_path).read_text(encoding="utf-8").strip()
        return True, answer, Path(answer_path)
    return True, "", None


def run_batch(
    samples: list[Sample],
    cfg: ExecutorConfig,
    run_dir: str | Path,
    *,
    run_id: str | None = None,
    resume: bool = True,
    epoch: int | None = None,
    metadata: dict | None = None,
) -> list[ExecResult]:
    """并发执行全部样本，返回与输入顺序一致的 ExecResult 列表。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_run_id = run_id or run_dir.name
    batch_started = time.perf_counter()

    state = _load_state(run_dir)
    to_run = [
        s
        for s in samples
        if not (resume and state.get(s.id, {}).get("status") == STATUS_DONE)
    ]
    log_event(
        logger,
        "INFO",
        "exec.batch.start",
        run_id=resolved_run_id,
        samples=len(samples),
        to_run=len(to_run),
        epochs=max(1, cfg.epochs),
        epoch=epoch,
        max_concurrent=cfg.max_concurrent,
        model=cfg.model,
    )

    fresh: dict[str, ExecResult] = {}
    if to_run:
        sem = asyncio.Semaphore(max(1, cfg.max_concurrent))

        async def worker(sample: Sample) -> ExecResult:
            async with sem:
                last: ExecResult | None = None
                for attempt in range(1, cfg.max_attempts + 1):
                    log_event(
                        logger,
                        "INFO",
                        "exec.sample.start",
                        sample_id=sample.id,
                        attempt=attempt,
                        epoch=epoch,
                    )
                    _append_state(
                        run_dir,
                        {
                            "sample_id": sample.id,
                            "status": STATUS_RUNNING,
                            "attempt": attempt,
                            "started_at": _now(),
                        },
                    )
                    if cfg.max_rounds > 1 and sample.target:
                        runner = run_one_multi_turn
                    else:
                        runner = run_one
                    attempt_started = time.perf_counter()
                    result = await asyncio.to_thread(
                        runner,
                        sample,
                        cfg,
                        run_dir,
                        run_id=resolved_run_id,
                        epoch=epoch,
                        metadata=metadata,
                    )
                    duration_ms = round(
                        (time.perf_counter() - attempt_started) * 1000, 2
                    )
                    last = result
                    if result.ok:
                        log_event(
                            logger,
                            "INFO",
                            "exec.sample.done",
                            sample_id=sample.id,
                            attempt=attempt,
                            epoch=epoch,
                            ok=True,
                            rounds_used=result.rounds_used,
                            duration_ms=duration_ms,
                            exit_code=result.exit_code,
                        )
                        _append_state(
                            run_dir,
                            {
                                "sample_id": sample.id,
                                "status": STATUS_DONE,
                                "attempt": attempt,
                            "epoch": epoch,
                                "exit_code": result.exit_code,
                                "answer_path": str(result.output_path)
                                if result.output_path
                                else None,
                                "finished_at": _now(),
                            },
                        )
                        return result
                    status = STATUS_FAILED if attempt == cfg.max_attempts else STATUS_PENDING
                    log_event(
                        logger,
                        "ERROR",
                        "exec.sample.error",
                        sample_id=sample.id,
                        attempt=attempt,
                        epoch=epoch,
                        ok=False,
                        duration_ms=duration_ms,
                        error=result.error,
                    )
                    _append_state(
                        run_dir,
                        {
                            "sample_id": sample.id,
                            "status": status,
                            "attempt": attempt,
                            "epoch": epoch,
                            "error": result.error,
                            "finished_at": _now(),
                        },
                    )
                    if attempt < cfg.max_attempts:
                        backoff_s = cfg.retry_backoff_seconds * (2 ** (attempt - 1))
                        log_event(
                            logger,
                            "WARNING",
                            "exec.retry",
                            sample_id=sample.id,
                            attempt=attempt,
                            backoff_s=backoff_s,
                        )
                        await asyncio.sleep(
                            backoff_s
                        )
                return last or ExecResult(
                    sample_id=sample.id, input=sample.input, ok=False, error="未执行"
                )

        async def run_all() -> list[ExecResult]:
            return list(await asyncio.gather(*(worker(s) for s in to_run)))

        fresh = {r.sample_id: r for r in asyncio.run(run_all())}

    state = _load_state(run_dir)
    results: list[ExecResult] = []
    for sample in samples:
        entry = state.get(sample.id, {})
        if entry.get("status") == STATUS_DONE:
            _, answer, output_path = _read_done_answer(state, sample.id)
            results.append(
                ExecResult(
                    sample_id=sample.id,
                    input=sample.input,
                    ok=True,
                    answer=answer,
                    output_path=output_path,
                )
            )
        else:
            r = fresh.get(sample.id)
            results.append(
                r
                or ExecResult(
                    sample_id=sample.id, input=sample.input, ok=False, error="未执行"
                )
            )
    log_event(
        logger,
        "INFO",
        "exec.batch.done",
        run_id=resolved_run_id,
        epoch=epoch,
        duration_ms=round((time.perf_counter() - batch_started) * 1000, 2),
        done=sum(1 for r in results if r.ok),
    )
    return results


def run_batch_epochs(
    samples: list[Sample],
    cfg: ExecutorConfig,
    run_dir: str | Path,
    *,
    run_id: str | None = None,
    resume: bool = True,
    metadata: dict | None = None,
) -> list[SampleRun]:
    """按 epochs 对每个样本多次执行，返回与输入顺序一致的 SampleRun 列表。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs = max(1, cfg.epochs)
    resolved_run_id = run_id or run_dir.name

    per_epoch: list[list[ExecResult]] = []
    for current_epoch in range(1, epochs + 1):
        sub_dir = run_dir if epochs == 1 else run_dir / f"epoch-{current_epoch}"
        epoch_started = time.perf_counter()
        log_event(
            logger,
            "INFO",
            "exec.epoch.start",
            run_id=resolved_run_id,
            epoch=current_epoch,
            samples=len(samples),
        )
        per_epoch.append(
            run_batch(
                samples,
                cfg,
                sub_dir,
                run_id=run_id,
                resume=resume,
                epoch=current_epoch,
                metadata=metadata,
            )
        )
        log_event(
            logger,
            "INFO",
            "exec.epoch.done",
            run_id=resolved_run_id,
            epoch=current_epoch,
            duration_ms=round((time.perf_counter() - epoch_started) * 1000, 2),
        )

    return [
        SampleRun(
            sample=samples[index],
            epochs=[per_epoch[e][index] for e in range(epochs)],
        )
        for index in range(len(samples))
    ]
