"""批量执行：asyncio 并发、状态落盘、断点续跑与失败重试。"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import ExecutorConfig
from ..data import Sample
from .runner import ExecResult, SampleRun, run_one, run_one_multi_turn

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

    state = _load_state(run_dir)
    to_run = [
        s
        for s in samples
        if not (resume and state.get(s.id, {}).get("status") == STATUS_DONE)
    ]

    fresh: dict[str, ExecResult] = {}
    if to_run:
        sem = asyncio.Semaphore(max(1, cfg.max_concurrent))

        async def worker(sample: Sample) -> ExecResult:
            async with sem:
                last: ExecResult | None = None
                for attempt in range(1, cfg.max_attempts + 1):
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
                    result = await asyncio.to_thread(
                        runner,
                        sample,
                        cfg,
                        run_dir,
                        run_id=resolved_run_id,
                        epoch=epoch,
                        metadata=metadata,
                    )
                    last = result
                    if result.ok:
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
                        await asyncio.sleep(
                            cfg.retry_backoff_seconds * (2 ** (attempt - 1))
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

    per_epoch: list[list[ExecResult]] = []
    for current_epoch in range(1, epochs + 1):
        sub_dir = run_dir if epochs == 1 else run_dir / f"epoch-{current_epoch}"
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

    return [
        SampleRun(
            sample=samples[index],
            epochs=[per_epoch[e][index] for e in range(epochs)],
        )
        for index in range(len(samples))
    ]
