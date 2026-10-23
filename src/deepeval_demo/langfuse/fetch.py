"""从 Langfuse 拉取 trace，并与执行结果做关联。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..executor import ExecResult
from ..log import get_logger, log_event

FIELDS = "core,basic,time,io,metadata,model,usage,trace_context"
logger = get_logger(__name__)


@dataclass
class TraceRecord:
    """一条 Langfuse trace（root observation）的结构化表示。"""

    trace_id: str
    name: str | None = None
    start_time: datetime | None = None
    input: str = ""
    output: str = ""
    model: str | None = None
    latency: float | None = None
    cost: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    observations: list[dict[str, Any]] = field(default_factory=list)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            return value["content"]
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _to_trace_record(obs: Any) -> TraceRecord:
    metadata = _as_dict(getattr(obs, "metadata", None))
    usage = _as_dict(getattr(obs, "usage_details", None)) or {
        "input": getattr(obs, "input_usage", None),
        "output": getattr(obs, "output_usage", None),
        "total": getattr(obs, "total_usage", None),
    }
    return TraceRecord(
        trace_id=obs.trace_id,
        name=getattr(obs, "trace_name", None) or getattr(obs, "name", None),
        start_time=getattr(obs, "start_time", None),
        input=_to_text(getattr(obs, "input", None)),
        output=_to_text(getattr(obs, "output", None)),
        model=getattr(obs, "model", None) or None,
        latency=getattr(obs, "latency", None),
        cost=getattr(obs, "total_cost", None),
        usage={k: v for k, v in usage.items() if v is not None},
        metadata=metadata,
    )


def fetch_trace_observations(client: Any, trace_id: str) -> list[dict[str, Any]]:
    """拉取某条 trace 下的全部 observation，返回 JSON 可序列化字典列表。"""
    collected: list[Any] = []
    seen: set[str] = set()
    cursor = None
    while True:
        response = client.api.observations.get_many(
            fields=FIELDS,
            trace_id=trace_id,
            limit=50,
            cursor=cursor,
        )
        for obs in response.data:
            if obs.id not in seen:
                seen.add(obs.id)
                collected.append(obs)
        cursor = response.meta.cursor if response.meta else None
        if not cursor or not response.data:
            break
    collected.sort(
        key=lambda o: o.start_time
        or datetime.min.replace(tzinfo=timezone.utc)
    )
    return [obs.model_dump(mode="json") for obs in collected]


def fetch_traces(
    client: Any,
    *,
    environment: str | None = None,
    limit: int | None = None,
    include_observations: bool = True,
    name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    from_time: datetime | None = None,
    to_time: datetime | None = None,
) -> list[TraceRecord]:
    """分页拉取 root observation（即 trace 列表），可选附带全量 observation。"""
    started = time.perf_counter()
    log_event(
        logger,
        "INFO",
        "langfuse.fetch.start",
        environment=environment,
        limit=limit,
    )
    collected: list[TraceRecord] = []
    seen: set[str] = set()
    cursor = None
    while True:
        page_size = 50 if limit is None else min(50, limit - len(collected))
        if page_size <= 0:
            break
        response = client.api.observations.get_many(
            fields=FIELDS,
            limit=page_size,
            cursor=cursor,
            is_root_observation=True,
            environment=environment,
            name=name,
            user_id=user_id,
            session_id=session_id,
            from_start_time=from_time,
            to_start_time=to_time,
        )
        for obs in response.data:
            if obs.trace_id in seen:
                continue
            seen.add(obs.trace_id)
            collected.append(_to_trace_record(obs))
        cursor = response.meta.cursor if response.meta else None
        if not cursor or not response.data:
            break
        if limit is not None and len(collected) >= limit:
            break

    if include_observations:
        for record in collected:
            record.observations = fetch_trace_observations(client, record.trace_id)
    log_event(
        logger,
        "INFO",
        "langfuse.fetch.done",
        environment=environment,
        limit=limit,
        trace_count=len(collected),
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return collected


def build_trace_index(
    traces: list[TraceRecord],
    *,
    prefer_latest_round: bool = False,
) -> dict[tuple[str, str | None], str]:
    """按 (sample_id, epoch) 建 trace 索引，供 epochs 场景精确关联。

    ``prefer_latest_round=True`` 时，同一 (sample_id, epoch) 取 ``round`` 元数据
    最大的 trace（缺省按 1），用于多轮评测把分数回传到最后答轮。
    """
    index: dict[tuple[str, str | None], str] = {}
    best_round: dict[tuple[str, str | None], int] = {}
    for trace in traces:
        sample_id = trace.metadata.get("sample_id")
        if sample_id is None:
            continue
        sample_id = str(sample_id)
        epoch_raw = trace.metadata.get("epoch")
        epoch = str(epoch_raw) if epoch_raw is not None else None
        key = (sample_id, epoch)
        if not prefer_latest_round:
            index.setdefault(key, trace.trace_id)
            continue
        try:
            round_num = int(trace.metadata.get("round", 1))
        except (TypeError, ValueError):
            round_num = 1
        if key not in index or round_num >= best_round[key]:
            index[key] = trace.trace_id
            best_round[key] = round_num
    return index


def resolve_trace_ids(
    exec_results: list[ExecResult],
    traces: list[TraceRecord],
    *,
    epoch: int | None = None,
) -> dict[str, str]:
    """建立 sample_id -> trace_id 的映射：metadata 优先，input 文本回退。"""
    index = build_trace_index(traces)
    first_by_sample: dict[str, str] = {}
    for (sample_id, _), trace_id in index.items():
        first_by_sample.setdefault(sample_id, trace_id)

    resolved: dict[str, str] = {}
    for result in exec_results:
        key_epoch = str(epoch) if epoch is not None else None
        trace_id = index.get((result.sample_id, key_epoch))
        if trace_id is None and epoch is None:
            trace_id = first_by_sample.get(result.sample_id)
        if trace_id:
            resolved[result.sample_id] = trace_id
            continue
        for trace in traces:
            if trace.input and result.input:
                if result.input in trace.input or trace.input in result.input:
                    resolved[result.sample_id] = trace.trace_id
                    break
    return resolved
