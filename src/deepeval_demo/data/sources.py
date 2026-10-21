"""数据读取：本地 parquet/json/jsonl 与 HuggingFace 下载。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ..config import DataSourceSpec

SUPPORTED_SUFFIXES = {".parquet", ".json", ".jsonl", ".ndjson"}


@dataclass
class Sample:
    """一条评测样本。"""

    id: str
    input: str
    target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    reference_output: str | None = None
    trace_id: str | None = None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _read_json_file(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip()
        ]
    if isinstance(data, dict):
        for key in ("data", "rows", "items", "samples", "records"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            raise ValueError(f"{path} 的 JSON 顶层是对象且未找到列表字段")
    if not isinstance(data, list):
        raise ValueError(f"{path} 的 JSON 顶层必须是列表")
    return [r for r in data if isinstance(r, dict)]


def _read_parquet_file(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _read_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return _read_parquet_file(path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return _read_json_file(path)
    raise ValueError(f"不支持的数据文件类型: {path}")


def _discover_files(root: Path, spec: DataSourceSpec) -> list[Path]:
    """从目录中找出数据文件，支持 HF 的 config/split 布局。"""
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"找不到数据集路径: {root}")

    if spec.config and spec.split:
        sub = root / spec.config
        if sub.is_dir():
            files = [
                p
                for p in sub.glob(f"{spec.split}-*")
                if p.suffix.lower() in SUPPORTED_SUFFIXES
            ]
            if files:
                return sorted(files)

    files = [
        p for p in root.glob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if files:
        return sorted(files)
    files = [
        p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files)


def download_hf(
    repo_id: str,
    local_dir: str | Path,
    *,
    repo_type: str = "dataset",
    allow_patterns: list[str] | None = None,
    token: str | None = None,
    revision: str | None = None,
) -> Path:
    """用 huggingface_hub 把数据集下载到本地目录，返回该目录。"""
    from huggingface_hub import snapshot_download

    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local),
        allow_patterns=allow_patterns,
        token=token,
        revision=revision,
    )
    return local


def _records_from_source(spec: DataSourceSpec) -> tuple[list[dict[str, Any]], Path]:
    if spec.kind == "huggingface":
        repo = spec.repo_id or spec.path
        if not repo:
            raise ValueError("huggingface 数据源需要 repo_id")
        if spec.path and Path(spec.path).is_dir():
            root = Path(spec.path)
        else:
            if spec.path:
                local = Path(spec.path)
            else:
                local = Path("mydataset") / repo.replace("/", "__")
            root = download_hf(repo, local)
    else:
        root = Path(spec.path)
        if not root.exists():
            raise FileNotFoundError(f"找不到数据集路径: {root}")

    files = _discover_files(root, spec)
    if not files:
        raise FileNotFoundError(f"目录中没有可读取的数据文件: {root}")
    records: list[dict[str, Any]] = []
    for f in files:
        records.extend(_read_file(f))
    return records, root


def load_dataset(spec: DataSourceSpec) -> list[Sample]:
    """按 DataSourceSpec 读取数据集，返回 Sample 列表。"""
    records, _ = _records_from_source(spec)
    fs = spec.field_spec

    samples: list[Sample] = []
    for index, rec in enumerate(records):
        if fs.input not in rec or rec[fs.input] is None:
            continue
        if fs.id and fs.id in rec and rec[fs.id] is not None:
            sample_id = _as_text(rec[fs.id])
        else:
            sample_id = f"sample-{index + 1:06d}"
        target = None
        if fs.target and fs.target in rec and rec[fs.target] is not None:
            target = _as_text(rec[fs.target])
        excluded = {fs.input, fs.target, fs.id}
        metadata = {k: v for k, v in rec.items() if k not in excluded}
        samples.append(
            Sample(
                id=sample_id,
                input=_as_text(rec[fs.input]),
                target=target,
                metadata=metadata,
            )
        )

    if spec.limit is not None:
        samples = samples[: spec.limit]
    return samples
