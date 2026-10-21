"""JSON 记录 -> HuggingFace 风格 parquet 数据集。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


def _dataset_card(name: str, split: str, table: pa.Table) -> str:
    features = "\n".join(
        f"  - name: {field.name}\n    dtype: {field.type}" for field in table.schema
    )
    return (
        "---\n"
        f"pretty_name: {name}\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        f"  - split: {split}\n"
        f"    path: {split}-*\n"
        "---\n\n"
        f"# Dataset Card for {name}\n\n"
        "## Dataset Structure\n\n"
        "### Data Fields\n\n"
        f"{features}\n\n"
        "### Data Splits\n\n"
        f"| name | {split} |\n"
        "|------|---------|\n"
        f"| default | {table.num_rows} |\n"
    )


def json_to_parquet(
    records: Iterable[dict[str, Any]],
    out_dir: str | Path,
    *,
    name: str,
    split: str = "train",
    schema: pa.Schema | None = None,
    write_card: bool = True,
) -> Path:
    """把 dict 列表写成 parquet（可选附带轻量 dataset 卡片），返回文件路径。"""
    rows = list(records)
    if not rows:
        raise ValueError("records 不能为空")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{split}-00000-of-00001.parquet"
    if schema is not None:
        table = pa.Table.from_pylist(rows, schema=schema)
    else:
        table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    if write_card:
        (out / "README.md").write_text(
            _dataset_card(name, split, table), encoding="utf-8"
        )
    return out_path
