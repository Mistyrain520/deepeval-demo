"""数据模块参考测试。"""

from __future__ import annotations

import pytest

from deepeval_demo import (
    DataSourceSpec,
    FieldSpec,
    json_to_parquet,
    load_dataset,
)


def test_json_to_parquet_roundtrip(tmp_path):
    records = [
        {"question": "1+1?", "answer": "2"},
        {"question": "2+2?", "answer": "4"},
    ]
    json_to_parquet(records, tmp_path / "ds", name="mini", split="train")

    spec = DataSourceSpec(
        kind="local",
        path=str(tmp_path / "ds"),
        field_spec=FieldSpec(input="question", target="answer"),
    )
    samples = load_dataset(spec)
    assert [s.input for s in samples] == ["1+1?", "2+2?"]
    assert [s.target for s in samples] == ["2", "4"]


def test_load_local_hf_layout(tmp_path):
    sub = tmp_path / "main"
    sub.mkdir()
    json_to_parquet(
        [{"question": "q1", "answer": "a1"}],
        sub,
        name="mini",
        split="test",
        write_card=False,
    )
    spec = DataSourceSpec(
        kind="local",
        path=str(tmp_path),
        config="main",
        split="test",
        field_spec=FieldSpec(input="question", target="answer"),
    )
    samples = load_dataset(spec)
    assert len(samples) == 1
    assert samples[0].input == "q1"
    assert samples[0].target == "a1"


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset(DataSourceSpec(kind="local", path="no/such/path"))
