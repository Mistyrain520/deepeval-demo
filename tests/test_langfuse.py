"""Langfuse 模块参考测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from deepeval_demo import (
    ExecResult,
    JudgeVerdict,
    LangfuseConfigError,
    ScoreResult,
    TraceRecord,
    build_trace_index,
    load_langfuse_config,
    resolve_trace_ids,
    upload_scores,
)


def test_load_config_from_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://x")
    creds = load_langfuse_config()
    assert creds.public_key == "pk"
    assert creds.secret_key == "sk"
    assert creds.base_url == "http://x"


def test_load_config_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)

    class FakePath:
        @staticmethod
        def home() -> Path:
            return tmp_path

        @staticmethod
        def cwd() -> Path:
            return tmp_path

    monkeypatch.setattr("deepeval_demo.langfuse.client.Path", FakePath)
    with pytest.raises(LangfuseConfigError):
        load_langfuse_config()


def test_resolve_trace_ids():
    exec_results = [
        ExecResult(sample_id="s1", input="q1", ok=True, answer="a1"),
        ExecResult(sample_id="s2", input="q2", ok=True, answer="a2"),
    ]
    traces = [
        TraceRecord(trace_id="t1", metadata={"sample_id": "s1"}),
        TraceRecord(trace_id="t2", input="q2", output="a2"),
    ]
    assert resolve_trace_ids(exec_results, traces) == {"s1": "t1", "s2": "t2"}


def test_build_trace_index_prefer_latest_round():
    traces = [
        TraceRecord(trace_id="t1", metadata={"sample_id": "s1", "epoch": "1", "round": "1"}),
        TraceRecord(trace_id="t2", metadata={"sample_id": "s1", "epoch": "1", "round": "3"}),
        TraceRecord(trace_id="t3", metadata={"sample_id": "s1", "epoch": "1", "round": "2"}),
    ]
    index = build_trace_index(traces, prefer_latest_round=True)
    assert index[("s1", "1")] == "t2"

    default = build_trace_index(traces)
    assert default[("s1", "1")] == "t1"


def test_upload_scores():
    calls = []

    class FakeClient:
        def create_score(self, **kwargs):
            calls.append(kwargs)

    verdict = JudgeVerdict(
        scores=[ScoreResult("relevance", 0.9, "r", True)],
        overall=0.9,
        passed=True,
    )
    records = upload_scores(
        FakeClient(), "t1", verdict, {"sample_id": "s1"}, max_retries=1
    )
    assert len(calls) == 2
    assert records[0].uploaded is True
    assert calls[0]["name"] == "judge_relevance"
    assert calls[1]["name"] == "judge_overall"
