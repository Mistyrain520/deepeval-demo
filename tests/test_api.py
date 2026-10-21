"""函数式四步门面的离线测试。"""

from __future__ import annotations

import json
import re

from deepeval_demo import (
    CodexAgent,
    ExecResult,
    JudgeVerdict,
    Sample,
    SampleResult,
    SampleRun,
    ScoreResult,
    ScorerSpec,
    TraceRecord,
    evaluate,
    hf_dataset,
    langfuse_dataset,
    local_dataset,
    publish,
    run,
    make_run_id,
)


def test_local_dataset(tmp_path):
    from deepeval_demo import json_to_parquet

    json_to_parquet(
        [{"question": "1+1", "answer": "2"}],
        tmp_path / "ds",
        name="mini",
        split="test",
        write_card=False,
    )
    samples = local_dataset(
        str(tmp_path / "ds"),
        field_spec={"input": "question", "target": "answer"},
    )
    assert len(samples) == 1
    assert samples[0].input == "1+1"
    assert samples[0].target == "2"


def test_hf_dataset_download(monkeypatch, tmp_path):
    calls = []

    def fake_download(repo_id, local_dir, *, token=None, **kwargs):
        calls.append(repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "data.json").write_text(
            json.dumps([{"q": "2+2", "a": "4"}]), encoding="utf-8"
        )
        return local_dir

    monkeypatch.setattr("deepeval_demo.api.download_hf", fake_download)
    samples = hf_dataset(
        "org/ds",
        local_dir=tmp_path / "hf",
        field_spec={"input": "q", "target": "a"},
    )
    assert calls == ["org/ds"]
    assert samples[0].input == "2+2"


def test_langfuse_dataset(monkeypatch):
    monkeypatch.setattr(
        "deepeval_demo.api.load_langfuse_config", lambda: None
    )
    monkeypatch.setattr("deepeval_demo.api.build_client", lambda creds: object())
    monkeypatch.setattr(
        "deepeval_demo.api.fetch_traces",
        lambda client, **kwargs: [
            TraceRecord(trace_id="t1", input="q", output="a", model="m")
        ],
    )
    samples = langfuse_dataset(limit=1)
    assert len(samples) == 1
    assert samples[0].trace_id == "t1"
    assert samples[0].reference_output == "a"
    assert samples[0].metadata["model"] == "m"


def test_codex_agent_skips_preanswered(monkeypatch):
    executed = []

    def fake_run_batch_epochs(samples, cfg, run_dir, **kwargs):
        executed.extend(samples)
        return [
            SampleRun(
                sample=s,
                epochs=[ExecResult(sample_id=s.id, input=s.input, ok=True, answer="ans")],
            )
            for s in samples
        ]

    monkeypatch.setattr("deepeval_demo.api.run_batch_epochs", fake_run_batch_epochs)
    agent = CodexAgent()
    pre = Sample(id="pre", input="q1", reference_output="已有答案", trace_id="t1")
    fresh = Sample(id="fresh", input="q2")
    runs = agent.run([pre, fresh], run_id="r")
    assert len(runs) == 2
    assert runs[0].epochs[0].answer == "已有答案"
    assert [s.id for s in executed] == ["fresh"]


def test_codex_agent_forwards_multiturn_config():
    agent = CodexAgent(
        model="deepseek-v4-flash",
        max_rounds=5,
        system_prompt="请直接给我最终答案，不要任何分析过程",
        challenge_message="再算一次",
        final_answer_pattern=r"ANS\s*=\s*(.+)$",
    )
    assert agent.config.max_rounds == 5
    assert agent.config.system_prompt == "请直接给我最终答案，不要任何分析过程"
    assert agent.config.challenge_message == "再算一次"
    assert agent.config.final_answer_pattern == r"ANS\s*=\s*(.+)$"


def test_evaluate(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def fake_judge_batch(inputs, cfg):
        return [
            JudgeVerdict(
                scores=[
                    ScoreResult(d.name, 0.9, "ok", True) for d in cfg.dimensions
                ],
                overall=0.9,
                passed=True,
            )
            for _ in inputs
        ]

    monkeypatch.setattr("deepeval_demo.api.judge_batch", fake_judge_batch)
    runs = [
        SampleRun(
            sample=Sample(id="s1", input="q", target="42"),
            epochs=[
                ExecResult(sample_id="s1", input="q", ok=True, answer="40"),
                ExecResult(sample_id="s1", input="q", ok=True, answer="42"),
            ],
        )
    ]
    results = evaluate(
        runs,
        scorers=[ScorerSpec(name="num", type="numeric", threshold=1.0)],
        epoch_reducer="pass_at_k",
        pass_k=2,
    )
    assert len(results) == 1
    assert results[0].selected_epoch == 2
    assert results[0].deterministic["num"]["pass_at_k"] is True
    assert results[0].verdict.overall == 0.9


def test_publish_resolves_and_uploads(monkeypatch):
    calls = []

    class FakeClient:
        def create_score(self, **kwargs):
            calls.append(kwargs)

    verdict = JudgeVerdict(
        scores=[ScoreResult("relevance", 0.9, "r", True)],
        overall=0.9,
        passed=True,
    )
    results = [
        SampleResult(
            sample=Sample(id="s1", input="q", trace_id="t1"),
            runs=SampleRun(
                sample=Sample(id="s1", input="q", trace_id="t1"),
                epochs=[ExecResult(sample_id="s1", input="q", ok=True, answer="a")],
            ),
            selected=ExecResult(sample_id="s1", input="q", ok=True, answer="a"),
            deterministic={},
            verdict=verdict,
        ),
        SampleResult(
            sample=Sample(id="s2", input="q2"),
            runs=SampleRun(
                sample=Sample(id="s2", input="q2"),
                epochs=[ExecResult(sample_id="s2", input="q2", ok=True, answer="b")],
            ),
            selected=ExecResult(sample_id="s2", input="q2", ok=True, answer="b"),
            deterministic={},
            verdict=verdict,
        ),
    ]
    traces = [TraceRecord(trace_id="t2", metadata={"sample_id": "s2", "epoch": "1"})]
    records = publish(results, run_id="run1", client=FakeClient(), traces=traces)

    assert len(calls) == 4
    assert {c["trace_id"] for c in calls} == {"t1", "t2"}
    assert {c["name"] for c in calls} == {"judge_relevance", "judge_overall"}
    assert all(c["metadata"]["evaluation_run_id"] == "run1" for c in calls)
    assert len(records) == 4


def test_run_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def fake_agent_run(self, samples, *, run_dir="runs", run_id=None, resume=False):
        return [
            SampleRun(
                sample=s,
                epochs=[ExecResult(sample_id=s.id, input=s.input, ok=True, answer="a")],
            )
            for s in samples
        ]

    def fake_judge_batch(inputs, cfg):
        return [
            JudgeVerdict(
                scores=[
                    ScoreResult(d.name, 0.7, "ok", True) for d in cfg.dimensions
                ],
                overall=0.7,
                passed=True,
            )
            for _ in inputs
        ]

    monkeypatch.setattr("deepeval_demo.api.CodexAgent.run", fake_agent_run)
    monkeypatch.setattr("deepeval_demo.api.judge_batch", fake_judge_batch)
    monkeypatch.setattr(
        "deepeval_demo.api.publish", lambda results, **kwargs: []
    )

    report = run(
        [Sample(id="s1", input="q")],
        run_id="r",
        run_dir=str(tmp_path),
        upload=False,
    )
    assert report.total_samples == 1
    assert report.overall_average == 0.7
    assert (tmp_path / "r" / "report.json").exists()


def test_make_run_id_format():
    rid = make_run_id("codex-gsm8k")
    assert re.fullmatch(r"codex-gsm8k-\d{8}-\d{4}", rid)
