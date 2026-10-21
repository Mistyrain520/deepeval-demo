"""run_eval 编排链路的离线集成测试。"""

from __future__ import annotations

from deepeval_demo import (
    DataSourceSpec,
    EvalConfig,
    ExecResult,
    ExecutorConfig,
    FieldSpec,
    JudgeConfig,
    JudgeVerdict,
    RunConfig,
    Sample,
    SampleRun,
    ScoreResult,
    ScorerSpec,
    TraceRecord,
    assemble_report,
    json_to_parquet,
    run_eval,
)


def test_run_eval_offline(monkeypatch, tmp_path):
    json_to_parquet(
        [{"question": "1+1", "answer": "2"}, {"question": "2+2", "answer": "4"}],
        tmp_path / "ds",
        name="mini",
        split="test",
        write_card=False,
    )
    config = EvalConfig(
        name="mini",
        dataset=DataSourceSpec(
            kind="local",
            path=str(tmp_path / "ds"),
            field_spec=FieldSpec(input="question", target="answer"),
        ),
        agent=ExecutorConfig(
            model="test-model",
            epochs=2,
            epoch_reducer="pass_at_k",
            max_concurrent=1,
        ),
        judge=JudgeConfig(),
        scorers=[
            ScorerSpec(
                name="num", type="numeric", threshold=1.0, params={"tolerance": 1e-9}
            )
        ],
        run=RunConfig(run_dir=str(tmp_path / "runs"), run_id="test-run", resume=False),
    )
    config.langfuse.upload = True

    runs = [
        SampleRun(
            sample=Sample(id="s1", input="1+1", target="2"),
            epochs=[
                ExecResult(sample_id="s1", input="1+1", ok=True, answer="2"),
                ExecResult(sample_id="s1", input="1+1", ok=True, answer="2"),
            ],
        ),
        SampleRun(
            sample=Sample(id="s2", input="2+2", target="4"),
            epochs=[
                ExecResult(sample_id="s2", input="2+2", ok=True, answer="3"),
                ExecResult(sample_id="s2", input="2+2", ok=True, answer="4"),
            ],
        ),
    ]

    def fake_agent_run(self, samples, *, run_dir="runs", run_id=None, resume=False):
        return runs

    monkeypatch.setattr("deepeval_demo.api.CodexAgent.run", fake_agent_run)

    def fake_judge_batch(inputs, cfg):
        dims = cfg.dimensions
        return [
            JudgeVerdict(
                scores=[ScoreResult(d.name, 0.8, "r", True) for d in dims],
                overall=0.8,
                passed=True,
            )
            for _ in inputs
        ]

    monkeypatch.setattr("deepeval_demo.api.judge_batch", fake_judge_batch)

    class FakeClient:
        pass

    trace = TraceRecord(
        trace_id="t1",
        input="1+1",
        output="2",
        metadata={"sample_id": "s1", "epoch": "1"},
    )
    monkeypatch.setattr(
        "deepeval_demo.orchestrate._fetch_traces",
        lambda config, env: (FakeClient(), [trace], None),
    )

    uploaded = []
    monkeypatch.setattr(
        "deepeval_demo.orchestrate.publish",
        lambda results, **kwargs: uploaded.append(kwargs) or [],
    )

    report = run_eval(config)

    assert report.total_samples == 2
    assert report.epochs == 2
    assert report.epoch_reducer == "pass_at_k"
    assert report.deterministic_stats["num"]["pass_at_k_rate"] == 1.0
    assert report.codex_version
    assert len(uploaded) == 1


def test_assemble_report_multiturn_stats():
    verdicts = [
        JudgeVerdict(
            scores=[ScoreResult("relevance", 0.8, "r", True)],
            overall=0.8,
            passed=True,
        )
        for _ in range(2)
    ]
    report = assemble_report(
        run_id="r",
        name="r",
        judge_model="deepseek-chat",
        threshold=0.5,
        uploaded=False,
        results=[
            {"sample_id": "s1", "rounds_used": 2, "held": True},
            {"sample_id": "s2", "rounds_used": 5, "held": False},
        ],
        verdicts=verdicts,
    )
    assert report.held_rate == 0.5
    assert report.mean_rounds == 3.5
