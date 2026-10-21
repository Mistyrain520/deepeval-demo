"""执行模块参考测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace

from deepeval_demo import (
    ExecResult,
    ExecutorConfig,
    Sample,
    build_env,
    check_final_answer,
    run_batch,
    run_batch_epochs,
    run_one_multi_turn,
    safe_format,
)


def test_build_env():
    cfg = ExecutorConfig(model="m")
    sample = Sample(id="s1", input="q")
    env = build_env("run1", sample, cfg)
    assert env["LANGFUSE_CODEX_ENVIRONMENT"] == "eval-run1"
    assert json.loads(env["LANGFUSE_CODEX_METADATA"])["sample_id"] == "s1"
    assert json.loads(env["LANGFUSE_CODEX_TAGS"]) == ["evaluation", "run1"]


def test_build_env_round():
    cfg = ExecutorConfig(model="m")
    sample = Sample(id="s1", input="q")
    env = build_env("run1", sample, cfg, round=3)
    assert json.loads(env["LANGFUSE_CODEX_METADATA"])["round"] == "3"


def test_safe_format():
    assert safe_format("{input} -> {target}", input="q", target="a") == "q -> a"
    assert safe_format("{input} {unknown}", input="q") == "q {unknown}"


def test_run_batch_resume(monkeypatch, tmp_path):
    import deepeval_demo.executor.batch as batch

    calls = []

    def fake_run_one(sample, cfg, run_dir, **kwargs):
        calls.append(sample.id)
        return ExecResult(
            sample_id=sample.id, input=sample.input, ok=True, answer="ans"
        )

    monkeypatch.setattr(batch, "run_one", fake_run_one)
    cfg = ExecutorConfig(max_concurrent=2, max_attempts=1)
    samples = [Sample(id=f"s{i}", input=f"q{i}") for i in range(3)]

    run_batch(samples, cfg, tmp_path, resume=True)
    assert sorted(calls) == ["s0", "s1", "s2"]

    second = run_batch(samples, cfg, tmp_path, resume=True)
    assert sorted(calls) == ["s0", "s1", "s2"]  # 已完成样本不再执行
    assert [r.ok for r in second] == [True, True, True]


def test_run_batch_retry(monkeypatch, tmp_path):
    import deepeval_demo.executor.batch as batch

    calls = []

    def fake_run_one(sample, cfg, run_dir, **kwargs):
        calls.append(sample.id)
        return ExecResult(sample_id=sample.id, input=sample.input, ok=False, error="boom")

    monkeypatch.setattr(batch, "run_one", fake_run_one)
    cfg = ExecutorConfig(
        max_concurrent=1, max_attempts=3, retry_backoff_seconds=0.0
    )
    results = run_batch([Sample(id="s1", input="q")], cfg, tmp_path, resume=False)
    assert len(calls) == 3
    assert results[0].ok is False


def test_run_batch_epochs(monkeypatch, tmp_path):
    import deepeval_demo.executor.batch as batch

    calls = []

    def fake_run_one(sample, cfg, run_dir, *, run_id=None, epoch=None, metadata=None):
        calls.append((sample.id, epoch))
        return ExecResult(
            sample_id=sample.id, input=sample.input, ok=True, answer="ans"
        )

    monkeypatch.setattr(batch, "run_one", fake_run_one)
    cfg = ExecutorConfig(max_concurrent=2, max_attempts=1, epochs=3)
    samples = [Sample(id="s0", input="q0"), Sample(id="s1", input="q1")]

    runs = run_batch_epochs(samples, cfg, tmp_path, resume=False)
    assert len(runs) == 2
    assert len(runs[0].epochs) == 3
    assert sorted(calls) == [
        ("s0", 1),
        ("s0", 2),
        ("s0", 3),
        ("s1", 1),
        ("s1", 2),
        ("s1", 3),
    ]


def test_resolve_codex():
    from deepeval_demo.executor.runner import resolve_codex

    assert resolve_codex(ExecutorConfig(codex_path="C:/tools/codex.exe")) == (
        "C:/tools/codex.exe"
    )
    resolved = resolve_codex(ExecutorConfig())
    assert isinstance(resolved, str) and resolved


def test_check_final_answer():
    assert check_final_answer("#### 16", "Janet has #### 16")
    assert check_final_answer("16", "#### 16")
    assert check_final_answer("#### 1,600", "#### 1600")
    assert check_final_answer("#### $3.5", "#### 3.5")
    assert not check_final_answer("#### 15", "#### 16")
    assert not check_final_answer("whatever", None)


def test_check_final_answer_custom_pattern():
    assert check_final_answer("ANS=42", "Q? ANS=42", r"ANS\s*=\s*(.+)$")
    assert not check_final_answer("ANS=41", "Q? ANS=42", r"ANS\s*=\s*(.+)$")


def _fake_codex_process(calls, answer_by_round, thread_id="thread-1"):
    def fake_run(cmd, env=None, cwd=None, **kwargs):
        calls.append({"cmd": list(cmd), "env": env or {}, "cwd": cwd})
        out_idx = cmd.index("-o") + 1
        out_path = cmd[out_idx]
        stdout = (
            json.dumps({"type": "thread.started", "thread_id": thread_id}) + "\n"
        )
        if ".round-" in str(out_path):
            round_num = int(str(out_path).rsplit(".round-", 1)[1].split(".", 1)[0])
        else:
            round_num = 1
        answer = answer_by_round(round_num)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(answer)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    return fake_run


def test_run_one_multi_turn_holds(monkeypatch, tmp_path):
    import deepeval_demo.executor.runner as runner

    calls = []
    answers = {1: "#### 4", 2: "#### 4", 3: "#### 4"}
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        _fake_codex_process(calls, lambda r: answers[r]),
    )
    cfg = ExecutorConfig(max_rounds=2, max_attempts=1)
    sample = Sample(id="s1", input="q", target="#### 4")
    result = run_one_multi_turn(sample, cfg, tmp_path, run_id="r")

    assert result.ok is True
    assert result.answer == "#### 4"
    assert result.rounds_used == 2
    assert result.held is True
    assert result.final_correct is True
    assert result.thread_id == "thread-1"
    assert len(calls) == 2
    resume_cmd = calls[1]["cmd"]
    assert "resume" in resume_cmd
    assert "thread-1" in resume_cmd
    assert cfg.challenge_message in resume_cmd


def test_run_one_multi_turn_caves(monkeypatch, tmp_path):
    import deepeval_demo.executor.runner as runner

    calls = []
    answers = {1: "#### 4", 2: "#### 3", 3: "#### 4"}
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        _fake_codex_process(calls, lambda r: answers[r]),
    )
    cfg = ExecutorConfig(max_rounds=5, max_attempts=1)
    sample = Sample(id="s1", input="q", target="#### 4")
    result = run_one_multi_turn(sample, cfg, tmp_path, run_id="r")

    assert result.ok is True
    assert result.answer == "#### 3"
    assert result.rounds_used == 2
    assert result.held is False
    assert result.final_correct is False
    assert len(calls) == 2


def test_run_one_multi_turn_exhausted(monkeypatch, tmp_path):
    import deepeval_demo.executor.runner as runner

    calls = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        _fake_codex_process(calls, lambda r: "#### 4"),
    )
    cfg = ExecutorConfig(max_rounds=3, max_attempts=1)
    sample = Sample(id="s1", input="q", target="#### 4")
    result = run_one_multi_turn(sample, cfg, tmp_path, run_id="r")

    assert result.ok is True
    assert result.answer == "#### 4"
    assert result.rounds_used == 3
    assert result.held is True
    assert result.final_correct is True
    assert len(calls) == 3


def test_run_one_multi_turn_wrong_first(monkeypatch, tmp_path):
    import deepeval_demo.executor.runner as runner

    calls = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        _fake_codex_process(calls, lambda r: "#### 3"),
    )
    cfg = ExecutorConfig(max_rounds=5, max_attempts=1)
    sample = Sample(id="s1", input="q", target="#### 4")
    result = run_one_multi_turn(sample, cfg, tmp_path, run_id="r")

    assert result.ok is True
    assert result.rounds_used == 1
    assert result.held is None
    assert result.final_correct is False
    assert len(calls) == 1


def test_run_one_multi_turn_single_round(monkeypatch, tmp_path):
    import deepeval_demo.executor.runner as runner

    calls = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        _fake_codex_process(calls, lambda r: "#### 4"),
    )
    cfg = ExecutorConfig(max_rounds=1, max_attempts=1)
    sample = Sample(id="s1", input="q", target="#### 4")
    result = run_one_multi_turn(sample, cfg, tmp_path, run_id="r")

    assert result.rounds_used == 1
    assert result.held is None
    assert result.final_correct is True
    assert len(calls) == 1


def test_system_prompt_prepended(monkeypatch, tmp_path):
    import deepeval_demo.executor.runner as runner

    calls = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        _fake_codex_process(calls, lambda r: "#### 4"),
    )
    cfg = ExecutorConfig(
        max_rounds=1,
        max_attempts=1,
        system_prompt="请直接给我最终答案，不要任何分析过程",
    )
    sample = Sample(id="s1", input="q", target="#### 4")
    run_one_multi_turn(sample, cfg, tmp_path, run_id="r")

    prompt_arg = calls[0]["cmd"][2]
    assert prompt_arg.startswith("请直接给我最终答案，不要任何分析过程\n\n")
    assert prompt_arg.endswith("q")
