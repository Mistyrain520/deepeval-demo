"""裁判模块参考测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace

from deepeval_demo import (
    DimensionSpec,
    JudgeConfig,
    JudgeInput,
    build_judge_input,
    default_dimensions,
    judge_batch,
)


def test_default_dimensions():
    dims = default_dimensions()
    assert [d.name for d in dims] == [
        "relevance",
        "completeness",
        "logical_coherence",
        "clarity",
        "helpfulness",
    ]


def test_build_judge_input_evidence():
    cfg = JudgeConfig(evidence="final_answer")
    inp = build_judge_input("s1", "问题", "回答", cfg=cfg)
    assert "问题" in inp.evidence
    assert "回答" in inp.evidence
    assert "【执行轨迹】" not in inp.evidence


def test_validate_scores_weighting():
    from deepeval_demo.judge.engine import _validate_scores

    dims = [
        DimensionSpec(name="a", criteria="x", weight=1.0),
        DimensionSpec(name="b", criteria="x", weight=3.0),
    ]
    raw = {
        "scores": [
            {"dimension": "a", "score": 0.5, "reason": "r"},
            {"dimension": "b", "score": 1.0, "reason": "r"},
        ]
    }
    verdict = _validate_scores(raw, dims, JudgeConfig(threshold=0.8))
    assert verdict.overall == 0.875
    assert verdict.passed is True


def test_judge_batch_mocked(monkeypatch):
    import deepeval_demo.judge.engine as engine

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    content = json.dumps(
        {
            "scores": [
                {"dimension": "relevance", "score": 1.0, "reason": "ok"},
                {"dimension": "completeness", "score": 0.9, "reason": "ok"},
                {"dimension": "logical_coherence", "score": 0.8, "reason": "ok"},
                {"dimension": "clarity", "score": 0.7, "reason": "ok"},
                {"dimension": "helpfulness", "score": 0.6, "reason": "ok"},
            ]
        }
    )

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(engine, "_make_client", lambda cfg: FakeClient())
    monkeypatch.setattr(engine, "_throttle", lambda interval: None)

    inputs = [JudgeInput(sample_id="s1", input="q", output="a", evidence="q\na")]
    verdicts = judge_batch(
        inputs, JudgeConfig(max_concurrent=1, throttle_seconds=0)
    )
    assert len(verdicts) == 1
    assert verdicts[0].overall == 0.8
    assert verdicts[0].passed is True
