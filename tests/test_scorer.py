"""确定性 scorer 参考测试。"""

from __future__ import annotations

from deepeval_demo import (
    ScorerResult,
    ScorerSpec,
    reduce_epoch_scores,
    run_scorers,
)


def test_exact():
    spec = ScorerSpec(name="ex", type="exact", threshold=1.0)
    assert run_scorers([spec], "hello", "hello")[0].score == 1.0
    assert run_scorers([spec], "hello", "bye")[0].score == 0.0


def test_contains():
    spec = ScorerSpec(name="has", type="contains", threshold=1.0)
    result = run_scorers([spec], "答案是 42", "42")[0]
    assert result.score == 1.0
    assert result.passed is True


def test_regex_extract_from_target():
    spec = ScorerSpec(
        name="final_answer",
        type="regex",
        threshold=1.0,
        params={
            "pattern": r"####\s*([-+]?\d+(?:,\d+)*(?:\.\d+)?)",
            "extract_from_target": True,
        },
    )
    result = run_scorers(
        [spec], "所以答案是 #### 72", "逐步计算后 #### 72"
    )[0]
    assert result.score == 1.0
    assert result.passed is True


def test_numeric():
    spec = ScorerSpec(
        name="num", type="numeric", threshold=1.0, params={"tolerance": 0.01}
    )
    assert run_scorers([spec], "结果是 3.14 元", "3.14159")[0].score == 1.0


def test_reduce_epoch_scores_pass_at_k():
    spec = ScorerSpec(name="a", type="exact", threshold=1.0)
    epoch_scores = [
        [ScorerResult(name="a", score=0.0, passed=False)],
        [ScorerResult(name="a", score=1.0, passed=True)],
        [ScorerResult(name="a", score=0.0, passed=False)],
    ]
    stats = reduce_epoch_scores(epoch_scores, [spec], "pass_at_k", pass_k=2)
    assert stats["a"]["value"] == 1.0
    assert stats["a"]["pass_at_1"] is False
    assert stats["a"]["pass_at_k"] is True
