from deepeval_demo import (
    local_dataset, hf_dataset, langfuse_dataset,  # 1 拉数据
    CodexAgent,                                     # 2 跑 agent（Agent 协议，可扩展）
    evaluate,                                       # 3 scorer + judge 评分
    publish,                                        # 4 上传 Langfuse
    run,                                            # 一步到位
    make_run_id,                                    # 自定义前缀 + 分钟后缀
)

def test_gsm8k():
    samples = local_dataset("mydataset/gsm8k/main", split="test", limit=5,
                            field_spec={"input": "question", "target": "answer"})
    agent = CodexAgent(
        model="deepseek-v4-flash",
        max_concurrent=2,
        max_rounds=5,
        system_prompt="请直接给我最终答案，不要任何分析过程",
    )
    run_id = make_run_id("codex-gsm8k")            # => codex-gsm8k-20261020-1603
    runs = agent.run(samples, run_id=run_id)
    results = evaluate(runs, model="deepseek-chat")
    publish(results, run_id=run_id)

def test_score_recent_traces():
    report = run(langfuse_dataset(limit=20), name="score-recent")  # 已答样本直接评并回传原 trace
    assert report.pass_rate > 0.5


test_gsm8k()
