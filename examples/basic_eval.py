"""一个基础的 DeepEval 示例。

运行方式（在项目根目录）:

    uv run python examples/basic_eval.py

本示例包含两部分：

1. `run_offline_example` —— 用 ExactMatchMetric 做纯字符串比对，
   不需要调用大模型，可离线直接运行，用来熟悉 evaluate() 的流程。
2. `run_llm_judge_example` —— 用 GEval（正确性）和
   AnswerRelevancyMetric（答案相关性）演示“大模型当裁判”。
   优先使用 DeepSeek（设置 DEEPSEEK_API_KEY），
   其次回退到 OpenAI（设置 OPENAI_API_KEY）。
"""

import os

# 关闭遥测上报（可选）。需在导入 deepeval 之前设置。
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval import assert_test, evaluate
from deepeval.metrics import AnswerRelevancyMetric, ExactMatchMetric, GEval
from deepeval.models import DeepSeekModel
from deepeval.test_case import LLMTestCase, SingleTurnParams


def run_offline_example() -> None:
    """不需要 API Key 的最小示例，演示 evaluate() 的基本流程。"""
    test_case = LLMTestCase(
        input="1 + 1 等于几？",
        actual_output="2",
        expected_output="2",
    )

    # threshold=1.0 表示必须完全一致才算通过
    metric = ExactMatchMetric(threshold=1.0)

    results = evaluate([test_case], [metric])

    # evaluate() 返回 EvaluationResult，按 test_case 逐个查看结果
    score = results.test_results[0].metrics_data[0].score
    success = results.test_results[0].success
    print(f"[离线示例] ExactMatch 得分: {score}, 是否通过: {success}")


def run_llm_judge_example() -> None:
    """LLM-as-judge 示例，优先使用 DeepSeek，其次 OpenAI。"""
    use_deepseek = bool(os.getenv("DEEPSEEK_API_KEY"))
    use_openai = bool(os.getenv("OPENAI_API_KEY"))

    if not use_deepseek and not use_openai:
        print(
            "[LLM 示例] 未检测到 DEEPSEEK_API_KEY 或 OPENAI_API_KEY，已跳过。"
            "请设置环境变量后重试，例如: $env:DEEPSEEK_API_KEY='sk-...'"
        )
        return

    # DeepSeek 与 OpenAI 不共用 key 和地址，这里根据环境变量选择 provider
    if use_deepseek:
        model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
        model = DeepSeekModel(model=model_name)
        provider = f"DeepSeek ({model_name})"
    else:
        model = None  # 为 None 时 DeepEval 使用默认 OpenAI 配置
        provider = "OpenAI"

    test_case = LLMTestCase(
        input="如果这双鞋不合脚怎么办？",
        actual_output="我们提供 30 天无理由全额退款，无需额外费用。",
        expected_output="我们提供 30 天无理由全额退款。",
        retrieval_context=["所有客户都有资格在 30 天内获得全额退款，无需额外费用。"],
    )

    # 正确性：依据输入、实际输出和期望输出做事实判断
    correctness = GEval(
        name="Correctness",
        criteria="判断 actual_output 是否基于 expected_output 在事实层面正确。",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=0.5,
    )

    # 答案相关性：回答是否紧扣输入问题
    relevancy = AnswerRelevancyMetric(model=model, threshold=0.5)

    # assert_test 会在指标不达标时抛出异常，适合放在 pytest 里使用
    assert_test(test_case, [correctness, relevancy])
    print(f"[LLM 示例] 使用 {provider}，断言通过：correctness 与 answer relevancy 均达到阈值。")


if __name__ == "__main__":
    run_offline_example()
    run_llm_judge_example()
