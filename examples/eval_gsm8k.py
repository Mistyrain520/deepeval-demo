"""结合 basic_eval 与 read_gsm8k：读取 GSM8K test 分区前 N 条并做评测。

用法（在项目根目录）:

    .venv\\Scripts\\python.exe examples\\eval_gsm8k.py
    .venv\\Scripts\\python.exe examples\\eval_gsm8k.py --count 20

流程：读取本地 parquet 数据集 -> 用 DeepSeek 生成答案 -> 用多个 GEval（LLM 裁判）
分别评估解题过程的完整性、计算正确性和逻辑连贯性，再对每条样本取平均分。
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---- 不走代理：必须在导入 deepeval / openai 之前设置 ----
# 移除已有的代理变量，并显式把所有地址加入 NO_PROXY。
for _proxy_key in list(os.environ):
    if _proxy_key.upper() in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}:
        os.environ.pop(_proxy_key, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# 关闭遥测上报（可选）。需在导入 deepeval 之前设置。
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import pyarrow.parquet as pq
from deepeval import evaluate
from deepeval.evaluate.configs import CacheConfig, DisplayConfig
from deepeval.metrics import GEval
from deepeval.models import DeepSeekModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "mydataset" / "gsm8k"

# 并发生成答案的线程数（手动控制，暂时写为 10）
GENERATE_WORKERS = 10


def load_samples(split: str, count: int) -> list[tuple[str, str]]:
    """读取 test 分区前 count 行，返回 [(question, answer), ...] 列表。"""
    parquet_path = DATASET_DIR / split / "test-00000-of-00001.parquet"
    if not parquet_path.exists():
        raise SystemExit(f"找不到文件: {parquet_path}")

    table = pq.read_table(parquet_path)
    records = table.slice(0, count).to_pydict()
    questions = records["question"]
    answers = records["answer"]
    return [(questions[i], answers[i]) for i in range(len(questions))]


def generate_answer(model: DeepSeekModel, question: str) -> str:
    """让模型解答题目，返回生成文本。"""
    output, _ = model.generate(question)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="读取 GSM8K test 分区前 N 条，用 DeepSeek 生成答案并做评测"
    )
    parser.add_argument(
        "--split",
        default="main",
        choices=["main", "socratic"],
        help="选择 main 或 socratic 版本（默认 main）",
    )
    parser.add_argument("--count", type=int, default=20, help="评测的样本数（默认 20）")
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count 必须为正整数")

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "未检测到 DEEPSEEK_API_KEY，请先设置环境变量，例如: "
            "$env:DEEPSEEK_API_KEY='sk-...'"
        )

    model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    model = DeepSeekModel(model=model_name)

    samples = load_samples(args.split, args.count)
    print(f"已读取 {len(samples)} 条测试样本（split={args.split}）")

    # 生成答案（并发）
    actual_outputs: list[str | None] = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=GENERATE_WORKERS) as pool:
        future_to_index = {
            pool.submit(generate_answer, model, question): index
            for index, (question, _) in enumerate(samples)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            actual_outputs[index] = future.result()
            print(f"生成进度: {sum(x is not None for x in actual_outputs)}/{len(samples)}")

    test_cases = [
        LLMTestCase(
            input=question,
            actual_output=actual_outputs[index],
            expected_output=answer,
        )
        for index, (question, answer) in enumerate(samples)
    ]

    evaluation_params = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ]

    # 方案一：把解题过程拆成多个维度，每个 GEval 只评一个维度，最后自己算平均
    process_metrics = [
        GEval(
            name="ReasoningCompleteness",
            criteria=(
                "评估 actual_output 是否针对 input 给出了完整、清晰的解题过程，"
                "没有跳步或省略关键推理。"
            ),
            evaluation_steps=[
                "检查是否覆盖了从题目条件到最终答案所需的全部主要步骤。",
                "检查是否存在关键推理被省略、合并或一句话带过。",
            ],
            evaluation_params=evaluation_params,
            model=model,
            threshold=0.5,
        ),
        GEval(
            name="CalculationCorrectness",
            criteria="评估 actual_output 解题过程中每一步的计算和数学推理是否正确。",
            evaluation_steps=[
                "逐步核验每一步的计算、单位换算和数学推导是否正确。",
                "检查是否存在计算错误、符号错误、数值错误或错误假设。",
            ],
            evaluation_params=evaluation_params,
            model=model,
            threshold=0.5,
        ),
        GEval(
            name="LogicalCoherence",
            criteria=(
                "评估 actual_output 的解题过程是否逻辑连贯、自洽，并能自然导向最终答案。"
            ),
            evaluation_steps=[
                "检查步骤之间的逻辑衔接是否自然，是否存在前后矛盾。",
                "确认推理过程是否导向 expected_output 中 '####' 之后的最终答案（数学上等价即可）。",
            ],
            evaluation_params=evaluation_params,
            model=model,
            threshold=0.5,
        ),
    ]

    results = evaluate(
        test_cases,
        process_metrics,
        hyperparameters={"model": model_name, "prompt_template": "GSM8K"},
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=True, print_results=False),
    )

    case_averages: list[float] = []
    for test_result in results.test_results:
        metric_scores = [
            metric_data.score for metric_data in test_result.metrics_data
        ]
        case_average = sum(metric_scores) / len(metric_scores)
        case_passed = case_average >= 0.5
        case_averages.append(case_average)

        detail = ", ".join(
            f"{metric_data.name}={metric_data.score:.2f}"
            for metric_data in test_result.metrics_data
        )
        print(
            f"[{test_result.index}] 平均分={case_average:.2f} "
            f"passed={case_passed} | {detail}"
        )

    passed = sum(1 for case_average in case_averages if case_average >= 0.5)
    average = sum(case_averages) / len(case_averages) if case_averages else 0.0
    print(
        f"评测完成: 共 {len(case_averages)} 条 | 通过 {passed} | "
        f"失败 {len(case_averages) - passed} | 平均分 {average:.2f}"
    )


if __name__ == "__main__":
    main()
