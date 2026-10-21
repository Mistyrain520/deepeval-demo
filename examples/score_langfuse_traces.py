"""完整评测流水线：Langfuse 取 trace -> DeepEval 打分 -> 分数回传 Langfuse。

流程：
    1. 从 Langfuse 拉取最近 N 条 trace（root observation）；
    2. 对每条 trace 拉取其下全部 observation（完整观测日志）；
    3. 用 DeepEval 的多个 GEval 指标（LLM 裁判，DeepSeek）打分；
    4. 把每条 trace 的各项分数通过 create_score 上报回 Langfuse；
    5. 把打分结果与上报信息写入本地 JSON 报告。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe examples\\score_langfuse_traces.py --limit 20
    .venv\\Scripts\\python.exe examples\\score_langfuse_traces.py --limit 20 --no-upload
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---- 必须在导入 deepeval / openai 之前设置 ----
# 评测走 DeepSeek 直连，不经过代理；同时关闭 DeepEval 遥测。
for _proxy_key in list(os.environ):
    if _proxy_key.upper() in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}:
        os.environ.pop(_proxy_key, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from langfuse import Langfuse  # noqa: E402
from deepeval import evaluate  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig  # noqa: E402
from deepeval.metrics import GEval  # noqa: E402
from deepeval.models import DeepSeekModel  # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402


# observations v2 默认只返回 core 字段，显式带上 io/metadata/model/usage。
FIELDS = "core,basic,time,io,metadata,model,usage,trace_context"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_run_id(explicit: Optional[str]) -> str:
    """返回本轮评测的 run id；未显式指定时按本地时间精确到分钟。

    格式为 codex-eval-YYYYMMDD-HHMM，例如 codex-eval-20261018-1804 表示
    18 点 04 分。同一分钟内重复运行会得到相同 id，需要区分时可手动 --run-id。"""
    if explicit:
        return explicit
    return "codex-eval-" + datetime.now().strftime("%Y%m%d-%H%M")


def _to_text(value: Any) -> str:
    """把 observation 的 input/output 归一化成纯文本。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            return value["content"]
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _preview(value: Any, max_len: int = 500) -> str:
    text = " ".join(_to_text(value).split())
    return text if len(text) <= max_len else text[:max_len] + "…"


def load_config() -> dict[str, str]:
    """按 环境变量 -> ~/.codex/langfuse.json 的顺序加载 Langfuse 配置。"""
    config: dict[str, str] = {}
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        config["public_key"] = os.environ["LANGFUSE_PUBLIC_KEY"]
    if os.getenv("LANGFUSE_SECRET_KEY"):
        config["secret_key"] = os.environ["LANGFUSE_SECRET_KEY"]
    if os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST"):
        config["base_url"] = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get(
            "LANGFUSE_HOST", ""
        )
    if not all(k in config for k in ("public_key", "secret_key", "base_url")):
        config_file = Path.home() / ".codex" / "langfuse.json"
        if config_file.exists():
            raw = json.loads(config_file.read_text(encoding="utf-8"))
            config.setdefault("public_key", raw.get("public_key") or "")
            config.setdefault("secret_key", raw.get("secret_key") or "")
            config.setdefault("base_url", raw.get("base_url") or "")
    missing = [
        key
        for key in ("public_key", "secret_key", "base_url")
        if not config.get(key)
    ]
    if missing:
        sys.exit(
            "缺少 Langfuse 配置："
            + ", ".join(missing)
            + "。请设置 LANGFUSE_* 环境变量或 ~/.codex/langfuse.json。"
        )
    return config


def build_client() -> Langfuse:
    config = load_config()
    return Langfuse(
        public_key=config["public_key"],
        secret_key=config["secret_key"],
        base_url=config["base_url"],
    )


def fetch_recent_traces(client: Langfuse, limit: int) -> list[Any]:
    """拉取最近 limit 条 trace（即 root observation）。"""
    collected: list[Any] = []
    seen: set[str] = set()
    cursor: Optional[str] = None
    while len(collected) < limit:
        page_size = min(50, limit - len(collected))
        response = client.api.observations.get_many(
            fields=FIELDS,
            limit=page_size,
            cursor=cursor,
            is_root_observation=True,
        )
        # observations v2 接口偶发会把同一条记录重复返回，这里按 trace_id 去重。
        for obs in response.data:
            if obs.trace_id not in seen:
                seen.add(obs.trace_id)
                collected.append(obs)
        cursor = response.meta.cursor if response.meta else None
        if not cursor or not response.data:
            break
    return collected


def fetch_trace_observations(client: Langfuse, trace_id: str) -> list[Any]:
    """拉取某条 trace 下的全部 observation（完整观测日志）。"""
    collected: list[Any] = []
    seen: set[str] = set()
    cursor: Optional[str] = None
    while True:
        response = client.api.observations.get_many(
            fields=FIELDS,
            trace_id=trace_id,
            limit=50,
            cursor=cursor,
        )
        # 同样按 observation id 去重，防止接口重复返回导致日志重复。
        for obs in response.data:
            if obs.id not in seen:
                seen.add(obs.id)
                collected.append(obs)
        cursor = response.meta.cursor if response.meta else None
        if not cursor or not response.data:
            break
    collected.sort(
        key=lambda o: o.start_time or datetime.min.replace(tzinfo=timezone.utc)
    )
    return collected


def find_root(observations: list[Any]) -> Optional[Any]:
    for obs in observations:
        if obs.is_root_observation:
            return obs
    return None


def build_metrics(model: DeepSeekModel, threshold: float) -> list[GEval]:
    """构造 5 个参考无关的 GEval 评分指标。"""
    params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
    # 关键：GEval 在给了显式 evaluation_steps 时，final prompt 里只包含
    # evaluation_steps（不包含 criteria），所以中文约束必须写进 steps 才会生效。
    lang_step = "使用中文撰写评分理由（reason 字段），不要输出英文。"
    return [
        GEval(
            name="Relevance",
            criteria="评估 actual_output 是否紧扣 input 的问题，没有跑题、答非所问或无关内容。",
            evaluation_steps=[
                "判断回答是否围绕用户的问题展开。",
                "检查是否存在大量与问题无关的内容。",
                lang_step,
            ],
            evaluation_params=params,
            model=model,
            threshold=threshold,
        ),
        GEval(
            name="Completeness",
            criteria="评估 actual_output 是否完整覆盖了 input 的问题，没有遗漏关键点。",
            evaluation_steps=[
                "识别问题涉及的所有关键方面。",
                "检查回答是否遗漏了这些关键方面。",
                lang_step,
            ],
            evaluation_params=params,
            model=model,
            threshold=threshold,
        ),
        GEval(
            name="LogicalCoherence",
            criteria="评估 actual_output 的论述是否逻辑连贯、自洽，前后没有矛盾。",
            evaluation_steps=[
                "检查段落和要点之间的逻辑衔接是否自然。",
                "检查是否存在前后矛盾或跳跃。",
                lang_step,
            ],
            evaluation_params=params,
            model=model,
            threshold=threshold,
        ),
        GEval(
            name="Clarity",
            criteria="评估 actual_output 的表达是否清晰、结构合理、易于理解。",
            evaluation_steps=[
                "检查语言是否清晰、无明显歧义。",
                "检查结构是否便于阅读和抓重点。",
                lang_step,
            ],
            evaluation_params=params,
            model=model,
            threshold=threshold,
        ),
        GEval(
            name="Helpfulness",
            criteria="评估 actual_output 是否对提问者有实际帮助，直接解决了问题。",
            evaluation_steps=[
                "判断回答是否给出了可执行或有用的信息。",
                "检查是否有实质内容，而非空话。",
                lang_step,
            ],
            evaluation_params=params,
            model=model,
            threshold=threshold,
        ),
    ]


def summarize_observations(observations: list[Any]) -> list[dict[str, Any]]:
    summary = []
    for obs in observations:
        summary.append(
            {
                "type": obs.type,
                "name": obs.name,
                "start_time": obs.start_time.isoformat() if obs.start_time else None,
                "latency": obs.latency,
                "model": obs.model,
                "input_preview": _preview(obs.input),
                "output_preview": _preview(obs.output),
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Langfuse trace 拉取 + DeepEval 打分 + 回传")
    parser.add_argument("--limit", type=int, default=20, help="评测最近的 trace 数量（默认 20）")
    parser.add_argument("--run-id", help="本轮评测的 evaluation_run_id（默认精确到分钟，如 codex-eval-20261018-1804）")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
                        help="DeepEval 裁判模型（默认 deepseek-chat）")
    parser.add_argument("--threshold", type=float, default=0.5, help="及格阈值（默认 0.5）")
    parser.add_argument("--max-concurrent", type=int, default=2, help="并发评测数（默认 2）")
    parser.add_argument("--throttle", type=float, default=1.0, help="请求间隔秒数（默认 1.0）")
    parser.add_argument("--out-dir", help="报告输出目录（默认当前目录）")
    parser.add_argument("--no-upload", action="store_true", help="只打分不上报（dry-run）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.getenv("DEEPSEEK_API_KEY"):
        sys.exit("未检测到 DEEPSEEK_API_KEY，请先设置环境变量。")

    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluation_run_id = resolve_run_id(args.run_id)

    client = build_client()
    traces = fetch_recent_traces(client, args.limit)
    print(
        f"已拉取 {len(traces)} 条 trace，evaluation_run_id={evaluation_run_id}，"
        f"并发={args.max_concurrent}，间隔={args.throttle}s，开始评测…"
    )

    eval_model = DeepSeekModel(model=args.model)
    metrics = build_metrics(eval_model, args.threshold)

    # 第一遍：拉全量观测、构造条目与 test_cases，记录 test_case -> entry 的映射。
    entries: list[dict[str, Any]] = []
    test_cases: list[LLMTestCase] = []
    case_entry_map: list[int] = []

    for idx, trace in enumerate(traces, 1):
        trace_id = trace.trace_id
        observations = fetch_trace_observations(client, trace_id)
        root = find_root(observations)
        question = _to_text(root.input) if root else ""
        answer = _to_text(root.output) if root else ""

        entry_index = len(entries)
        entry: dict[str, Any] = {
            "trace_id": trace_id,
            "name": trace.trace_name or trace.name,
            "start_time": trace.start_time.isoformat() if trace.start_time else None,
            "input": question,
            "answer_preview": _preview(answer),
            "observation_count": len(observations),
            "observations": summarize_observations(observations),
            "scores": [],
        }
        entries.append(entry)

        if not question or not answer:
            entry["skipped"] = "缺少 input 或 output，跳过评分"
            print(f"[{idx}/{len(traces)}] {trace_id} 跳过（缺少 input/output）")
            continue

        test_cases.append(LLMTestCase(input=question, actual_output=answer))
        case_entry_map.append(entry_index)

    # 第二遍：用 evaluate 批量并发评测（并发与间隔由命令行参数控制）。
    if test_cases:
        evaluation_result = evaluate(
            test_cases,
            metrics,
            hyperparameters={
                "model": args.model,
                "evaluation_run_id": evaluation_run_id,
            },
            async_config=AsyncConfig(
                run_async=True,
                max_concurrent=args.max_concurrent,
                throttle_value=args.throttle,
            ),
            display_config=DisplayConfig(show_indicator=True, print_results=False),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
        )

        for pos, test_result in enumerate(evaluation_result.test_results):
            entry_index = case_entry_map[pos]
            entry = entries[entry_index]
            trace_id = entry["trace_id"]
            score_values: list[float] = []

            for metric_data in test_result.metrics_data:
                if metric_data.score is None:
                    value = 0.0
                    reason = f"评测失败: {metric_data.error or '无分数'}"
                else:
                    value = float(metric_data.score)
                    reason = metric_data.reason or ""
                # DeepEval 会给 GEval 名追加 " [GEval]" 后缀，这里去掉。
                base_name = metric_data.name.split(" [")[0]
                passed = value >= args.threshold
                score_values.append(value)
                score_entry: dict[str, Any] = {
                    "name": f"deepeval_{base_name.lower()}",
                    "value": round(value, 4),
                    "passed": passed,
                    "reason": _preview(reason, 800),
                }
                if not args.no_upload:
                    try:
                        client.create_score(
                            name=score_entry["name"],
                            value=float(value),
                            trace_id=trace_id,
                            data_type="NUMERIC",
                            comment=reason[:2000],
                            metadata={"evaluation_run_id": evaluation_run_id},
                        )
                        score_entry["uploaded"] = True
                        score_entry["uploaded_at"] = now_iso()
                    except Exception as exc:
                        score_entry["uploaded"] = False
                        score_entry["upload_error"] = str(exc)
                entry["scores"].append(score_entry)

            entry["average"] = (
                round(sum(score_values) / len(score_values), 4)
                if score_values
                else None
            )
            print(
                f"[{entry_index + 1}/{len(traces)}] {trace_id} "
                f"平均分={entry['average']} ({entry['observation_count']} obs)"
            )

    report = {
        "generated_at": now_iso(),
        "evaluation_run_id": evaluation_run_id,
        "langfuse_host": load_config()["base_url"],
        "eval_model": args.model,
        "threshold": args.threshold,
        "max_concurrent": args.max_concurrent,
        "throttle": args.throttle,
        "uploaded": not args.no_upload,
        "total_traces": len(entries),
        "results": entries,
    }

    report_path = out_dir / f"score_report_{evaluation_run_id}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    avg_all = [r["average"] for r in entries if r.get("average") is not None]
    if avg_all:
        print(
            f"\n评测完成：共 {len(entries)} 条 trace | "
            f"平均分 {sum(avg_all) / len(avg_all):.2f}"
        )
    else:
        print("\n评测完成")
    print(f"evaluation_run_id：{evaluation_run_id}")
    print(f"报告已写入：{report_path}")


if __name__ == "__main__":
    main()
