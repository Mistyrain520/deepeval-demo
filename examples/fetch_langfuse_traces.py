"""从 Langfuse 拉取 trace 记录的命令行工具。

默认从环境变量读取凭据，其次回退到 `~/.codex/langfuse.json`（即 Codex
observability 插件的配置文件），因此开箱即可用，无需再单独配置。

环境变量优先级：
    LANGFUSE_PUBLIC_KEY  / LANGFUSE_SECRET_KEY  / LANGFUSE_BASE_URL

用法示例：
    uv run python examples/fetch_langfuse_traces.py --limit 20
    uv run python examples/fetch_langfuse_traces.py --name "check" --limit 10
    uv run python examples/fetch_langfuse_traces.py --trace-id <trace_id> --json

指定 --trace-id 时，除打印外还会把结果写入 `<trace_id>.json`（可用 --out-dir
指定目录，默认写入当前目录）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from langfuse import Langfuse


# observations v2 默认只返回 core 字段，这里显式带上 io/metadata/model/usage，
# 才能拿到输入输出、模型名、token 用量等内容。
FIELDS = "core,basic,time,io,metadata,model,usage,trace_context"


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    """把命令行传入的 ISO 时间字符串解析为带时区信息的 datetime。"""
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def load_config() -> dict[str, str]:
    """按 环境变量 -> ~/.codex/langfuse.json 的顺序加载配置。"""
    config: dict[str, str] = {}

    # 1) 环境变量
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        config["public_key"] = os.environ["LANGFUSE_PUBLIC_KEY"]
    if os.getenv("LANGFUSE_SECRET_KEY"):
        config["secret_key"] = os.environ["LANGFUSE_SECRET_KEY"]
    if os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST"):
        config["base_url"] = os.environ.get(
            "LANGFUSE_BASE_URL"
        ) or os.environ.get("LANGFUSE_HOST", "")

    # 2) 回退到 Codex 插件的配置文件
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
            + "。请设置环境变量 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / "
            "LANGFUSE_BASE_URL，或确保存在 ~/.codex/langfuse.json。"
        )
    return config


def build_client() -> Langfuse:
    config = load_config()
    return Langfuse(
        public_key=config["public_key"],
        secret_key=config["secret_key"],
        base_url=config["base_url"],
    )


def _preview(value: Any, max_len: int = 120) -> str:
    """把 input/output 压成一行可读的摘要文本。"""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


def _fmt_cost(cost: Optional[float]) -> str:
    if cost is None:
        return "-"
    return f"${cost:.6f}"


def _fmt_latency(latency: Optional[float]) -> str:
    if latency is None:
        return "-"
    return f"{latency:.2f}s"


def serialize_observations(observations: list[Any]) -> list[dict[str, Any]]:
    """把 observation 对象列表转成可 JSON 序列化的字典列表。"""
    return [obs.model_dump(mode="json") for obs in observations]


def write_trace_to_json(
    trace_id: str,
    observations: list[Any],
    out_dir: Optional[str],
) -> Path:
    """把单条 trace 的 observation 写入 `<trace_id>.json`，返回文件路径。"""
    directory = Path(out_dir) if out_dir else Path.cwd()
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / f"{trace_id}.json"
    out_path.write_text(
        json.dumps(serialize_observations(observations), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return out_path


def print_trace_summary(observations: list[Any]) -> None:
    """把 root observation（即 trace）打印成一行行的摘要。"""
    if not observations:
        print("没有找到任何 trace。")
        return

    print(f"共 {len(observations)} 条 trace：")
    for obs in observations:
        name = obs.trace_name or obs.name or "-"
        started = obs.start_time.strftime("%Y-%m-%d %H:%M:%S") if obs.start_time else "-"
        line = (
            f"- {obs.trace_id} | {name} | {started} | "
            f"latency={_fmt_latency(obs.latency)} | cost={_fmt_cost(obs.total_cost)}"
        )
        if obs.model:
            line += f" | model={obs.model}"
        print(line)
        if _preview(obs.input):
            print(f"    input : {_preview(obs.input)}")
        if _preview(obs.output):
            print(f"    output: {_preview(obs.output)}")


def fetch_traces(
    client: Langfuse,
    *,
    limit: int,
    name: Optional[str],
    user_id: Optional[str],
    session_id: Optional[str],
    environment: Optional[str],
    from_time: Optional[datetime],
    to_time: Optional[datetime],
) -> list[Any]:
    """分页拉取 root observation，等价于 trace 列表。"""
    collected: list[Any] = []
    cursor: Optional[str] = None

    while len(collected) < limit:
        page_size = min(50, limit - len(collected))
        response = client.api.observations.get_many(
            fields=FIELDS,
            limit=page_size,
            cursor=cursor,
            is_root_observation=True,
            name=name,
            user_id=user_id,
            session_id=session_id,
            environment=environment,
            from_start_time=from_time,
            to_start_time=to_time,
        )
        collected.extend(response.data)
        cursor = response.meta.cursor if response.meta else None
        if not cursor or not response.data:
            break

    return collected


def fetch_single_trace(client: Langfuse, trace_id: str) -> list[Any]:
    """拉取某条 trace 下的全部 observation。"""
    collected: list[Any] = []
    cursor: Optional[str] = None

    while True:
        response = client.api.observations.get_many(
            fields=FIELDS,
            trace_id=trace_id,
            limit=50,
            cursor=cursor,
        )
        collected.extend(response.data)
        cursor = response.meta.cursor if response.meta else None
        if not cursor or not response.data:
            break

    collected.sort(key=lambda obs: obs.start_time or datetime.min.replace(tzinfo=timezone.utc))
    return collected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 Langfuse SDK 拉取 trace 记录。"
    )
    parser.add_argument("--limit", type=int, default=50, help="最多返回的 trace 数量（默认 50）")
    parser.add_argument("--trace-id", help="按 trace_id 拉取单条 trace 的全部 observation")
    parser.add_argument("--name", help="按 trace 名称过滤")
    parser.add_argument("--user-id", help="按 user_id 过滤")
    parser.add_argument("--session-id", help="按 session_id 过滤")
    parser.add_argument("--environment", help="按 environment 过滤")
    parser.add_argument("--from", dest="from_time", help="起始时间（ISO 8601），如 2026-10-18T00:00:00Z")
    parser.add_argument("--to", dest="to_time", help="结束时间（ISO 8601）")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON 而非摘要")
    parser.add_argument("--out-dir", help="写入 JSON 文件的目录（默认当前目录）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = build_client()

    if args.trace_id:
        observations = fetch_single_trace(client, args.trace_id)
        out_path = write_trace_to_json(args.trace_id, observations, args.out_dir)
        # 文件提示走 stderr，避免 --json 时污染 stdout 的 JSON 内容
        print(
            f"已写入 {out_path}（{len(observations)} 条 observation）",
            file=sys.stderr,
        )
        if args.json:
            print(json.dumps(serialize_observations(observations), ensure_ascii=False, indent=2))
        else:
            print(f"trace {args.trace_id} 共 {len(observations)} 条 observation：")
            for obs in observations:
                print(
                    f"- {obs.type} | {obs.name} | "
                    f"latency={_fmt_latency(obs.latency)} | cost={_fmt_cost(obs.total_cost)}"
                )
                if _preview(obs.input):
                    print(f"    input : {_preview(obs.input)}")
                if _preview(obs.output):
                    print(f"    output: {_preview(obs.output)}")
        return

    observations = fetch_traces(
        client,
        limit=args.limit,
        name=args.name,
        user_id=args.user_id,
        session_id=args.session_id,
        environment=args.environment,
        from_time=_parse_iso(args.from_time),
        to_time=_parse_iso(args.to_time),
    )

    if args.json:
        print(json.dumps(serialize_observations(observations), ensure_ascii=False, indent=2))
    else:
        print_trace_summary(observations)


if __name__ == "__main__":
    main()
