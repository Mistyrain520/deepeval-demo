"""清理 Langfuse 项目里由评测流水线写入的 deepeval_* 分数。

主要用于删除因 observations v2 接口重复返回而造成的重复分数。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe examples\\cleanup_langfuse_scores.py --dry-run
    .venv\\Scripts\\python.exe examples\\cleanup_langfuse_scores.py
    .venv\\Scripts\\python.exe examples\\cleanup_langfuse_scores.py --run-id codex-eval-20261018-1805

注意：Langfuse 的分数删除接口有每日限流（约 50 次/天）。脚本会尽量删除，
被限流时会打印剩余数量并退出；可以第二天再跑一次继续清理。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langfuse import Langfuse


def load_config() -> dict[str, str]:
    config_file = Path.home() / ".codex" / "langfuse.json"
    if not config_file.exists():
        sys.exit("未找到 ~/.codex/langfuse.json")
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    return {
        "public_key": raw.get("public_key", ""),
        "secret_key": raw.get("secret_key", ""),
        "base_url": raw.get("base_url", ""),
    }


def list_target_scores(client: Langfuse, run_id: str | None) -> list:
    targets = []
    cursor = None
    while True:
        resp = client.api.scores.get_many(limit=50, cursor=cursor, fields="details")
        for s in resp.data:
            if not str(s.name).startswith("deepeval_"):
                continue
            if run_id and (s.metadata or {}).get("evaluation_run_id") != run_id:
                continue
            targets.append(s)
        cursor = resp.meta.cursor if resp.meta else None
        if not cursor or not resp.data:
            break
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="清理 Langfuse 里的 deepeval_* 分数")
    parser.add_argument("--run-id", help="只清理指定 evaluation_run_id 的分数")
    parser.add_argument("--dry-run", action="store_true", help="只统计不删除")
    args = parser.parse_args()

    config = load_config()
    client = Langfuse(
        public_key=config["public_key"],
        secret_key=config["secret_key"],
        base_url=config["base_url"],
    )

    targets = list_target_scores(client, args.run_id)
    print(f"待清理分数：{len(targets)} 条")

    if args.dry_run:
        from collections import Counter

        runs = Counter((s.metadata or {}).get("evaluation_run_id", "无 run_id") for s in targets)
        for rid, n in runs.items():
            print(f"  run_id={rid}: {n} 条")
        return

    deleted = 0
    for s in targets:
        try:
            client.api.scores.delete(score_id=s.id)
            deleted += 1
        except Exception as exc:
            print(f"删除中断（{exc}），已删除 {deleted} / {len(targets)} 条")
            remaining = len(targets) - deleted
            print(f"剩余 {remaining} 条，可稍后或次日再跑本脚本继续清理。")
            return
    print(f"已删除 {deleted} 条")


if __name__ == "__main__":
    main()
