"""薄命令行入口。"""

from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="deepeval-eval", description="LLM 评测框架")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="运行一次完整评测")
    run_parser.add_argument("-c", "--config", required=True, help="配置文件路径 (yaml/json)")
    run_parser.add_argument(
        "--log-level",
        default=os.environ.get("DEEPEVAL_LOG_LEVEL", "INFO"),
        help="日志级别（默认 INFO，可用 DEBUG/INFO/WARNING/ERROR）",
    )
    run_parser.add_argument(
        "--log-file",
        default=os.environ.get("DEEPEVAL_LOG_FILE"),
        help="可选：JSON lines 日志落盘路径",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        from .config import load_config
        from .log import configure_logging
        from .orchestrate import run_eval

        configure_logging(level=args.log_level, log_file=args.log_file)
        run_eval(load_config(args.config))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
