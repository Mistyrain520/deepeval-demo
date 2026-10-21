"""薄命令行入口。"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="deepeval-eval", description="LLM 评测框架")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="运行一次完整评测")
    run_parser.add_argument("-c", "--config", required=True, help="配置文件路径 (yaml/json)")

    args = parser.parse_args(argv)
    if args.command == "run":
        from .config import load_config
        from .orchestrate import run_eval

        run_eval(load_config(args.config))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
