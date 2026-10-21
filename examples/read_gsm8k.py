r"""读取本地 GSM8K 数据集的 test 分区，统计总行数并打印指定范围的数据。

用法（在项目根目录）:

    .venv\Scripts\python.exe examples\read_gsm8k.py
    .venv\Scripts\python.exe examples\read_gsm8k.py --split socratic --start 0 --count 3
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "mydataset" / "gsm8k"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="读取 GSM8K test 分区，统计总行数并打印指定范围的数据"
    )
    parser.add_argument(
        "--split",
        default="main",
        choices=["main", "socratic"],
        help="选择 main 或 socratic 版本（默认 main）",
    )
    parser.add_argument("--start", type=int, default=0, help="起始行号，从 0 开始（默认 0）")
    parser.add_argument("--count", type=int, default=5, help="打印的行数（默认 5）")
    args = parser.parse_args()

    parquet_path = DATASET_DIR / args.split / "test-00000-of-00001.parquet"
    if not parquet_path.exists():
        raise SystemExit(f"找不到文件: {parquet_path}")

    table = pq.read_table(parquet_path)
    total = table.num_rows

    print(f"文件: {parquet_path}")
    print(f"列名: {table.column_names}")
    print(f"总行数: {total}")

    if args.count <= 0:
        raise SystemExit("--count 必须为正整数")

    start = args.start
    if start >= total:
        print(f"起始行 {start} 超出范围（总行数 {total}），没有可打印的数据。")
        return

    end = min(total, start + args.count)
    subset = table.slice(start, end - start)
    records = subset.to_pydict()
    columns = table.column_names

    print(f"打印第 {start} ~ {end - 1} 行（共 {end - start} 行）:")
    for offset in range(end - start):
        row_no = start + offset
        print(f"\n===== 第 {row_no} 行 =====")
        for col in columns:
            print(f"[{col}]")
            print(records[col][offset])


if __name__ == "__main__":
    main()
