"""可测试的 LLM 评测框架。

把数据、执行、Langfuse、裁判四个环节收敛成可导入的模块，供 pytest 直接调用。
"""

from __future__ import annotations

from .config import (
    DEFAULT_DIMENSIONS,
    DataSourceSpec,
    DimensionSpec,
    EvalConfig,
    ExecutorConfig,
    FieldSpec,
    JudgeConfig,
    LangfuseConfig,
    RunConfig,
    ScorerSpec,
    load_config,
)
from .data import Sample, download_hf, json_to_parquet, load_dataset
from .api import (
    Agent,
    CodexAgent,
    SampleResult,
    evaluate,
    hf_dataset,
    langfuse_dataset,
    local_dataset,
    make_run_id,
    publish,
    result_entries,
    run,
)
from .executor import (
    ExecResult,
    SampleRun,
    build_env,
    check_final_answer,
    run_batch,
    run_batch_epochs,
    run_one,
    run_one_multi_turn,
    safe_format,
)
from .judge import (
    JudgeInput,
    JudgeVerdict,
    ScoreResult,
    build_judge_input,
    default_dimensions,
    judge_batch,
)
from .langfuse import (
    LangfuseConfigError,
    LangfuseCredentials,
    ScoreRecord,
    TraceRecord,
    build_client,
    build_trace_index,
    fetch_traces,
    load_langfuse_config,
    resolve_trace_ids,
    upload_scorer_scores,
    upload_scores,
)
from .log import configure_logging, get_logger
from .orchestrate import RunReport, run_eval
from .report import assemble_report
from .scorer import ScorerResult, reduce_epoch_scores, register_scorer, run_scorers

__all__ = [
    "DEFAULT_DIMENSIONS",
    "DataSourceSpec",
    "DimensionSpec",
    "EvalConfig",
    "ExecutorConfig",
    "FieldSpec",
    "JudgeConfig",
    "LangfuseConfig",
    "RunConfig",
    "ScorerSpec",
    "load_config",
    "Sample",
    "download_hf",
    "json_to_parquet",
    "load_dataset",
    "Agent",
    "CodexAgent",
    "SampleResult",
    "evaluate",
    "hf_dataset",
    "langfuse_dataset",
    "local_dataset",
    "make_run_id",
    "publish",
    "result_entries",
    "run",
    "ExecResult",
    "SampleRun",
    "build_env",
    "check_final_answer",
    "run_batch",
    "run_batch_epochs",
    "run_one",
    "run_one_multi_turn",
    "safe_format",
    "JudgeInput",
    "JudgeVerdict",
    "ScoreResult",
    "build_judge_input",
    "default_dimensions",
    "judge_batch",
    "LangfuseConfigError",
    "LangfuseCredentials",
    "ScoreRecord",
    "TraceRecord",
    "build_client",
    "build_trace_index",
    "fetch_traces",
    "load_langfuse_config",
    "resolve_trace_ids",
    "upload_scorer_scores",
    "upload_scores",
    "configure_logging",
    "get_logger",
    "ScorerResult",
    "reduce_epoch_scores",
    "register_scorer",
    "run_scorers",
    "RunReport",
    "run_eval",
    "assemble_report",
]


def main() -> None:
    """兼容旧 ``deepeval-demo`` 入口，转发给 CLI。"""
    from .cli import main as cli_main

    cli_main()
