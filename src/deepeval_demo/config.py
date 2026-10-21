"""配置模型：用 pydantic 描述整个评测流程，支持 yaml/json 加载。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class FieldSpec(BaseModel):
    """数据行到 Sample 的字段映射。"""

    id: str | None = None
    input: str = "input"
    target: str | None = None


class DataSourceSpec(BaseModel):
    """数据源描述：本地文件/HF 布局，或从 HuggingFace 下载。"""

    kind: Literal["local", "huggingface"] = "local"
    path: str | None = None
    repo_id: str | None = None
    config: str | None = None
    split: str | None = None
    limit: int | None = None
    field_spec: FieldSpec = Field(default_factory=FieldSpec)

    @model_validator(mode="after")
    def _check_source(self) -> "DataSourceSpec":
        if self.kind == "huggingface" and not (self.repo_id or self.path):
            raise ValueError("huggingface 数据源需要 repo_id（或 path）")
        if self.kind == "local" and not self.path:
            raise ValueError("local 数据源需要 path")
        return self


class ExecutorConfig(BaseModel):
    """被测 agent 的 codex exec 执行配置。"""

    model: str = "deepseek-v4-pro"
    prompt_template: str = "{input}"
    project_root: Path = Path(".")
    sandbox: str = "workspace-write"
    max_concurrent: int = 4
    max_attempts: int = 2
    retry_backoff_seconds: float = 1.0
    timeout_seconds: float | None = None
    ephemeral: bool = False
    codex_path: str | None = None
    epochs: int = 1
    epoch_reducer: Literal["pass_at_k", "average"] = "pass_at_k"
    pass_k: int | None = None
    seed: int | None = None
    max_rounds: int = 1
    system_prompt: str | None = None
    challenge_message: str = "不对，请重新算一下"
    final_answer_pattern: str = r"####\s*(.+)$"


class DimensionSpec(BaseModel):
    """单个评分维度的定义。"""

    name: str
    criteria: str
    steps: list[str] = Field(default_factory=list)
    weight: float = 1.0
    threshold: float = 0.5


DEFAULT_DIMENSIONS: list[DimensionSpec] = [
    DimensionSpec(
        name="relevance",
        criteria="评估回答是否紧扣问题，没有跑题、答非所问或无关内容。",
        steps=[
            "判断回答是否围绕问题展开。",
            "检查是否存在大量与问题无关的内容。",
        ],
    ),
    DimensionSpec(
        name="completeness",
        criteria="评估回答是否完整覆盖了问题，没有遗漏关键点。",
        steps=[
            "识别问题涉及的所有关键方面。",
            "检查回答是否遗漏了这些关键方面。",
        ],
    ),
    DimensionSpec(
        name="logical_coherence",
        criteria="评估回答的论述是否逻辑连贯、自洽，前后没有矛盾。",
        steps=[
            "检查段落和要点之间的逻辑衔接是否自然。",
            "检查是否存在前后矛盾或跳跃。",
        ],
    ),
    DimensionSpec(
        name="clarity",
        criteria="评估回答的表达是否清晰、结构合理、易于理解。",
        steps=[
            "检查语言是否清晰、无明显歧义。",
            "检查结构是否便于阅读和抓重点。",
        ],
    ),
    DimensionSpec(
        name="helpfulness",
        criteria="评估回答是否对提问者有实际帮助，直接解决了问题。",
        steps=[
            "判断回答是否给出了可执行或有用的信息。",
            "检查是否有实质内容，而非空话。",
        ],
    ),
]


class JudgeConfig(BaseModel):
    """裁判（LLM-as-judge）配置。"""

    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    evidence: str = "final_answer"
    scoring: Literal["single_call", "per_dimension"] = "single_call"
    dimensions: list[DimensionSpec] = Field(
        default_factory=lambda: [d.model_copy(deep=True) for d in DEFAULT_DIMENSIONS]
    )
    threshold: float = 0.5
    max_concurrent: int = 4
    throttle_seconds: float = 1.0
    max_retries: int = 3
    temperature: float = 0.0


class ScorerSpec(BaseModel):
    """确定性 scorer 的定义：type 对应注册表里的实现，params 传给实现。"""

    name: str
    type: str = "exact"
    params: dict[str, Any] = Field(default_factory=dict)
    threshold: float = 0.5


class LangfuseConfig(BaseModel):
    """Langfuse 相关开关。凭据从环境变量或 langfuse.json 读取。"""

    upload: bool = True
    environment_prefix: str = "eval"
    fetch_limit: int | None = None


class RunConfig(BaseModel):
    """评测运行相关配置。"""

    run_dir: Path = Path("runs")
    resume: bool = True
    run_id: str | None = None


class EvalConfig(BaseModel):
    """一份评测配置，驱动 data -> executor -> langfuse -> judge 全流程。"""

    name: str = "eval"
    dataset: DataSourceSpec
    agent: ExecutorConfig = Field(default_factory=ExecutorConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    scorers: list[ScorerSpec] = Field(default_factory=list)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    run: RunConfig = Field(default_factory=RunConfig)


def _parse_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "读取 yaml 配置需要 pyyaml，请先安装: uv add pyyaml"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是对象: {path}")
    return data


def load_config(path: str | Path) -> EvalConfig:
    """从 yaml/json 文件加载 EvalConfig。"""
    return EvalConfig.model_validate(_parse_file(Path(path)))
