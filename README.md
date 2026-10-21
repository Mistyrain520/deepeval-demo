# deepeval-demo

一个可测试的 LLM 评测框架，把评测拆成四个可单独 import 的模块，再串成一条流水线：

```text
data（数据） -> executor（执行） -> langfuse（取 trace / 回传分数） -> scorer（硬分数）+ judge（裁判）
```

## 函数式四步 API（推荐，pytest 直呼）

不用写 yaml，四步自由函数直接调用：

```python
from deepeval_demo import local_dataset, CodexAgent, evaluate, publish

def test_gsm8k():
    samples = local_dataset("mydataset/gsm8k/main", split="test", limit=5,
                            field_spec={"input": "question", "target": "answer"})
    agent = CodexAgent(model="deepseek-v4-pro")
    runs = agent.run(samples, run_id="gsm8k-demo")
    results = evaluate(runs, model="deepseek-chat")
    publish(results, run_id="gsm8k-demo")
```

四种数据来源：`local_dataset`（本地）、`hf_dataset`（HuggingFace 下载）、`langfuse_dataset`
（拉 trace 当“已答样本”，跳过 agent 直接评分并回传原 trace）。一步到位用 `run(...)`：

```python
from deepeval_demo import run, langfuse_dataset

report = run(langfuse_dataset(limit=20), run_id="score-recent")
assert report.pass_rate > 0.5
```

`Agent` 是协议，`CodexAgent` 是首个实现，未来可扩展其他 agent；`evaluate` 同时做确定性 scorer
和 LLM 裁判，`publish` 统一回传 Langfuse。

## yaml 批跑

```powershell
uv sync --extra deepeval --extra dev
$env:DEEPSEEK_API_KEY = "sk-..."

# 一条命令跑完整评测（读取 configs/example.yaml）
.\.venv\Scripts\deepeval-eval.exe run -c configs/example.yaml
```

`example.yaml` 只是函数式 API 的配置适配层，适合大规模批跑；执行层每个样本起一个 `codex exec`
子进程，并通过 `LANGFUSE_CODEX_ENVIRONMENT/TAGS/METADATA` 与 Langfuse 官方插件自动产生的
trace 精确关联。结果落在 `runs/<run_id>/`（含 `state.jsonl`、`answers/`、`report.json`）。

## 底层模块 API（进阶）

```python
from deepeval_demo import (
    DataSourceSpec, FieldSpec, load_dataset, json_to_parquet, download_hf,
    ExecutorConfig, run_one, run_batch, run_batch_epochs,
    ScorerSpec, run_scorers, reduce_epoch_scores, register_scorer,
    JudgeConfig, default_dimensions, build_judge_input, judge_batch,
    load_langfuse_config, fetch_traces, resolve_trace_ids, upload_scores, upload_scorer_scores,
    EvalConfig, run_eval,
)
```

每个函数都是纯 Python 可调用对象，不依赖命令行参数，方便直接用 pytest 覆盖：

- 数据：`load_dataset(DataSourceSpec(...))` 读本地 parquet/json/jsonl 或 HuggingFace 仓库；
  `json_to_parquet(records, out_dir, name=..., split=...)` 把 JSON 记录转成 HF 风格 parquet。
- 执行：`run_one(sample, cfg, run_dir)` 同步跑单条，`run_batch(samples, cfg, run_dir)` 并发跑全部并支持续跑。
- 裁判：`judge_batch([JudgeInput(...)], JudgeConfig(...))`，DeepSeek 结构化 JSON 多维度评分，
  `evidence` 可选 `final_answer` 或 `trajectory`，`register_evidence_builder` 可注册自定义方式。
- 确定性 scorer：`run_scorers([ScorerSpec(...)], output, target)`，内置 exact / contains / regex / numeric，
  `register_scorer` 可注册自定义规则；正确性走硬分数，质量走 LLM 裁判，两者分开入库。
- epochs / pass@k：`agent.epochs` 让每个样本跑多次，`epoch_reducer` 选 `pass_at_k` 或 `average`；
  报告给出每个 scorer 的 `pass@1` 与 `pass@k`，裁判在确定性分数最高的 epoch 上打分。
- Langfuse：`fetch_traces` 拉 trace，`resolve_trace_ids` 关联样本，`upload_scores` 回传分数。

## 配置

`configs/example.yaml` 是一份完整示例，包含 `dataset` / `agent` / `judge` / `scorers` /
`langfuse` / `run` 六段。裁判不写 `dimensions` 时使用内置通用 5 维（相关性 / 完整性 / 逻辑连贯 /
清晰度 / 有帮助），也可在配置里覆盖成任意维度与权重；`scorers` 里给 GSM8K 配了
`#### 答案` 的正则精确匹配。

所有上传到 Langfuse 的分数（`judge_<dim>`、`judge_overall`、`scorer_<name>`）都带统一的
metadata：`evaluation_run_id` / `model` / `name` / `sample_id` / `epoch`，可直接在 Langfuse 里
按 run 或模型筛选对比。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

`tests/` 下是不联网的参考测试，覆盖数据读取、JSON 转 parquet、环境变量注入、断点续跑与重试、
裁判 JSON 解析与加权、Langfuse 配置优先级与分数上传。可在此基础上扩展你自己的 pytest。
