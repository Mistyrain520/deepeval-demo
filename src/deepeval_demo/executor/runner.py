"""单样本执行：启动 codex exec 子进程，注入 Langfuse 关联环境变量。"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ExecutorConfig
from ..data import Sample


@dataclass
class SampleRun:
    """一个样本在多个 epoch 下的执行结果。"""

    sample: Sample
    epochs: list[ExecResult]


@dataclass
class ExecResult:
    """单条样本的执行结果。"""

    sample_id: str
    input: str
    ok: bool
    answer: str | None = None
    exit_code: int | None = None
    output_path: Path | None = None
    events_path: Path | None = None
    error: str | None = None
    trace_env: dict[str, str] = field(default_factory=dict)
    rounds_used: int = 1
    held: bool | None = None
    final_correct: bool | None = None
    thread_id: str | None = None


class _SafeFormat(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def safe_format(template: str, **values: Any) -> str:
    """格式化 prompt 模板，缺失的占位符原样保留。"""
    return template.format_map(_SafeFormat(**values))


def build_env(
    run_id: str,
    sample: Sample,
    cfg: ExecutorConfig,
    *,
    epoch: int | None = None,
    round: int | None = None,
    metadata: dict[str, Any] | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """构造子进程环境，注入 Langfuse 插件的 environment/tags/metadata。"""
    env = os.environ.copy()
    meta: dict[str, str] = {
        "evaluation_run_id": run_id,
        "sample_id": sample.id,
        "model": cfg.model,
    }
    if epoch is not None:
        meta["epoch"] = str(epoch)
    if round is not None:
        meta["round"] = str(round)
    if cfg.seed is not None:
        meta["seed"] = str(cfg.seed)
    if metadata:
        meta.update({k: str(v) for k, v in metadata.items() if v is not None})
    env["LANGFUSE_CODEX_ENVIRONMENT"] = f"eval-{run_id}"
    env["LANGFUSE_CODEX_TAGS"] = json.dumps(["evaluation", run_id], ensure_ascii=False)
    env["LANGFUSE_CODEX_METADATA"] = json.dumps(meta, ensure_ascii=False)
    if extra:
        env.update(extra)
    return env


def build_command(
    cfg: ExecutorConfig,
    prompt: str,
    output_path: Path,
    *,
    executable: str | None = None,
) -> list[str]:
    """拼出 codex exec 的命令行参数列表。"""
    cmd = [
        executable or resolve_codex(cfg),
        "exec",
        prompt,
        "-m",
        cfg.model,
        "-C",
        str(cfg.project_root),
        "--sandbox",
        cfg.sandbox,
        "--json",
        "-o",
        str(output_path),
    ]
    if cfg.ephemeral:
        cmd.append("--ephemeral")
    return cmd


def resolve_codex(cfg: ExecutorConfig | None = None) -> str:
    """定位 codex 可执行文件：显式路径 -> PATH -> 已知安装目录（取最新）。"""
    if cfg is not None and cfg.codex_path:
        return str(cfg.codex_path)
    found = shutil.which("codex")
    if found:
        return found
    localappdata = os.environ.get("LOCALAPPDATA", "")
    if localappdata:
        base = Path(localappdata) / "OpenAI" / "Codex" / "bin"
        if base.is_dir():
            candidates = sorted(
                [p for p in base.glob("*/codex.exe") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return str(candidates[0])
    return "codex"


def _extract_last_message(events_text: str) -> str:
    """从 codex --json 事件流里尽力提取最后一条 assistant 文本。"""
    last_text = ""
    for line in events_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") or event.get("message")
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            last_text = content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    last_text = part["text"]
    return last_text.strip()


def _extract_thread_id(events_text: str) -> str | None:
    """从 codex --json 事件流里尽力提取 thread.started 的 thread_id。"""
    for line in (events_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "thread.started":
            continue
        thread_id = event.get("thread_id")
        if not thread_id and isinstance(event.get("thread"), dict):
            thread_id = event["thread"].get("id")
        if thread_id:
            return str(thread_id)
    return None


def build_prompt(cfg: ExecutorConfig, sample: Sample) -> str:
    """拼出给 codex 的 prompt，支持前置系统提示词。"""
    question = safe_format(
        cfg.prompt_template,
        input=sample.input,
        target=sample.target or "",
    )
    if cfg.system_prompt:
        return f"{cfg.system_prompt}\n\n{question}"
    return question


def _norm_final_answer(value: str) -> str:
    value = (value or "").strip().lower()
    value = value.replace("$", "")
    value = re.sub(r"[\s,]", "", value)
    value = value.rstrip(".")
    return value.strip()


def _extract_final_answer(text: str, pattern: str) -> str | None:
    """按 pattern 抽取最终答案；匹配不到时回退到整段文本。"""
    if not pattern:
        return None
    match = re.search(pattern, text or "", re.MULTILINE)
    if not match:
        return _norm_final_answer(text)
    if match.lastindex:
        value = match.group(match.lastindex)
    else:
        value = match.group(0)
    return _norm_final_answer(value)


def check_final_answer(
    output: str,
    target: str | None,
    pattern: str = r"####\s*(.+)$",
) -> bool:
    """判断 output 与 target 的最终答案是否一致；target 为空时返回 False。"""
    if not target:
        return False
    out = _extract_final_answer(output, pattern)
    tgt = _extract_final_answer(target, pattern)
    if out is None or tgt is None:
        return False
    return out == tgt


def _run_subprocess(
    cmd: list[str],
    env: dict[str, str],
    *,
    cwd: Path | None,
    timeout: float | None,
) -> tuple[subprocess.CompletedProcess[str] | None, str | None, str]:
    """执行子进程，返回 (proc|None, error|None, executable)。"""
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return None, f"执行超时: {exc}", cmd[0]
    except FileNotFoundError:
        return (
            None,
            (
                f"找不到 codex 命令（尝试路径: {cmd[0]}）。"
                "请设置 ExecutorConfig.codex_path 指向 codex.exe，"
                "或把 Codex CLI 所在目录加入 PATH"
            ),
            cmd[0],
        )
    return proc, None, cmd[0]


def _finish_result(
    sample: Sample,
    proc: subprocess.CompletedProcess[str] | None,
    error: str | None,
    *,
    output_path: Path,
    events_path: Path,
    env: dict[str, str],
) -> tuple[ExecResult, str | None]:
    """把子进程结果整理成 ExecResult，并返回 thread_id。"""
    events_text = proc.stdout if proc is not None else ""
    if events_path is not None:
        events_path.write_text(events_text, encoding="utf-8")
    thread_id = _extract_thread_id(events_text)

    answer = ""
    if output_path.exists():
        answer = output_path.read_text(encoding="utf-8").strip()
    if not answer:
        answer = _extract_last_message(events_text)

    if proc is None:
        return (
            ExecResult(
                sample_id=sample.id,
                input=sample.input,
                ok=False,
                error=error,
                output_path=output_path,
                events_path=events_path,
                trace_env=env,
            ),
            thread_id,
        )

    if error is None and proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-500:]
        error = f"codex 退出码 {proc.returncode}"
        if stderr_tail:
            error += f": {stderr_tail}"
    ok = proc.returncode == 0 and bool(answer)
    if not ok and error is None:
        error = "没有生成答案"

    return (
        ExecResult(
            sample_id=sample.id,
            input=sample.input,
            ok=ok,
            answer=answer or None,
            exit_code=proc.returncode,
            output_path=output_path,
            events_path=events_path,
            error=error,
            trace_env=env,
        ),
        thread_id,
    )


def _codex_once(
    sample: Sample,
    cfg: ExecutorConfig,
    run_dir: Path,
    *,
    prompt: str,
    output_path: Path,
    events_path: Path,
    resolved_run_id: str,
    epoch: int | None,
    round_num: int,
    metadata: dict[str, Any] | None,
    resume_thread_id: str | None = None,
) -> tuple[ExecResult, str | None]:
    """执行一轮 codex（首轮或 resume），返回 (ExecResult, thread_id)。"""
    env = build_env(
        resolved_run_id,
        sample,
        cfg,
        epoch=epoch,
        round=round_num,
        metadata=metadata,
    )
    executable = resolve_codex(cfg)
    if resume_thread_id is None:
        cmd = build_command(cfg, prompt, output_path, executable=executable)
        cwd: Path | None = None
    else:
        cmd = [
            executable,
            "exec",
            "resume",
            resume_thread_id,
            prompt,
            "-m",
            cfg.model,
            "--json",
            "-o",
            str(output_path),
        ]
        cwd = cfg.project_root
    proc, error, _ = _run_subprocess(
        cmd, env, cwd=cwd, timeout=cfg.timeout_seconds
    )
    return _finish_result(
        sample,
        proc,
        error,
        output_path=output_path,
        events_path=events_path,
        env=env,
    )


def run_one(
    sample: Sample,
    cfg: ExecutorConfig,
    run_dir: str | Path,
    *,
    run_id: str | None = None,
    epoch: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExecResult:
    """同步执行一条样本，返回 ExecResult。"""
    run_dir = Path(run_dir).resolve()
    answers_dir = run_dir / "answers"
    events_dir = run_dir / "events"
    answers_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    resolved_run_id = run_id or run_dir.name
    prompt = build_prompt(cfg, sample)
    output_path = answers_dir / f"{sample.id}.txt"
    events_path = events_dir / f"{sample.id}.jsonl"
    result, thread_id = _codex_once(
        sample,
        cfg,
        run_dir,
        prompt=prompt,
        output_path=output_path,
        events_path=events_path,
        resolved_run_id=resolved_run_id,
        epoch=epoch,
        round_num=1,
        metadata=metadata,
    )
    result.rounds_used = 1
    result.thread_id = thread_id
    return result


def run_one_multi_turn(
    sample: Sample,
    cfg: ExecutorConfig,
    run_dir: str | Path,
    *,
    run_id: str | None = None,
    epoch: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExecResult:
    """多轮执行一条样本：答案正确就追问，验证其是否在反驳下坚持正确答案。"""
    run_dir = Path(run_dir).resolve()
    answers_dir = run_dir / "answers"
    events_dir = run_dir / "events"
    answers_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    resolved_run_id = run_id or run_dir.name
    prompt = build_prompt(cfg, sample)
    max_rounds = max(1, cfg.max_rounds)

    output_path = answers_dir / f"{sample.id}.txt"
    events_path = events_dir / f"{sample.id}.jsonl"
    result, thread_id = _codex_once(
        sample,
        cfg,
        run_dir,
        prompt=prompt,
        output_path=output_path,
        events_path=events_path,
        resolved_run_id=resolved_run_id,
        epoch=epoch,
        round_num=1,
        metadata=metadata,
    )

    rounds_used = 1
    first_correct = bool(result.ok) and check_final_answer(
        result.answer or "", sample.target, cfg.final_answer_pattern
    )
    final_correct = first_correct if result.ok else None
    held: bool | None = None
    while (
        result.ok
        and first_correct
        and rounds_used < max_rounds
        and thread_id
    ):
        rounds_used += 1
        round_output = answers_dir / f"{sample.id}.round-{rounds_used}.txt"
        round_events = events_dir / f"{sample.id}.round-{rounds_used}.jsonl"
        result, thread_id = _codex_once(
            sample,
            cfg,
            run_dir,
            prompt=cfg.challenge_message,
            output_path=round_output,
            events_path=round_events,
            resolved_run_id=resolved_run_id,
            epoch=epoch,
            round_num=rounds_used,
            metadata=metadata,
            resume_thread_id=thread_id,
        )
        if not result.ok:
            final_correct = None
            break
        final_correct = check_final_answer(
            result.answer or "", sample.target, cfg.final_answer_pattern
        )
        if not final_correct:
            held = False
            break
        held = True

    result.rounds_used = rounds_used
    result.held = held
    result.final_correct = final_correct
    result.thread_id = thread_id
    return result
