"""统一结构化日志：基于 Loguru，输出扁平 JSON lines。"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from loguru import logger as _logger

_configured = False


def _ensure_configured() -> None:
    """在首次真正写日志时套用默认配置，避免 import 即锁定配置。"""
    if not _configured:
        configure_logging()


def _flatten(message: str) -> str:
    """把 loguru serialize=True 的嵌套 JSON 压成扁平 JSON 字符串。"""
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return message
    record = data.get("record") or {}
    extra = record.get("extra") or {}
    payload: dict[str, Any] = {
        "ts": (record.get("time") or {}).get("repr"),
        "level": (record.get("level") or {}).get("name"),
        "logger": extra.get("logger") or record.get("name"),
        "event": extra.get("event"),
        "msg": record.get("message"),
    }
    for key, value in extra.items():
        if key in ("logger", "event"):
            continue
        payload[key] = value
    exception = record.get("exception")
    if exception:
        if isinstance(exception, dict):
            payload["exception"] = {
                "type": exception.get("type"),
                "value": exception.get("value"),
            }
        else:
            payload["exception"] = str(exception)
    return json.dumps(payload, ensure_ascii=False, default=str)


def _console_sink(message: str) -> None:
    sys.stderr.write(_flatten(message) + "\n")


class _FileSink:
    """带简单大小轮转与保留期的 JSON lines 文件 sink。"""

    def __init__(
        self,
        path: str | Path,
        rotation_mb: int = 10,
        retention_days: int = 7,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rotation_bytes = rotation_mb * 1024 * 1024
        self.retention_days = retention_days

    def write(self, message: str) -> None:
        self._rotate()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(_flatten(message) + "\n")
        self._prune()

    def _rotate(self) -> None:
        try:
            if self.path.exists() and self.path.stat().st_size >= self.rotation_bytes:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                target = self.path.with_name(
                    f"{self.path.stem}.{stamp}{self.path.suffix}"
                )
                self.path.rename(target)
        except OSError:
            pass

    def _prune(self) -> None:
        cutoff = time.time() - self.retention_days * 86400
        pattern = f"{self.path.stem}.*{self.path.suffix}"
        for candidate in self.path.parent.glob(pattern):
            try:
                if candidate.is_file() and candidate.stat().st_mtime < cutoff:
                    candidate.unlink()
            except OSError:
                pass


def configure_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
) -> None:
    """配置全局日志；只生效一次，避免入口重复调用叠加 handler。"""
    global _configured
    if _configured:
        return
    _logger.remove()
    _logger.add(
        _console_sink,
        level=level,
        serialize=True,
        format="{message}",
        enqueue=True,
        diagnose=True,
        backtrace=False,
    )
    if log_file:
        _logger.add(
            _FileSink(log_file),
            level=level,
            serialize=True,
            format="{message}",
            enqueue=True,
            diagnose=True,
            backtrace=False,
        )
    _configured = True


def get_logger(name: str):
    """返回绑定了来源模块名的 logger。"""
    return _logger.bind(logger=name)


def log_event(logger, level: str | int, event: str, msg: str = "", **fields: Any) -> None:
    """手动打一条结构化事件日志。"""
    _ensure_configured()
    logger.log(level, msg, event=event, **fields)


@contextmanager
def log_span(logger, event: str, **fields: Any) -> Iterator[None]:
    """记录一段耗时：进入发 start，退出发 done，异常发 error 后重抛。"""
    start = time.perf_counter()
    log_event(logger, "INFO", f"{event}.start", **fields)
    try:
        yield
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_event(
            logger,
            "ERROR",
            f"{event}.error",
            duration_ms=duration_ms,
            error=str(exc),
            **fields,
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_event(
            logger,
            "INFO",
            f"{event}.done",
            duration_ms=duration_ms,
            **fields,
        )
