"""Langfuse 凭据加载与客户端构造。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


class LangfuseConfigError(RuntimeError):
    """缺少 Langfuse 配置时抛出。"""


@dataclass
class LangfuseCredentials:
    public_key: str
    secret_key: str
    base_url: str


def load_langfuse_config() -> LangfuseCredentials:
    """按 环境变量 -> ~/.codex/langfuse.json -> <cwd>/.codex/langfuse.json 读取。"""
    config: dict[str, str] = {}
    env = os.environ
    if env.get("LANGFUSE_PUBLIC_KEY"):
        config["public_key"] = env["LANGFUSE_PUBLIC_KEY"]
    if env.get("LANGFUSE_SECRET_KEY"):
        config["secret_key"] = env["LANGFUSE_SECRET_KEY"]
    if env.get("LANGFUSE_BASE_URL") or env.get("LANGFUSE_HOST"):
        config["base_url"] = env.get("LANGFUSE_BASE_URL") or env.get("LANGFUSE_HOST", "")

    if not all(k in config for k in ("public_key", "secret_key", "base_url")):
        for path in (
            Path.home() / ".codex" / "langfuse.json",
            Path.cwd() / ".codex" / "langfuse.json",
        ):
            if not path.exists():
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            config.setdefault("public_key", raw.get("public_key") or "")
            config.setdefault("secret_key", raw.get("secret_key") or "")
            config.setdefault("base_url", raw.get("base_url") or "")

    missing = [
        key
        for key in ("public_key", "secret_key", "base_url")
        if not config.get(key)
    ]
    if missing:
        raise LangfuseConfigError(
            "缺少 Langfuse 配置: "
            + ", ".join(missing)
            + "。请设置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL，"
            "或提供 ~/.codex/langfuse.json。"
        )
    return LangfuseCredentials(
        public_key=config["public_key"],
        secret_key=config["secret_key"],
        base_url=config["base_url"],
    )


def build_client(credentials: LangfuseCredentials | None = None):
    """构造 Langfuse SDK 客户端。"""
    from langfuse import Langfuse

    creds = credentials or load_langfuse_config()
    return Langfuse(
        public_key=creds.public_key,
        secret_key=creds.secret_key,
        base_url=creds.base_url,
    )
