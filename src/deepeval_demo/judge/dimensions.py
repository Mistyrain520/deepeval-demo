"""默认评分维度。"""

from __future__ import annotations

from ..config import DEFAULT_DIMENSIONS, DimensionSpec


def default_dimensions() -> list[DimensionSpec]:
    """返回通用 5 维的深拷贝，避免调用方误改全局默认。"""
    return [d.model_copy(deep=True) for d in DEFAULT_DIMENSIONS]
