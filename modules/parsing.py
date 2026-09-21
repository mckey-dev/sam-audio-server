"""アンカー JSON などのリクエスト値パース。"""

from __future__ import annotations

import json
from typing import Any, Literal

Anchor = tuple[Literal["+", "-"], float, float]


def parse_anchors(raw: str | None) -> list[Anchor] | None:
    """フォームのアンカー JSON を [('+'|'-', start, end), ...] に変換する。"""
    if raw is None or not str(raw).strip():
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("anchors must be valid JSON") from exc

    if not isinstance(data, list):
        raise ValueError("anchors must be a JSON list")

    anchors: list[Anchor] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError('each anchor must be ["+"|"-", start_sec, end_sec]')
        token, start, end = item
        if token not in {"+", "-"}:
            raise ValueError('anchor token must be "+" or "-"')
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError) as exc:
            raise ValueError("anchor start/end must be numbers") from exc
        if end_f <= start_f:
            raise ValueError("anchor end must be greater than start")
        anchors.append((token, start_f, end_f))
    return anchors


def parse_bool(value: Any, default: bool = False) -> bool:
    """multipart の真偽文字列を bool にする。"""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
