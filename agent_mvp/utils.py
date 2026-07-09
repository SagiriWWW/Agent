"""Small shared helpers for hashing, JSON serialization, and filenames."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return a compact UTC timestamp for manifests and evidence nodes."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    """Return SHA-256 hex digest for an in-memory bytes payload."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file by streaming chunks instead of loading it all into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        # 1MB 一块读取，避免大文件上传时占用过多内存。
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def to_jsonable(value: Any) -> Any:
    """Convert pandas/numpy/path/datetime values into JSON-safe primitives.

    pandas/numpy 里的 NaN、Timestamp、np.int64 等类型不能被标准 json 直接
    序列化。这个函数是所有 manifest、报告、证据链写 JSON 前的统一出口。
    """
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        # JSON 标准里没有 NaN/Infinity；这里统一转成 None，避免前端或其他
        # 语言读取 evidence_chain.json 时出错。
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    try:
        # numpy/pandas 是可选依赖导入；如果某个轻量环境没有它们，工具函数
        # 仍然能处理普通 Python 类型。
        import numpy as np
        import pandas as pd

        if isinstance(value, np.generic):
            return to_jsonable(value.item())
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)


def stable_json(value: Any) -> str:
    """Serialize deterministically so hashes are reproducible."""
    return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_json(value: Any) -> str:
    """Hash any JSON-like payload after deterministic serialization."""
    return sha256_bytes(stable_json(value).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write a JSON file after converting values to JSON-safe primitives."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def truncate_text(text: str, limit: int = 1200) -> str:
    """Keep model/tool summaries compact for prompts, manifests, and reports."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def safe_filename(name: str) -> str:
    """Sanitize user-provided filenames before writing into run directories."""
    keep = []
    for ch in name.strip().replace("\\", "_").replace("/", "_"):
        if ch.isalnum() or ch in {"-", "_", ".", " ", "(", ")"}:
            keep.append(ch)
        else:
            keep.append("_")
    result = "".join(keep).strip()
    return result or "upload.bin"
