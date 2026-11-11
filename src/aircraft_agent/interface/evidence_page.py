"""Read-only evidence verification result presentation."""

from __future__ import annotations

import html
import json
from typing import Any
from urllib.parse import quote


STATUS_LABELS = {
    "pass": "PASS · 验证通过",
    "mismatch": "FAIL · 完整性失配",
    "legacy": "旧版可读 · 未验证",
    "unsupported_version": "不支持的版本 · 未验证",
    "invalid_structure": "非法结构 · 未验证",
    "invalid_json": "非法 JSON · 未验证",
    "file_not_found": "证据文件不存在 · 未验证",
    "read_error": "读取失败 · 未验证",
    "path_rejected": "访问路径被拒绝 · 未验证",
}


def evidence_url(run_id: str, *, api: bool = False) -> str:
    run_path = quote(run_id, safe="")
    return f"/api/runs/{run_path}/evidence/verify" if api else f"/runs/{run_path}/evidence"


def evidence_page_body(run_id: str, result: dict[str, Any]) -> str:
    """Render only the current verification result, without reading run files."""
    status = result["status"]
    passed = status == "pass" and result["valid"] is True and result["verified"] is True
    badge = "ok" if passed else "warn" if status == "legacy" else "danger"
    label = STATUS_LABELS.get(status, "无法完成校验")
    if status == "pass" and not passed:
        label = "未验证"

    def value(name: str, fallback: str = "无") -> str:
        item = result.get(name)
        if isinstance(item, bool):
            return str(item).lower()
        return html.escape(str(item)) if item is not None else fallback

    index = result.get("first_mismatch_index")
    node_id = result.get("first_mismatch_node_id")
    if index is None and node_id is None:
        location = "不适用（旧版未验证）" if status == "legacy" else "无可定位节点" if result.get("errors") else "无（未发现失配节点）"
    else:
        location = "见下方节点 ID 与索引"

    fields = [
        ("schema_version · 版本", value("schema_version", "未知")),
        ("node_count · 节点数", value("node_count", "未知")),
        ("valid", value("valid")),
        ("verified", value("verified")),
        ("expected_chain_hash · 文件中记录的预期值", value("expected_chain_hash", "未提供")),
        ("actual_chain_hash · 本次计算的实际值", value("actual_chain_hash", "不可计算")),
        ("first_mismatch_node_id · 首个失配节点 ID", value("first_mismatch_node_id", "无可定位节点")),
        ("first_mismatch_index · 首个失配索引（从 0 开始）", value("first_mismatch_index", "无可定位索引")),
    ]
    details = "".join(f"<div><dt>{html.escape(name)}</dt><dd><code>{item}</code></dd></div>" for name, item in fields)
    errors = "".join(
        "<li class='evidence-error'><strong>"
        + html.escape(str(error.get("kind", "unknown")))
        + "</strong><pre>"
        + html.escape(json.dumps(error, ensure_ascii=False, indent=2))
        + "</pre></li>"
        for error in result.get("errors", [])
    )
    error_block = f"<ol class='evidence-errors'>{errors}</ol>" if errors else "<p>无校验错误。</p>"
    return (
        "<style>"
        ".evidence-result,.evidence-result *{box-sizing:border-box}"
        ".evidence-result{min-width:0;overflow-wrap:anywhere}"
        ".evidence-fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:18px 0 0}"
        ".evidence-fields>div{min-width:0;background:#f8fbfc;border:1px solid #d6dde6;border-radius:6px;padding:12px}"
        ".evidence-fields dt{font-size:13px;color:#617080;margin-bottom:7px}"
        ".evidence-fields dd{margin:0;font-size:14px;line-height:1.7}"
        ".evidence-fields code{overflow-wrap:anywhere;word-break:break-word}"
        ".evidence-errors{list-style:none;padding:0;margin:0}.evidence-error+li{margin-top:18px}"
        ".evidence-error pre{font-size:13px;overflow-wrap:anywhere}"
        "@media(max-width:600px){.evidence-fields{grid-template-columns:1fr}.evidence-result .panel{padding:14px}}"
        "</style><div class='evidence-result'>"
        f"<section class='panel'><h2>当前快照校验结果</h2><p><span class='badge {badge}'>{html.escape(label)}</span></p>"
        f"<p>运行 ID：<code>{html.escape(run_id)}</code></p>"
        "<p class='muted'>每次打开或重新校验均读取磁盘当前 evidence_chain.json；页面结果来自本次读取。</p>"
        f"<dl class='evidence-fields'>{details}</dl></section>"
        f"<section class='panel'><h2>失配定位与错误类型</h2><p>{location}。索引从 0 开始；链级错误可能没有对应节点。</p>{error_block}</section>"
        f"<section class='panel'><h2>校验边界</h2><p>{html.escape(str(result.get('scope', '')))}</p>"
        "<p class='muted'>该入口仅执行只读校验，不修复、迁移或重写证据文件。</p></section></div>"
    )
