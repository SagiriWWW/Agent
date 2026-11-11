"""Import approved public-data projections and frozen results, without inference."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from aircraft_agent.common.config import PROJECT_ROOT


@dataclass(frozen=True)
class PublicCase:
    case_id: str
    label: str
    case: str
    publisher_kind: str
    root: Path
    manifest_hash: str


_NORMAL_ROOT = PROJECT_ROOT / "outputs/public_data/session36/run-1"
_NORMAL_HASH = "e9746c419dab80a924344cfc552a47f3843920cb5096f63b5fd7435ef86c0905"
CASES = {
    "rflymad-112-1": PublicCase(
        "rflymad-112-1", "RflyMAD / waypoint / 112_1（发布者故障案例）",
        "Real/waypoint/motor/112_1", "publisher_fault",
        PROJECT_ROOT / "outputs/public_data/session37b/run-1",
        "1a3723e62072041ea988e9bcba0329ef66a121853c36f0338eedb4b6efd7548b",
    ),
}
for _case in ("acce/452_1", "circling/396_1", "hover/101_1", "velocity/319_1", "waypoint/236_1", "acce/453_1"):
    _id = "rflymad-" + _case.split("/")[-1].replace("_", "-")
    CASES[_id] = PublicCase(_id, f"RflyMAD / {_case}（发布者正常容器）", _case,
                            "publisher_normal", _NORMAL_ROOT, _NORMAL_HASH)

METHOD_NAMES = {"single": "单来源统计", "three": "三来源统计", "forest": "Isolation Forest"}
BOUNDARIES = [
    "本次导入真实公开数据的已验收三轴投影，复用冻结分析；没有重新训练、恢复模型或预测。",
    "案例均保留开发接触身份，来源独立性未认证，不能作为独立测试集。",
    "命令参考不是物理故障起点；时间差不是在线延迟，提前或超窗报警不等于物理误报。",
    "正常容器时长不是已认证正常实飞暴露；不能计算正式误报率、precision、recall或F1。",
    "单个案例不能证明方法总体最优；历史开发结果包含未检出和报警增加。",
]


def available_cases() -> list[dict]:
    return [{"case_id": value.case_id, "label": value.label,
             "available": value.root.is_dir(), "analysis_mode": "frozen_result_reuse"}
            for value in CASES.values()]


def load_case(case_id: str) -> dict:
    if case_id not in CASES:
        raise ValueError("不在已批准的公开数据清单中")
    spec = CASES[case_id]
    root = spec.root.absolute()
    total = 0

    def source(name: str) -> bytes:
        nonlocal total
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("来源路径不允许")
        candidate = root / relative
        for component in (candidate, *candidate.parents):
            if component.is_symlink():
                raise ValueError("来源路径包含符号链接")
            if component == root:
                break
        if not candidate.resolve().is_relative_to(root.resolve()):
            raise ValueError("来源路径超出数据目录")
        size = candidate.stat().st_size
        if size > 32 * 1024**2 or total + size > 96 * 1024**2:
            raise ValueError("公开数据读取预算超限")
        raw = candidate.read_bytes()
        total += len(raw)
        return raw

    manifest_bytes = source("hashes.json")
    if hashlib.sha256(manifest_bytes).hexdigest() != spec.manifest_hash:
        raise ValueError("hash 清单完整性不符")
    hashes = json.loads(manifest_bytes)
    prefix = spec.case.replace("/", "-")
    names = [f"{prefix}-raw/features.csv", f"{prefix}-raw/manifest.json", f"{prefix}-result.json"]
    names += [f"{prefix}-{method}-scores.csv" for method in METHOD_NAMES]
    payloads, checked = {}, {}
    for name in names:
        raw = source(name)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != hashes.get(name):
            raise ValueError(f"hash 完整性不符：{name}")
        checked[name] = {"sha256": digest, "bytes": len(raw)}
        if name in names[:3]:
            payloads[name] = raw
    raw_manifest = json.loads(payloads[names[1]])
    analysis = json.loads(payloads[names[2]])
    if (analysis.get("case") != spec.case or not raw_manifest.get("source_sha256")
            or analysis.get("source_sha256") != raw_manifest["source_sha256"]
            or raw_manifest.get("test_eligible") is not False):
        raise ValueError("来源绑定或开发数据身份不一致")
    reader = csv.reader(io.StringIO(payloads[names[0]].decode("utf-8-sig")))
    columns = next(reader, [])
    count, preview = 0, []
    for row in reader:
        if not row:
            continue
        if len(preview) < 5:
            preview.append({"source_row_index": count, **dict(zip(columns, row))})
        count += 1
    if columns != raw_manifest.get("features") or count != raw_manifest.get("rows") or count <= 0:
        raise ValueError("原始投影的行列数与来源清单不一致")
    if set(analysis.get("methods", {})) != set(METHOD_NAMES):
        raise ValueError("冻结分析缺少方法")
    return {
        "case_id": spec.case_id, "label": spec.label, "case": spec.case,
        "publisher_kind": spec.publisher_kind, "analysis_mode": "frozen_result_reuse",
        "profile": {"rows": count, "columns": len(columns), "column_names": columns, "preview": preview,
                    "source_sha256": raw_manifest["source_sha256"]},
        "analysis": analysis, "features_bytes": payloads[names[0]],
        "provenance": {"schema_version": 1, "case": spec.case, "usage": "development",
                       "dataset": "RflyMAD", "source_url": "https://rfly-openha.github.io/documents/4_resources/dataset.html",
                       "source_sha256": raw_manifest["source_sha256"], "manifest_sha256": spec.manifest_hash,
                       "source_root": str(spec.root), "verified_files": checked,
                       "new_training": False, "new_prediction": False},
        "boundaries": BOUNDARIES,
    }


def build_facts(loaded: dict) -> list[dict]:
    facts = []

    def add(name, label, value, source):
        facts.append({"id": name, "label": label, "value": value, "source": source})

    add("data.rows", "三轴投影行数", loaded["profile"]["rows"], "public_dataset.json#/rows")
    add("data.columns", "三轴投影字段", loaded["profile"]["column_names"], "public_dataset.json#/column_names")
    add("data.case", "发布者案例与身份", {"case": loaded["case"], "publisher_kind": loaded["publisher_kind"]}, "public_sources.json")
    add("data.reference_us", "同ULog命令参考时间（微秒；正常容器为null）", loaded["analysis"].get("reference_us"), "public_analysis.json#/reference_us")
    for key, label in METHOD_NAMES.items():
        result = loaded["analysis"]["methods"][key]
        for field, title in (("alarm_count", "合并报警数"), ("positive_outputs", "阳性输出数"),
                             ("classifications", "报警分类"), ("support", "支持量与边界"),
                             ("alarm_time_difference_us", "离线回放相对ULog命令记录的报警时间差（微秒）")):
            add(f"{key}.{field}", f"{label}：{title}", result.get(field), f"public_analysis.json#/methods/{key}/{field}")
    return facts
