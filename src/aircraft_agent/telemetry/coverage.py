"""Versioned, deterministic report facts and invocation-bound evidence checks.

This is deliberately not a natural-language claim extractor. The catalog below
is both the report's data model and its denominator; references never select it.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

from aircraft_agent.common.utils import hash_json, stable_json, to_jsonable
from aircraft_agent.evidence.verification import read_evidence_snapshot

SCOPE_VERSION = "structured_report_claims_v1"
CLAIMS_FILE = "report_claims_v1.json"
COVERAGE_FILE = "report_coverage_v1.json"
SNAPSHOT_FILE = "report_evidence_snapshot_v1.json"
CATEGORIES = {
    "data_profile": "数据概况", "rule_execution": "规则命中摘要",
    "validation": "结果校验", "association": "关联规则挖掘结果",
}
TOOLS = {
    "profile_dataset": "data_profile", "apply_rules": "rule_execution",
    "apply_rule_library": "rule_execution", "validate_analysis": "validation",
    "mine_association_rules": "association",
}
SCOPE = {
    "version": SCOPE_VERSION,
    "unit": "每次实际调用的一个标量、声明的列表汇总或一条规则的结构化记录为一条结论；重复展示不重复计数。",
    "included": {
        "data_profile": ["每次画像的行数、列数、前30个原始字段名列表、前10个正缺失率字段(name,missing_rate)列表",
                         "各数据集最后一次画像的行数之和（每个数据集都是必要来源）"],
        "rule_execution": ["每次调用的规则总数、evaluated数、unparseable数、命中规则数",
                           "全部规则各一条记录：rule_id/name/status/severity/condition/matched_rows/match_rate/前5个sample_indices；未执行规则含message"],
        "validation": ["每次校验的status/evidence_nodes/missing_tools/warnings/errors",
                       "校验时evidence_verification.valid/verified及chain_hash（工具报告的历史状态）"],
        "association": ["每次挖掘的engine/transactions/frequent_itemsets/规则总数",
                        "六个阈值min_support/effective_min_support/min_confidence/min_lift/min_count/max_rules",
                        "原输出前10条规则各一条记录：antecedent/consequent/matched_rows/support/confidence/lift"],
    },
    "excluded": ["自由文本Agent总结、建议、工具summary、轨迹和模型备注",
                 "输入文件元数据、规则库解析/校验、数据质量、Schema/映射/适配/泛化、相关性、图表及验收综合判断",
                 "其他未声明字段、前30字段/前10缺失字段/前10关联规则之外的明细、样例索引前5项之外的明细",
                 "报告生成/运行状态及证据链自身审计元数据"],
    "interpretation": "覆盖率仅描述上述固定范围的可追溯性，不等同于诊断准确率或结论真实性。",
    "selection": "遍历全部实际调用；无引用照常纳入。未记录来源的已保存画像/校验仍纳入。按调用和数据集区分，不以工具名取最后一次证据。",
}
REASONS = {
    "missing_reference": "缺少必要证据引用或node_hash",
    "node_missing": "引用节点不在本次快照中",
    "hash_mismatch": "引用哈希、输入哈希或来源结果output_hash失配",
    "source_mismatch": "节点类型、工具、参数、调用顺序、数据集或字段值与来源不一致",
    "source_missing": "缺少实际调用记录",
    "unverifiable": "字段缺失、未声明汇总或数据无法核验",
    "snapshot_unassessable": "快照非严格v2通过状态，无法评估可验证覆盖率",
    "duplicate_claim_id": "结论ID重复；全部重复项均不计入覆盖分子",
    "conflicting_claim": "相同结论ID存在冲突记录",
    "duplicate_source_id": "来源调用ID重复，不能选择其中一条代替",
}


def _field(value: Any, path: list[str | int]) -> Any:
    for key in path:
        if isinstance(key, int) and not isinstance(key, bool) and isinstance(value, list):
            if key < 0:
                raise KeyError(key)
            value = value[key]
        elif isinstance(key, str) and isinstance(value, dict):
            value = value[key]
        else:
            raise KeyError(key)
    return value


def _project(value: Any, operation: str) -> Any:
    """Closed set of v1 transformations; never evaluate caller supplied code."""
    if operation == "identity":
        return value
    if not isinstance(value, list):
        raise ValueError("Expected list")
    if operation == "first30":
        return value[:30]
    if operation == "missing_top10":
        return [{"name": v["name"], "missing_rate": v["missing_rate"]}
                for v in value if v.get("missing_rate", 0) > 0][:10]
    if operation == "count":
        return len(value)
    if operation == "evaluated_count":
        return sum(v["status"] == "evaluated" for v in value)
    if operation == "unparseable_count":
        return sum(v["status"] == "unparseable" for v in value)
    if operation == "matched_count":
        return sum(v["status"] == "evaluated" and v["matched_rows"] > 0 for v in value)
    raise ValueError("Undeclared operation")


def _rule_record(value: dict[str, Any], association: bool) -> dict[str, Any]:
    if association:
        return {k: value[k] for k in ("antecedent", "consequent", "matched_rows", "support", "confidence", "lift")}
    result = {k: value[k] for k in ("rule_id", "name", "status", "severity")}
    if value["status"] == "evaluated":
        result.update({k: value[k] for k in ("condition", "matched_rows", "match_rate")})
        result["sample_indices"] = value["sample_indices"][:5]
    else:
        result["message"] = value["message"]
    return result


def _source_value(source: dict[str, Any], requirement: dict[str, Any]) -> Any:
    value = _field(source["result"], requirement["result_path"])
    operation = requirement["operation"]
    if operation in ("rule_record", "association_record"):
        return _rule_record(value, operation == "association_record")
    return _project(value, operation)


def display_value(value: Any) -> str:
    # The exact same text is persisted and displayed, including list/record facts.
    if value is None:
        return "null"
    if isinstance(value, str):
        return value.replace("\n", "\\n").replace("\r", "\\r").replace("`", "\\u0060")
    return stable_json(value).replace("`", "\\u0060")


def _list_or_empty(value: Any) -> list:
    return value if isinstance(value, list) else []


def _claim_specs(category: str, result: dict[str, Any]) -> list[tuple]:
    specs = []

    def add(key: str, label: str, path: list, operation: str = "identity") -> None:
        specs.append((key, label, path, operation))

    if category == "data_profile":
        for key, label, operation in [("rows", "行数", "identity"), ("columns", "列数", "identity"),
                                      ("column_names", "字段（前30）", "first30"),
                                      ("column_profiles", "存在缺失的字段（前10）", "missing_top10")]:
            add(key, label, ["profile", key], operation)
    elif category == "rule_execution":
        for key, label in [("count", "规则总数"), ("evaluated_count", "成功执行规则数"),
                           ("unparseable_count", "无法解析规则数"), ("matched_count", "命中规则数")]:
            add(key, label, ["rule_results"], key)
        for index, _ in enumerate(_list_or_empty(result.get("rule_results"))):
            add(f"rule_{index}", f"规则 {index + 1}（结构化记录）", ["rule_results", index], "rule_record")
    elif category == "validation":
        for key, label in [("status", "状态"), ("evidence_nodes", "校验时证据节点数"),
                           ("missing_tools", "缺失工具"), ("warnings", "告警"), ("errors", "错误"),
                           ("chain_hash", "校验时chain_hash")]:
            add(key, label, [key])
        for key in ("valid", "verified"):
            add(key, f"校验时{key}", ["evidence_verification", key])
    else:
        for key, label in [("engine", "后端"), ("transactions", "事务数"), ("frequent_itemsets", "频繁项集数")]:
            add(key, label, [key])
        add("count", "关联规则总数", ["rules"], "count")
        for key in ("min_support", "effective_min_support", "min_confidence", "min_lift", "min_count", "max_rules"):
            add(key, f"阈值 {key}", ["parameters", key])
        for index, _ in enumerate(_list_or_empty(result.get("rules"))[:10]):
            add(f"rule_{index}", f"关联规则 {index + 1}（结构化记录）", ["rules", index], "association_record")
    return specs


def collect_report_claims(context: Any) -> dict[str, Any]:
    """Freeze records before verification. Missing evidence cannot shrink scope."""
    calls = []
    for index, item in enumerate(context.tool_results):
        if item.get("tool") in TOOLS:
            calls.append({
                "call_id": f"{context.run_id}:tool:{index}",
                "tool_result_index": index,
                "invocation_index": item.get("evidence_tool_call_index", index),
                "tool": item["tool"], "arguments": deepcopy(item.get("arguments")),
                "result": deepcopy(item.get("result")),
                "evidence_reference": deepcopy(item.get("evidence_reference")),
                "recorded": True,
            })
    # Materialized facts from direct/unrecorded calls are not silently discarded.
    for dataset_id, profile in context.dataset_profiles.items():
        if not any(c["tool"] == "profile_dataset" and isinstance(c["result"], dict)
                   and c["result"].get("dataset_id") == dataset_id
                   and stable_json(c["result"].get("profile")) == stable_json(profile) for c in calls):
            calls.append({"call_id": f"{context.run_id}:unrecorded:profile:{dataset_id}",
                          "invocation_index": None, "tool": "profile_dataset", "arguments": None,
                          "result": {"dataset_id": dataset_id, "profile": deepcopy(profile)},
                          "evidence_reference": None, "recorded": False})
    recorded_validations = Counter(stable_json(c["result"]) for c in calls if c["tool"] == "validate_analysis")
    for index, result in enumerate(context.validation_results):
        key = stable_json(result)
        if recorded_validations[key]:
            recorded_validations[key] -= 1
        else:
            calls.append({"call_id": f"{context.run_id}:unrecorded:validation:{index}",
                          "invocation_index": None, "tool": "validate_analysis", "arguments": None,
                          "result": deepcopy(result), "evidence_reference": None, "recorded": False})
    claims: list[dict[str, Any]] = []

    def add(call: dict[str, Any], key: str, label: str, path: list, operation: str = "identity") -> None:
        requirement = {"call_id": call["call_id"], "result_path": path, "operation": operation,
                       "evidence_reference": deepcopy(call["evidence_reference"])}
        result = call["result"] if isinstance(call["result"], dict) else {}
        dataset_id = result.get("dataset_id")
        try:
            value = _source_value(call, requirement)
            available = True
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            value, available = None, False
        claim_id = f"{call['call_id']}:{key}"
        claims.append({"claim_id": claim_id, "category": TOOLS[call["tool"]], "key": key,
                       "report_location": {"section": CATEGORIES[TOOLS[call["tool"]]], "anchor": claim_id},
                       "label": label, "dataset_id": dataset_id, "value": to_jsonable(value),
                       "value_available": available, "display_value": display_value(value) if available else "未提供",
                       "aggregation": "single", "required_sources": [requirement]})

    for call in calls:
        result = call["result"] if isinstance(call["result"], dict) else {}
        for key, label, path, operation in _claim_specs(TOOLS[call["tool"]], result):
            add(call, key, label, path, operation)

    latest_rows = {}
    for claim in claims:
        if claim["category"] == "data_profile" and claim["key"] == "rows":
            latest_rows[stable_json(claim["dataset_id"])] = claim
    if latest_rows:
        rows = list(latest_rows.values())
        available = all(c["value_available"] and type(c["value"]) in (int, float) for c in rows)
        value = sum(c["value"] for c in rows) if available else None
        claims.append({"claim_id": f"{context.run_id}:datasets:total_rows", "category": "data_profile",
                       "key": "total_rows", "report_location": {"section": "数据概况", "anchor": f"{context.run_id}:datasets:total_rows"},
                       "label": "各数据集最后画像总行数", "dataset_id": None, "value": value,
                       "value_available": available, "display_value": display_value(value) if available else "未提供",
                       "aggregation": "sum", "required_sources": [deepcopy(c["required_sources"][0]) for c in rows]})
    return {"schema_version": 1, "scope_version": SCOPE_VERSION, "scope": deepcopy(SCOPE),
            "run_id": context.run_id, "source_calls": to_jsonable(calls), "claims": claims}


def _matches_scope(claim: dict[str, Any], catalog: dict[str, Any]) -> bool:
    """Bind labels and paths to the exact declaration used by the renderer."""
    try:
        if catalog.get("scope_version") != SCOPE_VERSION or catalog.get("schema_version") != 1:
            return False
        reqs = claim["required_sources"]
        if claim["key"] == "total_rows":
            latest = {}
            for source in catalog["source_calls"]:
                if source["tool"] == "profile_dataset":
                    result = source["result"] if isinstance(source["result"], dict) else {}
                    latest[stable_json(result.get("dataset_id"))] = source["call_id"]
            return bool(latest) and (
                claim["category"] == "data_profile" and claim["aggregation"] == "sum"
                and claim["dataset_id"] is None
                and claim["claim_id"] == f"{catalog['run_id']}:datasets:total_rows"
                and claim["label"] == "各数据集最后画像总行数"
                and [r["call_id"] for r in reqs] == list(latest.values())
                and all(r["result_path"] == ["profile", "rows"] and r["operation"] == "identity" for r in reqs)
                and claim["report_location"] == {"section": "数据概况", "anchor": claim["claim_id"]}
            )
        if claim["aggregation"] != "single" or len(reqs) != 1:
            return False
        req = reqs[0]
        sources = [c for c in catalog["source_calls"] if c["call_id"] == req["call_id"]]
        if len(sources) != 1:
            return False
        source = sources[0]
        category = TOOLS[source["tool"]]
        result = source["result"] if isinstance(source["result"], dict) else {}
        return (
            claim["category"] == category
            and claim["claim_id"] == f"{source['call_id']}:{claim['key']}"
            and (claim["key"], claim["label"], req["result_path"], req["operation"]) in _claim_specs(category, result)
            and claim["report_location"] == {"section": CATEGORIES[category], "anchor": claim["claim_id"]}
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def verify_report_claims(catalog: dict[str, Any], raw: Any, verification: dict[str, Any]) -> dict[str, Any]:
    """Check each necessary source and annotate the complete denominator in place."""
    assessable = verification.get("status") == "pass" and verification.get("valid") is True and verification.get("verified") is True
    nodes = raw.get("nodes", []) if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    nodes = [n for n in nodes if isinstance(n, dict)] if isinstance(nodes, list) else []
    node_map = {n["node_id"]: n for n in nodes if isinstance(n.get("node_id"), str)}
    tool_nodes = [n for n in nodes if n.get("type") == "tool_call"]
    calls_by_id = defaultdict(list)
    for source in catalog["source_calls"]:
        calls_by_id[source["call_id"]].append(source)
    id_counts = Counter(c["claim_id"] for c in catalog["claims"])
    id_variants = defaultdict(set)
    for c in catalog["claims"]:
        id_variants[c["claim_id"]].add(stable_json({k: v for k, v in c.items() if k != "validation"}))
    referenced = verified = 0
    reasons_count: Counter = Counter()
    for claim in catalog["claims"]:
        reasons: list[str] = []
        if not _matches_scope(claim, catalog):
            reasons.append("source_mismatch")
        source_checks = []
        values = []
        requirements = claim.get("required_sources") or []
        has_reference = bool(requirements) and all(isinstance(r.get("evidence_reference"), dict)
                                                  and isinstance(r["evidence_reference"].get("node_id"), str)
                                                  and r["evidence_reference"]["node_id"] for r in requirements)
        if id_counts[claim["claim_id"]] > 1:
            reasons.append("duplicate_claim_id")
            if len(id_variants[claim["claim_id"]]) > 1:
                reasons.append("conflicting_claim")
        if not requirements:
            reasons.extend(["missing_reference", "source_missing"])
        for req in requirements:
            errors: list[str] = []
            sources = calls_by_id.get(req.get("call_id"), [])
            source = sources[0] if len(sources) == 1 else None
            ref = req.get("evidence_reference")
            if len(sources) > 1:
                errors.append("duplicate_source_id")
            if source is None or not source.get("recorded"):
                errors.append("source_missing")
            if not isinstance(ref, dict) or not isinstance(ref.get("node_id"), str) or not ref.get("node_id") or not ref.get("node_hash"):
                errors.append("missing_reference")
            node = node_map.get(ref.get("node_id")) if isinstance(ref, dict) and isinstance(ref.get("node_id"), str) else None
            if isinstance(ref, dict) and ref.get("node_id") and node is None:
                errors.append("node_missing")
            if source is not None:
                try:
                    values.append(_source_value(source, req))
                except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                    errors.append("unverifiable")
                try:
                    if source["tool"] not in TOOLS:
                        errors.append("source_mismatch")
                    if stable_json(ref) != stable_json(source.get("evidence_reference")):
                        errors.append("source_mismatch")
                    result_dataset = source["result"].get("dataset_id") if isinstance(source["result"], dict) else None
                    if source["tool"] != "validate_analysis" and (not isinstance(result_dataset, str) or not result_dataset):
                        errors.append("unverifiable")
                    if claim["aggregation"] == "single" and result_dataset != claim.get("dataset_id"):
                        errors.append("source_mismatch")
                    if node is not None:
                        if node.get("node_hash") != ref.get("node_hash"):
                            errors.append("hash_mismatch")
                        index = source.get("invocation_index")
                        if (type(index) is not int or index < 0 or index >= len(tool_nodes)
                                or tool_nodes[index].get("node_id") != node.get("node_id")):
                            errors.append("source_mismatch")
                        if (node.get("type") != "tool_call" or node.get("tool_name") != source["tool"]
                                or ref.get("type") != node.get("type") or ref.get("tool_name") != source["tool"]
                                or ref.get("schema_version") != node.get("schema_version")
                                or stable_json(node.get("arguments")) != stable_json(source["arguments"])):
                            errors.append("source_mismatch")
                        if (hash_json(source["arguments"]) != node.get("input_hash")
                                or hash_json(source["result"]) != node.get("output_hash")):
                            errors.append("hash_mismatch")
                except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                    errors.append("unverifiable")
            if not assessable:
                errors.append("snapshot_unassessable")
            source_checks.append({"call_id": req.get("call_id"), "result_path": req.get("result_path"),
                                  "reasons": sorted(set(errors))})
            reasons.extend(errors)
        try:
            if not claim.get("value_available") or len(values) != len(requirements) or not values:
                raise ValueError("No verifiable value")
            if claim["aggregation"] == "single" and len(values) == 1:
                actual = values[0]
            elif claim["aggregation"] == "sum" and all(type(v) in (int, float) for v in values):
                actual = sum(values)
            else:
                raise ValueError("Undeclared aggregate")
            if stable_json(actual) != stable_json(claim["value"]) or display_value(actual) != claim["display_value"]:
                reasons.append("source_mismatch")
        except (KeyError, TypeError, ValueError):
            reasons.append("unverifiable")
        if not assessable:
            reasons.append("snapshot_unassessable")
        reasons = sorted(set(reasons))
        unique = id_counts[claim["claim_id"]] == 1
        reference_counted = bool(has_reference and unique)
        complete = not reasons and assessable
        referenced += reference_counted
        verified += complete
        reasons_count.update(reasons)
        claim["validation"] = {"has_all_references": bool(has_reference), "reference_counted": reference_counted,
                               "verified": complete, "status": "verified" if complete else "uncovered" if assessable else "unassessable",
                               "reasons": reasons, "sources": source_checks}
    denominator = len(catalog["claims"])
    snapshot = {"phase": "report_render", "chain_hash": verification.get("expected_chain_hash"),
                "node_count": verification.get("node_count"), "status": verification.get("status"),
                "artifact": SNAPSHOT_FILE if raw is not None else None,
                "note": "本次渲染快照；后续write_report/Agent轨迹追加属于最终运行状态，需另行只读校验。"}
    catalog["snapshot"] = snapshot
    return {"schema_version": 1, "scope_version": SCOPE_VERSION, "run_id": catalog["run_id"],
            "scope": deepcopy(SCOPE), "snapshot": snapshot, "snapshot_verification": verification,
            "reference_coverage": {"numerator": referenced, "denominator": denominator,
                                   "ratio": referenced / denominator if denominator else None},
            "verified_coverage": {"status": "assessable" if assessable else "unassessable",
                                  "numerator": verified if assessable else None, "denominator": denominator,
                                  "ratio": verified / denominator if denominator and assessable else None},
            "reason_counts": dict(sorted(reasons_count.items())),
            "issues": [{"claim_id": c["claim_id"], "report_location": c["report_location"],
                        "reasons": c["validation"]["reasons"]} for c in catalog["claims"] if c["validation"]["reasons"]]}


def build_report_coverage(context: Any) -> dict[str, Any]:
    catalog = collect_report_claims(context)
    raw, verification = read_evidence_snapshot(context.evidence.path)
    coverage = verify_report_claims(catalog, raw, verification)
    return {"catalog": catalog, "coverage": coverage, "raw_snapshot": raw}
