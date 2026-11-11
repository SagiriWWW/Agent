"""Command-line runner for local demos and smoke tests."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

from aircraft_agent.agent.runner import AgentRunner
from aircraft_agent.telemetry.batch import build_batch_dataset_adaptation, build_batch_rule_analysis, create_batch_job, named_batch_dataset_dirs, run_batch_job
from aircraft_agent.common.config import get_settings
from aircraft_agent.telemetry.demo_data import DEMO_DATASET_PATH, DEMO_RULE_PATH
from aircraft_agent.agent.runtime import RunStore
from aircraft_agent.tools.executor import ToolExecutor
from aircraft_agent.evidence.verification import EXIT_CODES, verify_evidence_file
from aircraft_agent.evidence.storage import check_evidence_storage
from aircraft_agent.evidence.recovery import apply_evidence_recovery, plan_evidence_recovery


_DEFAULT_GOAL = "以 Agent 为主结构，调度本地工具箱完成数据分析、结果校验和证据链报告。"


def main() -> None:
    """Run the same pipeline as the web app without starting FastAPI.

    CLI 适合做 smoke test、自动化脚本和本地调试。它与 Web 入口共用同一套
    RunStore、ToolExecutor 和 AgentRunner，因此生成的 runtime 目录结构一致。
    """
    parser = argparse.ArgumentParser(description="运行规则库驱动的数据 Agent MVP，或检查、显式恢复已有证据存储。")
    verification = parser.add_mutually_exclusive_group()
    verification.add_argument("--verify-evidence", "--verify", metavar="PATH", help="只读校验已有 evidence_chain.json，不创建运行或调用模型。")
    verification.add_argument("--verify-evidence-storage", metavar="RUN_DIR", help="只读检查运行的 JSON 快照、JSONL 日志及双文件一致性，不修复运行。")
    verification.add_argument("--recover-evidence-storage", metavar="RUN_DIR", help="默认只读输出证据存储恢复计划及 JSON；仅支持有效 pending append，不恢复 Agent 任务。")
    parser.add_argument("--apply", action="store_true", help="显式执行证据存储恢复；必须同时提供计划摘要和操作者声明。")
    parser.add_argument("--plan-digest", metavar="HEX", help="只读恢复计划返回的 64 位十六进制状态摘要；不是数字签名或身份认证。")
    parser.add_argument("--operator", metavar="TEXT", help="执行恢复的非空操作者声明；不代表已认证身份。")
    parser.add_argument("--recovery-id", metavar="HEX32", help="检查或重试已有恢复的 32 位十六进制 recovery_id。")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出快照校验或存储检查结果；恢复模式始终输出完整 JSON。")
    parser.add_argument("--demo", action="store_true", help="运行内置遥测数据工具箱 Demo。")
    parser.add_argument("--public-data", metavar="CASE_ID", help="接入已批准的RflyMAD案例并实际调用模型解读，不重训或预测。")
    parser.add_argument("--public-protocol", choices=("responses", "chat_completions"), help="公开数据解读协议；百炼入口默认Chat Completions，不修改旧流程配置。")
    parser.add_argument("--public-import-only", action="store_true", help="仅接入公开数据与已封存结果，不发起模型请求。")
    parser.add_argument("--batch-last-avg", action="store_true", help="批量运行 dataset/last_avg 下的数据集验收。")
    parser.add_argument("--batch-dir", help="批量验收数据集目录；可填 customer_variants、last_avg 或本地路径。")
    parser.add_argument("--max-datasets", type=int, default=None, help="批量验收最多处理的数据集数量。")
    parser.add_argument("--rule", help="规则库文件路径。")
    parser.add_argument("--dataset", action="append", help="数据集 CSV/XLSX 路径；可重复传入多个数据集。")
    parser.add_argument("--goal", default=None)
    args = parser.parse_args()
    if (args.public_protocol is not None or args.public_import_only) and args.public_data is None:
        parser.error("--public-protocol / --public-import-only 仅可与 --public-data 一起使用。")

    recovery_modifiers = [
        name for name, supplied in (
            ("--apply", args.apply),
            ("--plan-digest", args.plan_digest is not None),
            ("--operator", args.operator is not None),
            ("--recovery-id", args.recovery_id is not None),
        ) if supplied
    ]
    if recovery_modifiers and args.recover_evidence_storage is None:
        parser.error(f"{', '.join(recovery_modifiers)} 仅可与 --recover-evidence-storage 一起使用。")

    if args.verify_evidence is not None or args.verify_evidence_storage is not None or args.recover_evidence_storage is not None:
        mode = next(name for name, value in (
            ("--verify-evidence", args.verify_evidence),
            ("--verify-evidence-storage", args.verify_evidence_storage),
            ("--recover-evidence-storage", args.recover_evidence_storage),
        ) if value is not None)
        conflicts = [
            name for name, supplied in (
                ("--demo", args.demo),
                ("--public-data", args.public_data is not None),
                ("--batch-last-avg", args.batch_last_avg),
                ("--batch-dir", args.batch_dir is not None),
                ("--max-datasets", args.max_datasets is not None),
                ("--rule", args.rule is not None),
                ("--dataset", args.dataset is not None),
                ("--goal", args.goal is not None),
            ) if supplied
        ]
        if conflicts:
            parser.error(f"{mode} 不能与执行参数 {', '.join(conflicts)} 同时使用。")
        if args.recover_evidence_storage is not None:
            if not args.apply and (args.plan_digest is not None or args.operator is not None):
                parser.error("--plan-digest 和 --operator 仅可与 --apply 一起使用。")
            if args.apply and (args.plan_digest is None or args.operator is None or not args.operator.strip()):
                parser.error("--apply 必须同时提供 --plan-digest 和非空 --operator 声明。")
            if args.plan_digest is not None and re.fullmatch(r"[0-9a-fA-F]{64}", args.plan_digest) is None:
                parser.error("--plan-digest 必须是 64 位十六进制摘要。")
            if args.recovery_id is not None and re.fullmatch(r"[0-9a-fA-F]{32}", args.recovery_id) is None:
                parser.error("--recovery-id 必须是 32 位十六进制 recovery_id。")
            if args.plan_digest is not None:
                args.plan_digest = args.plan_digest.lower()
            if args.recovery_id is not None:
                args.recovery_id = args.recovery_id.lower()
            run_dir = Path(args.recover_evidence_storage)
            if args.apply:
                result = apply_evidence_recovery(
                    run_dir, plan_digest=args.plan_digest, operator=args.operator,
                    recovery_id=args.recovery_id,
                )
            else:
                result = plan_evidence_recovery(run_dir, recovery_id=args.recovery_id)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(result["exit_code"])
        if args.verify_evidence_storage is not None:
            result = check_evidence_storage(Path(args.verify_evidence_storage))
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                _print_evidence_storage(result)
            raise SystemExit(result["exit_code"])
        result = verify_evidence_file(Path(args.verify_evidence))
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_evidence_verification(result)
        raise SystemExit(EXIT_CODES[result["status"]])
    if args.json:
        parser.error("--json 仅可与 --verify-evidence 一起使用，或用于 --verify-evidence-storage 检查、--recover-evidence-storage 恢复。")
    if args.goal is None:
        args.goal = _DEFAULT_GOAL

    settings = get_settings()
    if args.public_data is not None:
        if args.demo or args.batch_last_avg or args.batch_dir or args.rule or args.dataset or args.max_datasets is not None:
            parser.error("--public-data 不能与旧Demo、规则、上传数据或批量模式混用。")
        from aircraft_agent.agent.public_analysis import run_public_analysis
        from aircraft_agent.telemetry.public_data import CASES
        if args.public_data not in CASES:
            parser.error("未知公开数据案例；可选：" + ", ".join(CASES))
        context = RunStore(settings.storage_dir).create_run(args.goal)
        protocol = args.public_protocol or ("chat_completions" if settings.provider in {"bailian", "dashscope", "modelstudio", "model_studio"} else settings.api_protocol)
        settings = replace(settings, api_protocol=protocol)
        result = run_public_analysis(context, settings, args.public_data, import_only=args.public_import_only)
        print(json.dumps({"run_id": context.run_id, "status": context.status, "dataset": result["dataset"],
                          "model": result["model"], "report": str(context.report_html_path)}, ensure_ascii=False, indent=2))
        raise SystemExit(0 if context.status == "complete" else 3 if result["model"]["status"] == "blocked" else 1)
    if args.batch_last_avg or args.batch_dir:
        dataset_dir = _resolve_batch_dir(args.batch_dir)
        manifest = create_batch_job(
            settings,
            goal=args.goal,
            rule_path=Path(args.rule) if args.rule else None,
            dataset_dir=dataset_dir,
            max_datasets=args.max_datasets,
        )
        result = run_batch_job(settings, str(manifest["batch_id"]))
        analysis = build_batch_rule_analysis(settings.storage_dir, str(manifest["batch_id"]))
        adaptation = build_batch_dataset_adaptation(settings.storage_dir, str(manifest["batch_id"]))
        print(f"批次 ID: {result['batch_id']}")
        print(f"批次状态: {result['status']}")
        print(f"完成: {result['completed']}/{result['dataset_total']}")
        print(f"数据目录: {result['dataset_dir']}")
        print(f"通过: {result['passed']}；需复核: {result['needs_review']}；失败: {result['failed']}")
        print(f"批量报告: {result['report_html_path']}")
        print(f"规则稳定性标签: {analysis['summary'].get('quality_counts', {})}")
        print(f"数据适配标签: {adaptation['summary'].get('quality_counts', {})}")
        return

    if args.demo:
        # --demo 使用项目自带遥测规则和 dataset/last_avg 代表数据，且后面会强制关闭外部模型 API，
        # 保证命令行演示稳定可重复。
        args.rule = str(DEMO_RULE_PATH)
        args.dataset = [str(DEMO_DATASET_PATH)]
    if not args.rule or not args.dataset:
        parser.error("请使用 --demo，或同时提供 --rule 和至少一个 --dataset。")

    if args.demo:
        # 临时覆盖配置，不修改 .env。本次 CLI demo 只走本地 deterministic Agent。
        settings = replace(settings, provider="local_demo", openai_api_key=None, force_openai=False)
    store = RunStore(settings.storage_dir)
    context = store.create_run(args.goal)
    context.status = "running"
    context.save_manifest()

    with open(args.rule, "rb") as f:
        # CLI 传入的是本地路径；保存时仍复制到 run/uploads/，确保每次运行可审计。
        store.save_bytes(context, data=f.read(), original_name=args.rule, kind="rule_library")
    for dataset in args.dataset:
        with open(dataset, "rb") as f:
            store.save_bytes(context, data=f.read(), original_name=dataset, kind="dataset")

    executor = ToolExecutor(context)
    # AgentRunner 会根据 settings 决定走模型工具调用还是本地 demo 计划。
    result = AgentRunner(context, executor, settings).run()
    context.status = "complete"
    context.save_manifest()
    print(f"运行 ID: {context.run_id}")
    print(f"运行模式: {result['mode']}")
    print(f"计划步骤数: {result.get('plan_steps', len(context.agent_plan))}")
    print(f"关联规则后端: {_association_backend_label(context.tool_results)}")
    print(f"报告路径: {context.report_html_path}")
    print(f"证据链路径: {context.evidence.path}")


def _print_evidence_verification(result: dict) -> None:
    labels = {
        "pass": "验证通过",
        "mismatch": "校验失配",
        "legacy": "旧版可读，但不可验证",
        "unsupported_version": "不支持的证据链版本",
        "invalid_structure": "非法证据链结构",
        "invalid_json": "非法 JSON",
        "file_not_found": "证据链文件不存在",
        "read_error": "证据链读取失败",
        "path_rejected": "证据链路径不允许访问",
    }
    print(f"校验结果: {labels[result['status']]} ({result['status']})")
    print(f"版本: {result['schema_version'] if result['schema_version'] is not None else '未知'}")
    print(f"节点数: {result['node_count'] if result['node_count'] is not None else '未知'}")
    print(f"valid: {str(result['valid']).lower()}")
    print(f"verified: {str(result['verified']).lower()}")
    print(f"预期 chain_hash: {result['expected_chain_hash'] or '无'}")
    print(f"实际 chain_hash: {result['actual_chain_hash'] or '无'}")
    print(f"首个失配节点 ID: {result['first_mismatch_node_id'] or '无可定位节点'}")
    index = result["first_mismatch_index"]
    print(f"首个失配节点索引（从 0 开始）: {index if index is not None else '无可定位节点'}")
    print(f"错误类型: {', '.join(error['kind'] for error in result['errors']) or '无'}")
    print(f"校验边界: {result['scope']}")


def _print_evidence_storage(result: dict) -> None:
    labels = {
        "consistent": "双文件内容一致", "snapshot_behind": "快照落后于日志",
        "snapshot_ahead": "快照超前于日志", "diverged": "双文件内容分叉",
        "log_corrupt": "追加日志损坏", "snapshot_corrupt": "快照损坏",
        "missing_log": "新模式追加日志缺失", "legacy_no_log": "旧模式无日志",
        "invalid_protocol": "存储协议不完整或非法", "file_not_found": "运行或文件不存在",
        "read_error": "存储读取失败", "busy": "正在写入，本次视图不可比较",
        "pending": "写入事务需要恢复",
    }
    status = result["status"]
    print(f"存储检查结果: {labels.get(status, status)} ({status})")
    print(f"存储协议版本: {result.get('storage_protocol_version') or '无'}")
    print(f"持久化模式: {result.get('mode') or '未知'}")
    print(f"本次视图可比较: {str(result['comparable']).lower()}")
    print(f"双文件一致: {str(result['consistent']).lower()}")
    print(f"停止追加: {str(result['write_blocked']).lower()}")
    print(f"写入状态: {result.get('write_state', 'unknown')}")
    for key, label in (("snapshot", "快照完整性"), ("log", "追加日志完整性")):
        verdict = result.get(key) or {}
        print(f"{label}: {verdict.get('status', '未检查')}")
        print(f"{label}节点数: {verdict.get('node_count', '未知')}")
        print(f"{label} chain_hash: {verdict.get('actual_chain_hash') or '无'}")
    print(f"错误类型: {', '.join(error['kind'] for error in result.get('errors', [])) or '无'}")
    print(f"检查边界: {result['scope']}")


def _association_backend_label(tool_results: list[dict]) -> str:
    for item in reversed(tool_results):
        if item.get("tool") == "mine_association_rules" and isinstance(item.get("result"), dict):
            return str(item["result"].get("engine") or "未知")
        if item.get("tool") == "check_association_backend" and isinstance(item.get("result"), dict):
            return str(item["result"].get("engine") or "未知")
    return "未执行"


def _resolve_batch_dir(value: str | None) -> Path | None:
    if not value:
        return None
    named = named_batch_dataset_dirs()
    if value in named:
        return named[value]
    return Path(value)


if __name__ == "__main__":
    main()
