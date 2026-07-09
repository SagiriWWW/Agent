"""Command-line runner for local demos and smoke tests."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .agent import AgentRunner
from .batch import build_batch_dataset_adaptation, build_batch_rule_analysis, create_batch_job, named_batch_dataset_dirs, run_batch_job
from .config import get_settings
from .demo_data import DEMO_DATASET_PATH, DEMO_RULE_PATH
from .runtime import RunStore
from .tools import ToolExecutor


def main() -> None:
    """Run the same pipeline as the web app without starting FastAPI.

    CLI 适合做 smoke test、自动化脚本和本地调试。它与 Web 入口共用同一套
    RunStore、ToolExecutor 和 AgentRunner，因此生成的 runtime 目录结构一致。
    """
    parser = argparse.ArgumentParser(description="运行规则库驱动的数据 Agent MVP。")
    parser.add_argument("--demo", action="store_true", help="运行内置遥测数据工具箱 Demo。")
    parser.add_argument("--batch-last-avg", action="store_true", help="批量运行 dataset/last_avg 下的数据集验收。")
    parser.add_argument("--batch-dir", help="批量验收数据集目录；可填 customer_variants、last_avg 或本地路径。")
    parser.add_argument("--max-datasets", type=int, default=None, help="批量验收最多处理的数据集数量。")
    parser.add_argument("--rule", help="规则库文件路径。")
    parser.add_argument("--dataset", action="append", help="数据集 CSV/XLSX 路径；可重复传入多个数据集。")
    parser.add_argument("--goal", default="以 Agent 为主结构，调度本地工具箱完成数据分析、结果校验和证据链报告。")
    args = parser.parse_args()

    settings = get_settings()
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
