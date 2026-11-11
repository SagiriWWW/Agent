"""FP-growth adapter for SagiriWWW/Datamining-style rule mining.

关联规则挖掘统一以 FP-growth 生成树方法为准。这个适配器把 Agent 工具
层构造出的事务集合转换为 one-hot 布尔矩阵，然后调用 FP-growth 和
association_rules 生成规则。

如果本地配置了 DATAMINING_REPO_PATH，会优先复用
SagiriWWW/Datamining 中核心脚本暴露的 fpgrowth、
association_rules 和 is_same_sensor_rule。若未配置仓库，则直接使用
mlxtend.frequent_patterns 的 FP-growth 实现，并保留同样的输出格式。
"""

from __future__ import annotations

import importlib.util
import os
from itertools import combinations
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pandas as pd


DATAMINING_CORE_DIR = Path("scripts/core")
DATAMINING_REQUIRED_SYMBOLS = ("fpgrowth", "association_rules", "is_same_sensor_rule")


class DataminingUnavailable(RuntimeError):
    """Raised when neither Datamining nor mlxtend FP-growth can be loaded.

    这个异常不会一定导致 Agent 失败：mine_with_fpgrowth 会捕获它，并退回
    到本文件里的轻量 fallback，以保证 demo 在依赖不完整时也能跑通。
    """


def resolve_datamining_repo() -> Path | None:
    """Find an optional local Datamining clone.

    查找顺序：
    1. 用户显式配置的 DATAMINING_REPO_PATH；
    2. 项目附近的几个约定目录。

    找到后会复用 Datamining 项目里的 FP-growth 和同传感器过滤逻辑。
    """
    for candidate in _datamining_repo_candidates():
        repo = candidate.expanduser().resolve()
        if _find_fpgrowth_core_file(repo) is not None:
            return repo
    return None


def resolve_datamining_core_file(repo_path: Path | None = None) -> Path | None:
    """Find a Datamining Python file exposing the required FP-growth symbols."""
    configured_core = os.getenv("DATAMINING_FPGROWTH_CORE_PATH")
    if configured_core:
        path = Path(configured_core).expanduser().resolve()
        if path.exists() and _looks_like_fpgrowth_core(path):
            return path

    repo = repo_path.expanduser().resolve() if repo_path else None
    if repo is None:
        for candidate in _datamining_repo_candidates():
            candidate_repo = candidate.expanduser().resolve()
            core_file = _find_fpgrowth_core_file(candidate_repo)
            if core_file is not None:
                return core_file
        return None
    return _find_fpgrowth_core_file(repo)


def _datamining_repo_candidates() -> list[Path]:
    candidates = []
    configured = os.getenv("DATAMINING_REPO_PATH")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path("third_party/Datamining"),
            Path("../Datamining"),
            Path("Datamining"),
        ]
    )
    return candidates


def _find_fpgrowth_core_file(repo: Path) -> Path | None:
    core_dir = repo / DATAMINING_CORE_DIR
    if not core_dir.exists():
        return None
    for path in sorted(core_dir.glob("*.py")):
        if _looks_like_fpgrowth_core(path):
            return path
    return None


def _looks_like_fpgrowth_core(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
    except OSError:
        return False
    return all(symbol in text for symbol in DATAMINING_REQUIRED_SYMBOLS)


def load_datamining_core(repo_path: Path | None = None) -> ModuleType | None:
    """Load Datamining core module if a local clone is available.

    这里用 importlib 从文件路径动态加载，是为了不要求 Datamining 必须安装成
    Python package。只要用户把仓库 clone 到指定路径即可。
    """
    core_file = resolve_datamining_core_file(repo_path)
    if core_file is None:
        return None
    spec = importlib.util.spec_from_file_location("_agent_mvp_datamining_core", core_file)
    if spec is None or spec.loader is None:
        raise DataminingUnavailable(f"Could not create import spec for: {core_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in DATAMINING_REQUIRED_SYMBOLS if not hasattr(module, name)]
    if missing:
        raise DataminingUnavailable(f"Datamining core is missing required FP-growth symbols: {', '.join(missing)}")
    return module


def datamining_backend_status() -> dict[str, Any]:
    """Return a JSON-safe description of the active FP-growth backend.

    这个函数给 Agent 工具箱和报告使用：它不执行挖掘，只检查当前环境会使用
    Datamining、mlxtend 还是本地 fallback。
    """
    configured = os.getenv("DATAMINING_REPO_PATH") or ""
    core_file = resolve_datamining_core_file()
    repo = resolve_datamining_repo()
    if core_file is not None:
        try:
            core = load_datamining_core(repo)
            symbols = {name: hasattr(core, name) for name in DATAMINING_REQUIRED_SYMBOLS}
            return {
                "status": "available",
                "engine": "datamining_fpgrowth",
                "configured_path": configured,
                "repo_path": str(repo) if repo else None,
                "core_file": str(core_file) if core_file else None,
                "required_symbols": symbols,
                "summary": "SagiriWWW/Datamining FP-growth 后端可用。",
            }
        except Exception as exc:
            return {
                "status": "error",
                "engine": "datamining_fpgrowth",
                "configured_path": configured,
                "repo_path": str(repo) if repo else None,
                "core_file": str(core_file) if core_file else None,
                "required_symbols": {name: False for name in DATAMINING_REQUIRED_SYMBOLS},
                "error": str(exc),
                "summary": f"已找到 Datamining 仓库，但 FP-growth 后端加载失败：{exc}",
            }

    try:
        from mlxtend.frequent_patterns import association_rules, fpgrowth  # noqa: F401

        return {
            "status": "available",
            "engine": "mlxtend_fpgrowth",
            "configured_path": configured,
            "repo_path": None,
            "core_file": None,
            "required_symbols": {"fpgrowth": True, "association_rules": True, "is_same_sensor_rule": "fallback"},
            "summary": "未配置 Datamining 仓库，当前可使用 mlxtend FP-growth 后端。",
        }
    except ModuleNotFoundError:
        return {
            "status": "fallback",
            "engine": "local_fpgrowth_fallback",
            "configured_path": configured,
            "repo_path": None,
            "core_file": None,
            "required_symbols": {"fpgrowth": False, "association_rules": False, "is_same_sensor_rule": "fallback"},
            "summary": "Datamining 仓库和 mlxtend 均不可用，将使用本地演示 fallback。",
        }


def mine_with_fpgrowth(
    transactions: list[set[str]],
    *,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    max_len: int,
    max_antecedent: int,
    max_rules: int,
    min_count: int = 1,
    filter_same_sensor: bool = True,
) -> dict[str, Any]:
    """Run FP-growth and return Agent-friendly rule records.

    输入 transactions 是工具层生成的事务集合，例如：
    {"status=WARN", "mission_phase=climb"}

    输出统一为 JSON-friendly dict，包含 engine、rules、frequent_itemsets、
    summary，报告和 Agent 不需要关心底层到底用了 Datamining、mlxtend 还是
    fallback。
    """
    one_hot = transactions_to_one_hot(transactions)
    min_count = normalize_min_count(min_count)
    parameters = mining_parameters(
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_len=max_len,
        max_antecedent=max_antecedent,
        max_rules=max_rules,
        min_count=min_count,
        transaction_count=len(one_hot),
    )
    warnings = association_mining_warnings(len(one_hot), min_count)
    if one_hot.empty:
        return {
            "engine": datamining_backend_status()["engine"],
            "backend": datamining_backend_status(),
            "rules": [],
            "frequent_itemsets": 0,
            "parameters": parameters,
            "warnings": warnings,
            "summary": "没有可用于 FP-growth 挖掘的事务数据。",
        }

    try:
        # 优先使用正式 FP-growth 实现；只有依赖缺失时才走本地 fallback。
        fpgrowth_fn, association_rules_fn, same_sensor_fn, engine = resolve_fpgrowth_functions()
    except DataminingUnavailable:
        return mine_with_local_fpgrowth_fallback(
            transactions,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_len=max_len,
            max_antecedent=max_antecedent,
            max_rules=max_rules,
            min_count=min_count,
            filter_same_sensor=filter_same_sensor,
        )
    # FP-growth 先挖 frequent itemsets，再由 association_rules 生成 A -> B 规则。
    effective_min_support = float(parameters["effective_min_support"])
    frequent = fpgrowth_fn(one_hot, min_support=effective_min_support, use_colnames=True, max_len=max_len, verbose=0)
    if frequent.empty:
        return {
            "engine": engine,
            "backend": datamining_backend_status(),
            "rules": [],
            "frequent_itemsets": 0,
            "parameters": parameters,
            "warnings": warnings,
            "summary": "在当前阈值下，FP-growth 未发现频繁项集。",
        }

    rules_df = association_rules_fn(frequent, metric="lift", min_threshold=min_lift)
    if rules_df.empty:
        return {
            "engine": engine,
            "backend": datamining_backend_status(),
            "rules": [],
            "frequent_itemsets": int(len(frequent)),
            "parameters": parameters,
            "warnings": warnings,
            "summary": "FP-growth 已发现频繁项集，但没有关联规则满足 min_lift 阈值。",
        }

    rules_df = rules_df.copy()
    # 限制前件长度和后件长度，避免 demo 报告中出现过长、难解释的规则。
    rules_df["ante_len"] = rules_df["antecedents"].apply(len)
    rules_df["cons_len"] = rules_df["consequents"].apply(len)
    rules_df = rules_df[
        (rules_df["ante_len"] <= max_antecedent)
        & (rules_df["cons_len"] == 1)
        & (rules_df["confidence"] >= min_confidence)
    ]

    records = []
    filtered_by_min_count = 0
    for _, row in rules_df.iterrows():
        antecedents = tuple(sorted(str(x) for x in row["antecedents"]))
        consequents = tuple(sorted(str(x) for x in row["consequents"]))
        if filter_same_sensor and same_sensor_fn(antecedents, consequents):
            continue
        record = format_fpgrowth_rule(row, antecedents, consequents, len(one_hot))
        if record["matched_rows"] < min_count:
            filtered_by_min_count += 1
            continue
        records.append(record)

    records.sort(key=lambda r: (r["lift"], r["confidence"], r["support"]), reverse=True)
    visible = records[:max_rules]
    return {
        "engine": engine,
        "backend": datamining_backend_status(),
        "rules": visible,
        "frequent_itemsets": int(len(frequent)),
        "parameters": parameters,
        "warnings": warnings,
        "filtered_by_min_count": filtered_by_min_count,
        "summary": (
            f"从 {len(one_hot)} 条事务中挖掘出 {len(visible)} 条 FP-growth 关联规则"
            f"（min_count={min_count}, max_rules={max_rules}）。"
        ),
    }


def mine_with_local_fpgrowth_fallback(
    transactions: list[set[str]],
    *,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    max_len: int,
    max_antecedent: int,
    max_rules: int,
    min_count: int,
    filter_same_sensor: bool,
) -> dict[str, Any]:
    """Small dependency-free association miner for demo environments.

    This is intentionally conservative and only used when the configured
    Datamining project and mlxtend are unavailable. It keeps the same output
    contract so the Agent demo remains runnable before full dependencies are
    installed.

    注意：fallback 不是高性能 FP-growth 实现，而是枚举小规模 itemset 的保守
    算法。它适合 demo/测试，正式数据量大时应安装 mlxtend 或配置 Datamining。
    """
    n_rows = len(transactions)
    min_count = normalize_min_count(min_count)
    parameters = mining_parameters(
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_len=max_len,
        max_antecedent=max_antecedent,
        max_rules=max_rules,
        min_count=min_count,
        transaction_count=n_rows,
    )
    warnings = association_mining_warnings(n_rows, min_count)
    if not n_rows:
        return {
            "engine": "local_fpgrowth_fallback",
            "backend": datamining_backend_status(),
            "rules": [],
            "frequent_itemsets": 0,
            "parameters": parameters,
            "warnings": warnings,
            "summary": "没有可用于本地 FP-growth fallback 挖掘的事务数据。",
        }

    counts: dict[tuple[str, ...], int] = {}
    for tx in transactions:
        ordered = sorted(tx)
        for size in range(1, min(max_len, len(ordered)) + 1):
            # 枚举每条事务中的组合并计数。max_len 默认很小，所以 demo 可接受。
            for combo in combinations(ordered, size):
                counts[combo] = counts.get(combo, 0) + 1

    # support_by_itemset 只保留满足最小支持度的频繁项集。
    effective_min_support = float(parameters["effective_min_support"])
    support_by_itemset = {
        itemset: count / n_rows for itemset, count in counts.items() if count / n_rows >= effective_min_support
    }
    records = []
    filtered_by_min_count = 0
    for itemset, support in support_by_itemset.items():
        if len(itemset) < 2:
            continue
        itemset_set = set(itemset)
        for ant_size in range(1, min(max_antecedent, len(itemset) - 1) + 1):
            for antecedents in combinations(itemset, ant_size):
                consequents = tuple(sorted(itemset_set - set(antecedents)))
                if len(consequents) != 1:
                    continue
                antecedents = tuple(sorted(antecedents))
                if filter_same_sensor and default_same_sensor_rule(antecedents, consequents):
                    continue
                antecedent_support = support_by_itemset.get(antecedents, counts.get(antecedents, 0) / n_rows)
                consequent_support = support_by_itemset.get(consequents, counts.get(consequents, 0) / n_rows)
                if not antecedent_support or not consequent_support:
                    continue
                # 关联规则常用指标：
                # support = A 和 B 同时出现的比例；
                # confidence = 出现 A 时也出现 B 的比例；
                # lift = confidence / B 自身出现比例，>1 表示正相关。
                confidence = support / antecedent_support
                lift = confidence / consequent_support
                if confidence < min_confidence or lift < min_lift:
                    continue
                leverage = support - (antecedent_support * consequent_support)
                conviction = None if confidence >= 1 else (1 - consequent_support) / (1 - confidence)
                matched_rows = int(round(support * n_rows))
                if matched_rows < min_count:
                    filtered_by_min_count += 1
                    continue
                records.append(
                    {
                        "antecedent": " & ".join(antecedents),
                        "consequent": " & ".join(consequents),
                        "antecedents": list(antecedents),
                        "consequents": list(consequents),
                        "support": round(float(support), 6),
                        "confidence": round(float(confidence), 6),
                        "lift": round(float(lift), 6),
                        "leverage": round(float(leverage), 6),
                        "conviction": round(float(conviction), 6) if conviction is not None else None,
                        "matched_rows": matched_rows,
                    }
                )

    records.sort(key=lambda r: (r["lift"], r["confidence"], r["support"]), reverse=True)
    visible = records[:max_rules]
    return {
        "engine": "local_fpgrowth_fallback",
        "backend": datamining_backend_status(),
        "rules": visible,
        "frequent_itemsets": len(support_by_itemset),
        "parameters": parameters,
        "warnings": warnings,
        "filtered_by_min_count": filtered_by_min_count,
        "summary": (
            f"使用本地 FP-growth fallback 从 {n_rows} 条事务中挖掘出 {len(visible)} 条关联规则"
            f"（min_count={min_count}, max_rules={max_rules}）。"
        ),
    }


def normalize_min_count(min_count: int) -> int:
    """Keep minimum matched-row threshold in a safe integer range."""
    try:
        value = int(min_count)
    except (TypeError, ValueError):
        value = 1
    return max(1, value)


def mining_parameters(
    *,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    max_len: int,
    max_antecedent: int,
    max_rules: int,
    min_count: int,
    transaction_count: int,
) -> dict[str, Any]:
    effective_min_support = max(min_support, min_count / transaction_count) if transaction_count else min_support
    # FP-growth libraries expect min_support in [0, 1]. When a small customer
    # sample has fewer rows than min_count, keep min_count as the rule-output
    # filter but cap the mining threshold so the tool returns an empty result
    # with warnings instead of failing the whole Agent run.
    effective_min_support = min(1.0, effective_min_support)
    return {
        "min_support": round(float(min_support), 6),
        "effective_min_support": round(float(effective_min_support), 6),
        "min_confidence": round(float(min_confidence), 6),
        "min_lift": round(float(min_lift), 6),
        "max_len": int(max_len),
        "max_antecedent": int(max_antecedent),
        "max_rules": int(max_rules),
        "min_count": int(min_count),
    }


def association_mining_warnings(transaction_count: int, min_count: int) -> list[str]:
    warnings = []
    if 0 < transaction_count < 30:
        warnings.append(
            f"样本量较小（{transaction_count} 条事务），关联规则仅适合探索和演示，不应直接视为稳定业务规律。"
        )
    if transaction_count and min_count / transaction_count > 0.5:
        warnings.append(
            f"min_count={min_count} 相对 {transaction_count} 条事务偏高，可能过滤掉较多候选规则。"
        )
    if transaction_count and min_count > transaction_count:
        warnings.append(
            f"min_count={min_count} 大于事务数 {transaction_count}，本次关联规则会被视为小样本不足并过滤输出。"
        )
    return warnings


def transactions_to_one_hot(transactions: list[set[str]]) -> pd.DataFrame:
    """Convert item-set transactions to the boolean matrix required by FP-growth.

    mlxtend 的 fpgrowth 需要 one-hot 布尔矩阵：每列是一个 item，每行是一次
    事务，True 表示该事务包含该 item。
    """
    items = sorted({item for tx in transactions for item in tx})
    if not transactions or not items:
        return pd.DataFrame()
    rows = [{item: item in tx for item in items} for tx in transactions]
    return pd.DataFrame(rows, columns=items).astype(bool)


def resolve_fpgrowth_functions() -> tuple[Callable[..., Any], Callable[..., Any], Callable[[tuple[str, ...], tuple[str, ...]], bool], str]:
    """Resolve FP-growth functions from Datamining when possible, else mlxtend.

    返回四个元素：fpgrowth 函数、association_rules 函数、同传感器过滤函数、
    engine 名称。engine 会写入报告，方便知道本次运行实际用了哪个后端。
    """
    try:
        core = load_datamining_core()
    except ModuleNotFoundError:
        core = None

    if core is not None and hasattr(core, "fpgrowth") and hasattr(core, "association_rules"):
        return core.fpgrowth, core.association_rules, core.is_same_sensor_rule, "datamining_fpgrowth"

    try:
        from mlxtend.frequent_patterns import association_rules, fpgrowth
    except ModuleNotFoundError as exc:
        raise DataminingUnavailable(
            "FP-growth requires mlxtend. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return fpgrowth, association_rules, default_same_sensor_rule, "mlxtend_fpgrowth"


def default_same_sensor_rule(antecedents: tuple[str, ...], consequents: tuple[str, ...]) -> bool:
    """Fallback for Datamining's same-sensor filter.

    Datamining 项目里有“同传感器规则过滤”的业务逻辑。这里提供一个简化版：
    如果 item 名以 `数字_` 开头，且前后件都属于同一个数字 ID，就认为是同一
    传感器内部规则，可按需过滤。
    """
    import re

    ids = set()
    for item in list(antecedents) + list(consequents):
        match = re.match(r"^(\d+)_", str(item))
        if match:
            ids.add(match.group(1))
    return bool(ids) and len(ids) == 1


def format_fpgrowth_rule(row: pd.Series, antecedents: tuple[str, ...], consequents: tuple[str, ...], n_rows: int) -> dict[str, Any]:
    """Convert an mlxtend/Datamining rule row to this project's JSON format.

    这样做的好处是报告层不依赖 pandas，也不依赖 mlxtend 的 DataFrame 字段名。
    """
    support = float(row["support"])
    confidence = float(row["confidence"])
    lift = float(row["lift"])
    leverage = float(row["leverage"]) if "leverage" in row else None
    conviction = float(row["conviction"]) if "conviction" in row else None
    return {
        "antecedent": " & ".join(antecedents),
        "consequent": " & ".join(consequents),
        "antecedents": list(antecedents),
        "consequents": list(consequents),
        "support": round(support, 6),
        "confidence": round(confidence, 6),
        "lift": round(lift, 6),
        "leverage": round(leverage, 6) if leverage is not None else None,
        "conviction": round(conviction, 6) if conviction is not None else None,
        "matched_rows": int(round(support * n_rows)),
    }
