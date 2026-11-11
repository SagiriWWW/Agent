"""Explicit, state-bound recovery of narrowly defined append interruptions.

No caller of this module may infer historical fsync success from visible bytes.
Only apply mutates storage; plans and completed-id inspection are read-only.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any

from aircraft_agent.common.utils import atomic_write_bytes, stable_json
from aircraft_agent.evidence import storage as s
from aircraft_agent.evidence.verification import verify_evidence_snapshot_bytes

RECOVERY_FILE = "evidence_recovery.json"
RECOVERY_DIR = "evidence_recoveries"
RECOVERY_VERSION = 1
PRIMARY = (s.PROTOCOL_FILE, s.LOG_FILE, s.SNAPSHOT_FILE, s.PENDING_FILE,
           RECOVERY_FILE, s.LOCK_FILE, "manifest.json")
SCOPE = (
    "只恢复证据存储，不恢复 Agent 任务、不执行工具、不改业务状态或历史报告。"
    "仅支持严格 append 意图的取消或向前完成；JSONL 原始字节不变。"
    "摘要是状态绑定，不是数字签名或身份认证；操作者为未经认证的声明，时间为本机时间。"
    "本次重新同步不证明历史 fsync 成功；持续介质故障时必须人工隔离。"
)
MATERIALS = [*PRIMARY, RECOVERY_DIR, "原始异常/系统日志、故障阶段、进程与文件系统信息；保留原件并人工隔离，勿删除停止标记"]


class RecoveryRefused(ValueError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: Any) -> bytes:
    return (stable_json(value) + "\n").encode("utf-8")


def _fingerprint(data: bytes | None) -> dict[str, Any]:
    return {"exists": data is not None, "sha256": _sha(data) if data is not None else None,
            "size": len(data) if data is not None else None}


def _id(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise RecoveryRefused("recovery_id 必须为32位小写十六进制安全标识。")
    return value


def _directory(path: Path) -> None:
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise RecoveryRefused(f"目录必须为非符号链接目录：{path}")


def _identity(run: Path) -> dict[str, Any]:
    _directory(run)
    info = run.stat()
    lock = (run / s.LOCK_FILE).lstat()
    return {"resolved_path": str(run.resolve()), "device": info.st_dev, "inode": info.st_ino,
            "lock_device": lock.st_dev, "lock_inode": lock.st_ino}


def _raw_files(run: Path) -> dict[str, bytes | None]:
    return {name: s._read_regular(run / name) if s._entry_exists(run / name) else None for name in PRIMARY}


def _tree(run: Path) -> dict[str, bytes | None]:
    root = run / RECOVERY_DIR
    if not s._entry_exists(root):
        return {}
    _directory(root)
    result: dict[str, bytes | None] = {"/": None}
    for folder in sorted(root.iterdir()):
        _id(folder.name)
        _directory(folder)
        result[folder.name + "/"] = None
        for item in sorted(folder.iterdir()):
            # Atomic helper remnants are retained and bound by the plan too.
            result[folder.name + "/" + item.name] = s._read_regular(item)
    return result


def _guard(raw: bytes | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = s._strict_json(raw)
    if (not isinstance(value, dict) or set(value) != {"recovery_protocol_version", "recovery_id", "phase", "audit_sha256"}
            or type(value["recovery_protocol_version"]) is not int or value["recovery_protocol_version"] != RECOVERY_VERSION
            or value["phase"] not in {"active", "released"}):
        raise RecoveryRefused("恢复停止/代际标记损坏或版本未知。")
    _id(value["recovery_id"])
    if value["phase"] == "active":
        if value["audit_sha256"] is not None:
            raise RecoveryRefused("active 标记不能声明完成审计。")
    elif not isinstance(value["audit_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", value["audit_sha256"]) is None:
        raise RecoveryRefused("released 标记缺少完成审计哈希。")
    return value


def _validate_intent(intent: Any, rid: str, identity: dict[str, Any]) -> None:
    """Validate an intent before either incomplete-backup resume or archive loading."""
    fields = {"recovery_protocol_version", "recovery_id", "original_binding", "plan_digest",
              "action", "reason", "operator_declaration", "local_time",
              "identity_authenticated", "time_trusted", "before", "result"}
    _id(rid)
    if (not isinstance(intent, dict) or set(intent) != fields
            or type(intent["recovery_protocol_version"]) is not int
            or intent["recovery_protocol_version"] != RECOVERY_VERSION or intent["recovery_id"] != rid
            or intent["identity_authenticated"] is not False or intent["time_trusted"] is not False
            or intent["result"] != "started"
            or not isinstance(intent["operator_declaration"], str) or not intent["operator_declaration"].strip()
            or not isinstance(intent["local_time"], str)
            or not isinstance(intent["action"], str) or intent["action"] not in {"cancel_append", "complete_append"}
            or not isinstance(intent["reason"], str) or not intent["reason"].strip()
            or not isinstance(intent["plan_digest"], str)
            or re.fullmatch(r"[0-9a-f]{64}", intent["plan_digest"]) is None):
        raise RecoveryRefused("恢复意图字段、类型、身份或声明不合法。")
    if datetime.fromisoformat(intent["local_time"]).tzinfo is None:
        raise RecoveryRefused("恢复本地时间缺少时区声明。")
    binding = intent["original_binding"]
    if (not isinstance(binding, dict) or set(binding) != {"run_identity", "files"}
            or stable_json(binding["run_identity"]) != stable_json(identity)
            or not isinstance(binding["files"], dict) or set(binding["files"]) != set(PRIMARY)
            or _sha(_json(binding))[:32] != rid):
        raise RecoveryRefused("恢复意图原始运行身份或文件绑定不匹配。")
    for fingerprint in binding["files"].values():
        if (not isinstance(fingerprint, dict) or set(fingerprint) != {"exists", "sha256", "size"}
                or type(fingerprint["exists"]) is not bool):
            raise RecoveryRefused("恢复意图原始文件指纹结构非法。")
        if fingerprint["exists"]:
            if (not isinstance(fingerprint["sha256"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", fingerprint["sha256"]) is None
                    or type(fingerprint["size"]) is not int or fingerprint["size"] < 0):
                raise RecoveryRefused("恢复意图原始文件哈希或大小非法。")
        elif fingerprint["sha256"] is not None or fingerprint["size"] is not None:
            raise RecoveryRefused("不存在的原始文件不能声明哈希或大小。")


def _validate_completed_archive(tree: dict[str, bytes | None], rid: str,
                                identity: dict[str, Any]) -> dict[str, Any]:
    """Validate preserved originals and finalized facts without trusting a phase label."""
    _id(rid)
    prefix = rid + "/"
    intent_raw, audit_raw = tree.get(prefix + "intent.json"), tree.get(prefix + "final.json")
    if intent_raw is None or audit_raw is None:
        raise RecoveryRefused("恢复意图或独立完成审计缺失。")
    intent, audit = s._strict_json(intent_raw), s._strict_json(audit_raw)
    _validate_intent(intent, rid, identity)
    audit_fields = {"recovery_protocol_version", "recovery_id", "plan_digest", "execution_plan_digest",
                    "action", "reason", "operator_declaration", "local_time", "identity_authenticated",
                    "time_trusted", "before", "after", "intent_sha256", "result",
                    "log_resynced_this_recovery", "historical_fsync_proven", "release_condition"}
    for value, fields in ((audit, audit_fields),):
        if (not isinstance(value, dict) or set(value) != fields
                or type(value["recovery_protocol_version"]) is not int
                or value["recovery_protocol_version"] != RECOVERY_VERSION or value["recovery_id"] != rid
                or value["identity_authenticated"] is not False or value["time_trusted"] is not False
                or not isinstance(value["operator_declaration"], str) or not value["operator_declaration"].strip()
                or not isinstance(value["local_time"], str)):
            raise RecoveryRefused("恢复意图/审计字段、类型、身份或声明不合法。")
        if datetime.fromisoformat(value["local_time"]).tzinfo is None:
            raise RecoveryRefused("恢复本地时间缺少时区声明。")
        if not isinstance(value["plan_digest"], str) or re.fullmatch(r"[0-9a-f]{64}", value["plan_digest"]) is None:
            raise RecoveryRefused("恢复原计划摘要非法。")
    initial = {}
    for name in PRIMARY:
        fingerprint = intent["original_binding"]["files"][name]
        original = tree.get(prefix + name + ".original")
        if _fingerprint(original) != fingerprint:
            raise RecoveryRefused(f"恢复原始备份缺失、损坏或存在性不符：{name}")
        initial[name] = original
    action, reason, records, target = _classify(initial)
    before = _summary(initial)
    expected_after = {**initial, s.SNAPSHOT_FILE: target, s.PENDING_FILE: None,
                      RECOVERY_FILE: _json(_active(rid))}
    if (intent["action"] != action or intent["reason"] != reason
            or stable_json(intent["before"]) != stable_json(before)
            or audit["result"] != "finalized" or audit["action"] != action or audit["reason"] != reason
            or audit["plan_digest"] != intent["plan_digest"]
            or not isinstance(audit["execution_plan_digest"], str)
            or re.fullmatch(r"[0-9a-f]{64}", audit["execution_plan_digest"]) is None
            or audit["intent_sha256"] != _sha(intent_raw)
            or stable_json(audit["before"]) != stable_json(before)
            or stable_json(audit["after"]) != stable_json(_summary(expected_after))
            or audit["log_resynced_this_recovery"] is not True
            or audit["historical_fsync_proven"] is not False
            or audit["release_condition"] != "bound released marker; finalized audit alone does not release writers"):
        raise RecoveryRefused("恢复完成审计与意图、原始备份或预期收尾状态不匹配。")
    previous_guard = _guard(initial[RECOVERY_FILE])
    if previous_guard is not None and previous_guard["phase"] != "released":
        raise RecoveryRefused("恢复原现场包含未完成的前次恢复，不能建立放行历史。")
    _validate_attempt_audits(tree, rid, intent)
    return {"intent": intent, "intent_raw": intent_raw, "initial": initial, "audit": audit,
            "audit_raw": audit_raw, "action": action, "reason": reason, "records": records,
            "target": target, "previous_guard": previous_guard}


def _validate_attempt_audits(tree: dict, rid: str, intent: dict) -> None:
    for key, data in tree.items():
        if not key.startswith(rid + "/"):
            continue
        name = key.split("/", 1)[1]
        if not name.startswith(("failure-", "release-")):
            continue
        kind = name.split("-", 1)[0]
        if data is None or name != kind + "-" + _sha(data) + ".json":
            raise RecoveryRefused("独立恢复尝试审计的文件名/原始哈希不匹配。")
        attempt = s._strict_json(data)
        if (not isinstance(attempt, dict) or attempt.get("recovery_id") != rid
                or attempt.get("plan_digest") != intent["plan_digest"]
                or attempt.get("identity_authenticated") is not False
                or attempt.get("time_trusted") is not False
                or attempt.get("result") not in ({"failed", "interrupted"} if kind == "failure" else {"release_prepared"})):
            raise RecoveryRefused("独立恢复尝试审计身份/结果不匹配。")
        if kind == "failure" and attempt.get("manual_isolation_required") is not False:
            raise RecoveryRefused("持久化故障审计要求人工隔离；不能以可见 released 标记解除停写。")


def inspect_recovery_state(run: Path) -> tuple[dict[str, Any] | None, bytes]:
    """Writer/checker gate only. Never creates or continues a recovery."""
    raw = None
    try:
        raw = s._read_regular(run / RECOVERY_FILE) if s._entry_exists(run / RECOVERY_FILE) else None
        tree = _tree(run)
        if raw is None and not tree:
            return None, b""
        guard = _guard(raw)
        if guard is None:
            raise RecoveryRefused("存在未完成恢复材料但没有 released 标记。")
        if guard["phase"] != "released":
            raise RecoveryRefused("恢复 active，必须显式重新预检并继续。")
        identity = _identity(run)
        visited = _released_history(tree, guard, identity)
        folders = {key[:-1] for key in tree if key.endswith("/") and key != "/"}
        if folders != visited:
            raise RecoveryRefused("存在未完成、孤立或尚未放行的恢复目录。")
        return {**guard, "write_blocked": False}, raw or b""
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RecursionError) as exc:
        return {"write_blocked": True, "reason": s._diagnostic(exc)}, raw or b""


def _released_history(tree: dict, guard: dict | None, identity: dict) -> set[str]:
    visited: set[str] = set()
    while guard is not None:
        rid = guard["recovery_id"]
        if guard["phase"] != "released" or rid in visited:
            raise RecoveryRefused("恢复放行历史未完成或存在循环。")
        visited.add(rid)
        archive = _validate_completed_archive(tree, rid, identity)
        if _sha(archive["audit_raw"]) != guard["audit_sha256"]:
            raise RecoveryRefused("恢复放行标记与独立完成审计哈希不匹配。")
        guard = archive["previous_guard"]
    return visited


def _summary(raw: dict[str, bytes | None]) -> dict[str, Any]:
    records, _ = s._parse_log(raw[s.LOG_FILE] or b"")
    snapshot, _ = verify_evidence_snapshot_bytes(raw[s.SNAPSHOT_FILE] or b"")
    return {"files": {name: _fingerprint(data) for name, data in raw.items()},
            "log_record_count": len(records), "snapshot_node_count": len(snapshot["nodes"]),
            "chain_hash": s._envelope([r["node"] for r in records])["chain_hash"],
            "tail_node_ids": records[-1]["tail_node_ids"] if records else []}


def _classify(raw: dict[str, bytes | None]) -> tuple[str, str, list[dict], bytes]:
    for name in (s.PROTOCOL_FILE, s.LOG_FILE, s.SNAPSHOT_FILE, s.PENDING_FILE, s.LOCK_FILE):
        if raw[name] is None:
            raise RecoveryRefused(f"必要文件缺失：{name}；无有效 append pending 的异常、旧模式及部分初始化不恢复。")
    protocol = s._strict_json(raw[s.PROTOCOL_FILE])
    if stable_json(protocol) != stable_json(s.PROTOCOL):
        raise RecoveryRefused("未知或非法存储协议；拒绝旧模式迁移。")
    pending = s._strict_json(raw[s.PENDING_FILE])
    fields = {*s.PROTOCOL, "operation", "committed_count", "previous_chain_hash", "record"}
    if (not isinstance(pending, dict) or set(pending) != fields
            or stable_json({k: pending[k] for k in s.PROTOCOL}) != stable_json(s.PROTOCOL)
            or pending["operation"] != "append" or type(pending["committed_count"]) is not int
            or pending["committed_count"] < 0):
        raise RecoveryRefused("pending 必须为协议有效、字段和类型严格有效的 append 事务意图；不恢复 initialize。")
    records, verdict = s._parse_log(raw[s.LOG_FILE])
    if verdict["status"] != "pass":
        raise RecoveryRefused(f"日志非法/损坏（不截断或补齐）：{verdict.get('errors')}")
    snapshot, verdict = verify_evidence_snapshot_bytes(raw[s.SNAPSHOT_FILE])
    if verdict["status"] != "pass":
        raise RecoveryRefused(f"快照未通过严格核验：{verdict.get('errors')}")
    count = pending["committed_count"]
    if len(records) not in {count, count + 1}:
        raise RecoveryRefused("日志不是 pending 声明的原前缀或恰好多一条完整记录。")
    prefix = records[:count]
    if pending["previous_chain_hash"] != s._envelope([r["node"] for r in prefix])["chain_hash"]:
        raise RecoveryRefused("pending 原前缀 chain_hash 不匹配。")
    candidate = pending["record"]
    trial, verdict = s._parse_log(b"".join(_json(r) for r in [*prefix, candidate]))
    if verdict["status"] != "pass":
        raise RecoveryRefused(f"pending 候选记录/序号/完整字段/哈希/父关系/tail 非法：{verdict.get('errors')}")
    prefix_nodes = [r["node"] for r in prefix]
    if len(records) == count:
        if stable_json(snapshot["nodes"]) != stable_json(prefix_nodes):
            raise RecoveryRefused("A 要求快照恰等于原前缀；拒绝超前、分叉或意图不符。")
        return "cancel_append", "A：取消未追加事务，保留原节点集合，不执行意图中的新节点。", records, raw[s.SNAPSHOT_FILE]
    if stable_json(records[-1]) != stable_json(candidate):
        raise RecoveryRefused("完整可见记录与 pending 意图不匹配（包含完整内容、序号及 tail）。")
    if stable_json(snapshot["nodes"]) not in {stable_json(prefix_nodes), stable_json([r["node"] for r in trial])}:
        raise RecoveryRefused("B 要求快照为原前缀或完整日志；拒绝超前、分叉。")
    return "complete_append", "B：本次重新同步完整可见记录，向前完成快照及事务；历史 fsync 结果仍未知。", records, _json(s._envelope([r["node"] for r in records]))


def _result(status: str, reason: str, **values: Any) -> dict[str, Any]:
    return {"status": status, "exit_code": 0 if status in {"planned", "no_recovery_needed", "completed"} else 8,
            "reason": reason, "plan_digest": None, "recovery_id": None, "scope": SCOPE,
            "manual_materials": MATERIALS, **values}


def _base_binding(identity: dict, raw: dict) -> dict:
    return {"run_identity": identity, "files": {k: _fingerprint(v) for k, v in raw.items()}}


def _build_locked(run: Path, recovery_id: str | None) -> tuple[dict, dict]:
    identity, raw, tree = _identity(run), _raw_files(run), _tree(run)
    binding = _base_binding(identity, raw)
    guard = _guard(raw[RECOVERY_FILE])
    if recovery_id is not None:
        recovery_id = _id(recovery_id)
    unfinished = [key[:-1] for key in tree if key.endswith("/") and key != "/"
                  and tree.get(key + "final.json") is None]
    if len(unfinished) > 1:
        raise RecoveryRefused("多个未完成恢复目录，需要人工核对。")
    rid = recovery_id or (guard["recovery_id"] if guard and guard["phase"] == "active" else None)
    rid = rid or (unfinished[0] if unfinished else None)
    if rid is None and raw[s.PENDING_FILE] is None:
        result = s._inspect_locked(run)[0]
        if not result["write_blocked"]:
            return _result("no_recovery_needed", "健康存储，无需恢复。", storage=result), {}
    rid = rid or _sha(_json(binding))[:32]
    prefix = rid + "/"
    intent_raw = tree.get(prefix + "intent.json")
    initial = raw
    intent = None
    if intent_raw is not None:
        intent = s._strict_json(intent_raw)
        _validate_intent(intent, rid, identity)
        _validate_attempt_audits(tree, rid, intent)
        initial = {}
        for name in PRIMARY:
            expected = intent["original_binding"]["files"][name]
            backup = tree.get(prefix + name + ".original")
            if expected["exists"]:
                if backup is None:
                    if guard and guard["recovery_id"] == rid:
                        raise RecoveryRefused(f"恢复开始后原始备份缺失：{name}")
                    backup = raw[name]
                if _fingerprint(backup) != expected:
                    raise RecoveryRefused(f"原始备份损坏或现场变化：{name}；不覆盖备份。")
            elif backup is not None:
                raise RecoveryRefused(f"本应不存在的原文件出现备份：{name}")
            initial[name] = backup
    elif recovery_id is not None and prefix not in tree:
        # Explicit initial id must be the id from this exact read-only plan.
        if recovery_id != _sha(_json(binding))[:32]:
            raise RecoveryRefused("指定 recovery_id 不属于此运行现场。")
    elif prefix in tree:
        if any(k.startswith(prefix) for k in tree if not k.endswith("/")):
            raise RecoveryRefused("恢复目录缺少意图但已有材料，需要人工核对。")
        if rid != _sha(_json(binding))[:32]:
            raise RecoveryRefused("尚未写入意图的恢复目录不属于本次现场。")
    action, reason, records, target = _classify(initial)
    if intent and (intent["action"] != action or intent["reason"] != reason
                   or stable_json(intent["before"]) != stable_json(_summary(initial))):
        raise RecoveryRefused("恢复意图的决定或原现场摘要不符。")
    audit_raw = tree.get(prefix + "final.json")
    audit = s._strict_json(audit_raw) if audit_raw else None
    prior_ids = _released_history(tree, _guard(initial[RECOVERY_FILE]), identity)
    released_ids = _released_history(tree, guard, identity) if guard and guard["phase"] == "released" else set()
    folders = {key[:-1] for key in tree if key.endswith("/") and key != "/"}
    if folders - prior_ids - released_ids - {rid}:
        raise RecoveryRefused("存在原意图以外的孤立恢复目录，需要人工核对。")
    if audit is not None:
        _validate_completed_archive(tree, rid, identity)
        if rid in released_ids:
            # Completed retries never undo subsequent ordinary appends/recoveries.
            outcome = "completed"
        else:
            outcome = "planned"
    else:
        outcome = "planned"
    if outcome != "completed":
        for name in PRIMARY:
            allowed = [initial[name]]
            if name == s.SNAPSHOT_FILE:
                allowed.append(target)
            elif name == s.PENDING_FILE and raw[s.SNAPSHOT_FILE] == target:
                allowed.append(None)
            elif name == RECOVERY_FILE:
                allowed.append(_json(_active(rid)))
            if raw[name] not in allowed:
                raise RecoveryRefused(f"恢复当前状态超出原意图允许范围：{name}")
        started = raw[RECOVERY_FILE] == _json(_active(rid))
        if not started and any(raw[n] != initial[n] for n in PRIMARY):
            raise RecoveryRefused("目标发生变化但缺少本次 active 停止标记。")
        if unfinished and unfinished[0] != rid:
            raise RecoveryRefused("另一个未完成恢复存在，不能开始本次恢复。")
        # Verify every existing original, even before intent publication or on retry.
        for name, data in initial.items():
            key = prefix + name + ".original"
            if key in tree and tree[key] != data:
                raise RecoveryRefused(f"原始备份不符：{name}；不覆盖。")
    phase = "completed" if outcome == "completed" else "finalized" if audit else "snapshot_published" if raw[s.PENDING_FILE] is None else "active" if guard and guard["recovery_id"] == rid else "backup" if prefix in tree else "not_started"
    plan = _result(outcome, reason, recovery_id=rid, action=action, phase=phase, before=_summary(initial),
                   current=_summary(raw), run_identity=identity, storage_protocol=s.PROTOCOL,
                   binding={**binding, "recovery_files": {k: ({"exists": True, "kind": "directory"} if v is None
                                                            else {**_fingerprint(v), "kind": "file"}) for k, v in tree.items()}},
                   original_plan_digest=intent["plan_digest"] if intent else None,
                   audit=audit, historical_fsync_proven=False)
    plan["plan_digest"] = _sha(_json(plan))
    return plan, {"raw": raw, "initial": initial, "intent": intent, "target": target, "records": records, "tree": tree}


def plan_evidence_recovery(run_dir: Path, *, recovery_id: str | None = None) -> dict[str, Any]:
    run = Path(run_dir)
    try:
        _directory(run)
        with s.storage_lock(run, shared=True, blocking=False):
            return _build_locked(run, recovery_id)[0]
    except BlockingIOError:
        return _result("busy", "会话14的存储锁忙；未取得可比较视图。", recovery_id=recovery_id)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError, s.EvidenceStorageError) as exc:
        return _result("refused", s._diagnostic(exc), recovery_id=recovery_id)


def _active(rid: str) -> dict:
    return {"recovery_protocol_version": RECOVERY_VERSION, "recovery_id": rid, "phase": "active", "audit_sha256": None}


def _sync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("同步目标必须为普通文件。")
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(path: Path, data: bytes) -> None:
    atomic_write_bytes(path, data)
    s._sync_directory(path.parent)
    if s._read_regular(path) != data:
        raise OSError(f"发布后原始字节校验失败：{path.name}")


def _immutable(path: Path, data: bytes) -> None:
    if s._entry_exists(path):
        if s._read_regular(path) != data:
            raise RecoveryRefused(f"已有备份/审计与预期不符，不能覆盖：{path.name}")
    else:
        _publish(path, data)
    _sync_file(path)
    s._sync_directory(path.parent)
    if s._read_regular(path) != data:
        raise OSError(f"备份/审计同步后校验失败：{path.name}")


def _checkpoint(stage: str) -> None:
    """Fault-injection seam for interruptions between durable stages."""


def _execute(run: Path, plan: dict, context: dict, operator: str) -> dict:
    rid = plan["recovery_id"]
    folder = run / RECOVERY_DIR / rid
    initial = context["initial"]
    intent = context["intent"] or {
        "recovery_protocol_version": RECOVERY_VERSION, "recovery_id": rid,
        "original_binding": _base_binding(plan["run_identity"], initial),
        "plan_digest": plan["plan_digest"], "action": plan["action"], "reason": plan["reason"],
        "operator_declaration": operator, "local_time": datetime.now().astimezone().isoformat(),
        "identity_authenticated": False, "time_trusted": False, "before": plan["before"], "result": "started",
    }
    started = False
    try:
        folder.parent.mkdir(exist_ok=True)
        _directory(folder.parent)
        s._sync_directory(run)
        folder.mkdir(exist_ok=True)
        _directory(folder)
        s._sync_directory(folder.parent)
        _checkpoint("directory")
        _immutable(folder / "intent.json", _json(intent))
        _checkpoint("intent")
        for name, data in initial.items():
            if data is not None:
                _immutable(folder / (name + ".original"), data)
            _checkpoint("backup:" + name)
        s._sync_directory(folder)
        s._sync_directory(folder.parent)
        s._sync_directory(run)
        _checkpoint("backups_verified")
        # Mark attempted publication too: a replace may have happened before error.
        started = True
        _publish(run / RECOVERY_FILE, _json(_active(rid)))
        _checkpoint("active")
        for name in (s.LOG_FILE, s.PROTOCOL_FILE, s.SNAPSHOT_FILE):
            _sync_file(run / name)
        s._sync_directory(run)
        _checkpoint("resynced")
        if plan["action"] == "complete_append":
            _publish(run / s.SNAPSHOT_FILE, context["target"])
        if s._read_regular(run / s.LOG_FILE) != initial[s.LOG_FILE] or s._read_regular(run / s.SNAPSHOT_FILE) != context["target"]:
            raise OSError("恢复目标内容不符；JSONL 必须保持原始字节。")
        _checkpoint("snapshot")
        if s._entry_exists(run / s.PENDING_FILE):
            (run / s.PENDING_FILE).unlink()
        s._sync_directory(run)
        _checkpoint("pending_removed")
        current = _raw_files(run)
        after = _summary(current)
        audit = plan["audit"] or {
            "recovery_protocol_version": RECOVERY_VERSION, "recovery_id": rid,
            "plan_digest": intent["plan_digest"], "execution_plan_digest": plan["plan_digest"],
            "action": plan["action"], "reason": plan["reason"], "operator_declaration": operator,
            "local_time": datetime.now().astimezone().isoformat(), "identity_authenticated": False,
            "time_trusted": False, "before": plan["before"], "after": after,
            "intent_sha256": _sha(_json(intent)), "result": "finalized",
            "log_resynced_this_recovery": True, "historical_fsync_proven": False,
            "release_condition": "bound released marker; finalized audit alone does not release writers",
        }
        if audit["after"] != after:
            raise RecoveryRefused("已持久化完成审计的目标状态已变化。")
        _immutable(folder / "final.json", _json(audit))
        _checkpoint("audit")
        execution_audit = {
            "recovery_protocol_version": RECOVERY_VERSION, "recovery_id": rid,
            "plan_digest": intent["plan_digest"], "execution_plan_digest": plan["plan_digest"],
            "action": plan["action"], "reason": plan["reason"], "operator_declaration": operator,
            "local_time": datetime.now().astimezone().isoformat(), "identity_authenticated": False,
            "time_trusted": False, "before": plan["current"], "after": after,
            "final_audit_sha256": _sha(_json(audit)), "result": "release_prepared",
            "release_condition": "released marker required; this record alone does not prove release",
        }
        payload = _json(execution_audit)
        _immutable(folder / ("release-" + _sha(payload) + ".json"), payload)
        _checkpoint("release_audit")
        released = {**_active(rid), "phase": "released", "audit_sha256": _sha(_json(audit))}
        _publish(run / RECOVERY_FILE, _json(released))
        _checkpoint("released")
        result = s._inspect_locked(run)[0]
        if result["write_blocked"]:
            raise OSError("恢复后存储没有 ready，仍须停止追加。")
        return _result("completed", plan["reason"], recovery_id=rid, plan_digest=plan["plan_digest"],
                       action=plan["action"], audit=audit, execution_audit=execution_audit, storage=result,
                       reconstruct_instance_required=True, historical_fsync_proven=False)
    except BaseException as exc:
        isolation = False
        if started:
            try:
                _publish(run / RECOVERY_FILE, _json(_active(rid)))
            except BaseException:
                isolation = True
        reason = f"恢复未完整成功：{s._diagnostic(exc)}；保留原始备份，重新预检同一 recovery_id 后显式继续。"
        if isolation:
            reason += " 无法可靠持久化停止标记，必须在任何重启/写入前人工隔离运行目录。"
        # A failure record is independent of business evidence. If even this
        # cannot be persisted, retain intent/guard and report the limitation.
        try:
            if s._read_regular(folder / "intent.json") == _json(intent):
                failure = {
                    "recovery_protocol_version": RECOVERY_VERSION, "recovery_id": rid,
                    "plan_digest": intent["plan_digest"], "execution_plan_digest": plan["plan_digest"],
                    "action": plan["action"], "reason": reason, "operator_declaration": operator,
                    "local_time": datetime.now().astimezone().isoformat(), "identity_authenticated": False,
                    "time_trusted": False, "before": plan["before"], "after": _failure_summary(run),
                    "result": "failed" if isinstance(exc, Exception) else "interrupted",
                    "manual_isolation_required": isolation,
                }
                payload = _json(failure)
                _immutable(folder / ("failure-" + _sha(payload) + ".json"), payload)
        except BaseException:
            reason += " 本次失败审计未能可靠保存；保留原异常与系统日志供人工核对。"
        if not isinstance(exc, Exception):
            exc.add_note(reason)
            raise
        return _result("failed", reason, recovery_id=rid, plan_digest=plan["plan_digest"],
                       manual_isolation_required=isolation, action=plan["action"])


def _failure_summary(run: Path) -> dict:
    """Keep unknown counts/tails explicit when a failed write damaged a file."""
    raw = _raw_files(run)
    records, log = s._parse_log(raw[s.LOG_FILE] or b"")
    _, snapshot = verify_evidence_snapshot_bytes(raw[s.SNAPSHOT_FILE] or b"")
    return {"files": {k: _fingerprint(v) for k, v in raw.items()},
            "log_record_count": len(records) if log["status"] == "pass" else None,
            "snapshot_node_count": snapshot.get("node_count"),
            "tail_node_ids": (records[-1]["tail_node_ids"] if records else []) if log["status"] == "pass" else None,
            "log_verdict": log["status"], "snapshot_verdict": snapshot["status"]}


def apply_evidence_recovery(run_dir: Path, *, plan_digest: str, operator: str,
                            recovery_id: str | None = None) -> dict[str, Any]:
    if not isinstance(plan_digest, str) or re.fullmatch(r"[0-9a-f]{64}", plan_digest) is None:
        return _result("refused", "apply 必须携带对应预检的64位小写十六进制计划摘要。")
    if not isinstance(operator, str) or not operator.strip():
        return _result("refused", "apply 必须提供未经认证的非空操作者声明。")
    run = Path(run_dir)
    try:
        _directory(run)
        with s.storage_lock(run, shared=False, blocking=False):
            plan, context = _build_locked(run, recovery_id)
            if plan["plan_digest"] != plan_digest:
                return _result("stale_plan", "文件/身份/恢复阶段已变化，计划失效；请重新只读预检。", current_plan=plan,
                               recovery_id=plan["recovery_id"])
            if plan["status"] == "completed":
                return {**plan, "storage": s._inspect_locked(run)[0]}
            if plan["status"] != "planned":
                return plan
            return _execute(run, plan, context, operator)
    except BlockingIOError:
        return _result("busy", "存储独占锁忙，拒绝执行。", recovery_id=recovery_id)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError, s.EvidenceStorageError) as exc:
        return _result("refused", s._diagnostic(exc), recovery_id=recovery_id)
