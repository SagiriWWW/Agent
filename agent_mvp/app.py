"""FastAPI web/API entrypoint for the MVP."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from .agent import AgentRunner
from .batch import (
    BATCH_GOAL,
    build_batch_dataset_adaptation,
    build_batch_generalization_readiness,
    build_batch_hit_samples,
    build_batch_mapping_regression,
    build_batch_review_decisions,
    build_batch_rule_analysis,
    build_batch_schema_mapping_review,
    compare_batch_rule_versions,
    create_batch_job,
    default_batch_dataset_dir,
    get_batch_manifest,
    list_batch_manifests,
    named_batch_dataset_dirs,
    run_batch_job,
)
from .config import MODEL_RUN_MODE_AUTO, MODEL_RUN_MODE_LOCAL, MODEL_RUN_MODE_REQUIRED, build_run_settings, get_settings
from .demo_data import DEMO_DATASET_PATH, DEMO_RULE_PATH, PROJECT_ROOT
from .evidence import EvidenceStore
from .models import StoredFile
from .reporting import markdown_to_html
from .review_decisions import append_rule_mapping_decision, load_review_decisions
from .review_promotions import build_review_promotion_report, render_review_promotion_markdown
from .runtime import RunContext, RunStore
from .tools import TOOLBOX_CATALOG, ToolExecutor
from .utils import read_json


settings = get_settings()
store = RunStore(settings.storage_dir)
app = FastAPI(title="Rule Mining Agent MVP", version="0.1.0")

DEFAULT_GOAL = "以 Agent 为主结构，调度本地工具箱完成飞行遥测数据分析、规则命中、关联挖掘、结果校验和证据链报告。"
ALLOWED_RULE_EXTENSIONS = {".md", ".yaml", ".yml"}
ALLOWED_DATASET_EXTENSIONS = {".csv", ".xlsx"}

UI_STYLE = """
<style>
body{margin:0;background:#f5f7fa;color:#17212b;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1240px;margin:0 auto;padding:24px 20px 56px}
h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin:0 0 10px}h3{font-size:16px;margin:0 0 6px}
p,li,td,th,label,input,select,textarea,button{font-size:14px;line-height:1.6}
a{color:#2563a8;text-decoration:none;font-weight:700}a:hover{text-decoration:underline}
code{background:#eef3f7;border:1px solid #dce5ee;border-radius:4px;padding:2px 5px;word-break:break-word}
.topbar,.actions,.nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.topbar{justify-content:space-between;border-bottom:1px solid #d6dde6;padding-bottom:14px;margin-bottom:20px}
.panel{border:1px solid #d6dde6;background:#fff;border-radius:8px;padding:18px;margin:16px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.metric{border:1px solid #d6dde6;background:#f8fbfc;border-radius:8px;padding:14px}.metric strong{display:block;font-size:24px}
.button-link,button{display:inline-flex;align-items:center;justify-content:center;min-height:38px;border:1px solid #cbd6e2;border-radius:6px;background:#fff;color:#243447;padding:8px 12px;font-weight:800;text-decoration:none;cursor:pointer}
.primary{background:#0f766e;border-color:#0f766e;color:#fff}.secondary{background:#fff}
form{display:grid;gap:12px}label{display:grid;gap:6px;font-weight:750}
input[type=text],input[type=file],select,textarea{width:100%;border:1px solid #c9d4df;border-radius:6px;padding:9px;background:#fff}
textarea{min-height:120px}.hint,.muted{color:#617080}
.table-wrap{overflow-x:auto;border:1px solid #d6dde6;border-radius:8px;background:#fff}
table{width:100%;border-collapse:collapse;min-width:780px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #e5ebf1;padding:10px 12px}th{background:#eef3f7}
.badge{display:inline-flex;min-height:23px;align-items:center;border-radius:6px;border:1px solid #d6dde6;background:#f8fafc;padding:3px 8px;font-weight:750;font-size:12px}
.ok{background:#e8f5ee;color:#17673a;border-color:#b7dec6}.warn{background:#fff6df;color:#8a5200;border-color:#e8cc85}.danger{background:#fdecea;color:#a93a2f;border-color:#f1b4ad}
.decision-line{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.decision-form{display:inline-flex;gap:6px;align-items:center}.decision-form input[type=text]{width:150px;min-height:30px;padding:5px 7px}.mini-button{min-height:30px;padding:4px 8px}.confirm{background:#e8f5ee;color:#17673a}.reject{background:#fdecea;color:#a93a2f}
pre{white-space:pre-wrap;background:#0f172a;color:#e2e8f0;border-radius:8px;padding:14px;overflow:auto}
</style>
"""


def _page(title: str, body: str, actions: str = "") -> str:
    nav = (
        '<div class="nav">'
        '<a class="button-link secondary" href="/">控制台</a>'
        '<a class="button-link secondary" href="/demo-ui">标准演示</a>'
        '<a class="button-link secondary" href="/runs/new">客户数据分析</a>'
        '<a class="button-link secondary" href="/batch-ui">批量验收</a>'
        '<a class="button-link secondary" href="/toolbox-ui">能力中心</a>'
        '<a class="button-link secondary" href="/docs/mvp">MVP 文档</a>'
        '</div>'
    )
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>{UI_STYLE}</head><body><main><div class='topbar'><div><h1>{html.escape(title)}</h1></div>{nav}</div>{actions}{body}</main></body></html>"


def _artifact_path(run_id: str, artifact_name: str) -> Path:
    return store.get_run_dir(run_id) / "artifacts" / artifact_name


def _load_artifact(run_id: str, artifact_name: str) -> dict[str, Any]:
    path = _artifact_path(run_id, artifact_name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{artifact_name} 不存在。")
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _manifest(run_id: str) -> dict[str, Any]:
    path = store.get_run_dir(run_id) / "manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="运行不存在。")
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _json_block(data: Any) -> str:
    return f"<pre>{html.escape(json.dumps(data, ensure_ascii=False, indent=2))}</pre>"


def _metrics(items: list[tuple[str, Any]]) -> str:
    cards = "".join(f"<div class='metric'><strong>{html.escape(str(value))}</strong><span>{html.escape(label)}</span></div>" for label, value in items)
    return f"<section class='grid'>{cards}</section>"


def _artifact_links(run_id: str, names: list[str]) -> str:
    links = []
    for name in names:
        path = store.get_run_dir(run_id) / ("review_decisions.json" if name == "review_decisions.json" else f"artifacts/{name}")
        if path.exists():
            links.append(f'<a class="button-link secondary" href="/runs/{html.escape(run_id)}/artifacts/{html.escape(name)}">{html.escape(name)}</a>')
    return f"<div class='actions'>{''.join(links)}</div>"


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    body = """
    <section class="panel"><h2>飞行遥测智能分析平台</h2>
    <p class="muted">以 Agent 为主结构，调度本地工具箱完成数据画像、规则校验、映射复核、关联挖掘、可视化报告和证据链。</p>
    <div class="actions"><a class="button-link primary" href="/runs/new">客户数据分析</a><a class="button-link secondary" href="/demo-ui">标准演示场景</a><a class="button-link secondary" href="/batch-ui">批量验收</a></div></section>
    """
    return _page("飞行遥测智能分析平台", body)


@app.get("/demo-ui", response_class=HTMLResponse)
def demo_ui() -> str:
    body = """
    <section class="panel"><h2>标准演示场景</h2>
    <form action="/demo" method="post"><label>分析目标<textarea name="goal">运行内置 Demo，生成可视化报告。</textarea></label><button class="primary" type="submit">启动演示</button></form></section>
    """
    return _page("标准演示场景", body)


@app.post("/demo")
def run_demo(background_tasks: BackgroundTasks, goal: str = Form(DEFAULT_GOAL)) -> RedirectResponse:
    context = store.create_run(goal)
    context.run_kind = "demo"
    store.save_bytes(context, data=DEMO_RULE_PATH.read_bytes(), original_name=DEMO_RULE_PATH.name, kind="rule_library")
    store.save_bytes(context, data=DEMO_DATASET_PATH.read_bytes(), original_name=DEMO_DATASET_PATH.name, kind="dataset")
    local_settings, model_config = build_run_settings(settings, mode=MODEL_RUN_MODE_LOCAL)
    context.model_config = model_config
    context.save_manifest()
    background_tasks.add_task(run_agent_background, context.run_id, True, local_settings)
    return RedirectResponse(f"/runs/{context.run_id}/progress", status_code=303)


@app.get("/runs/new", response_class=HTMLResponse)
def new_run_ui() -> str:
    body = """
    <section class="panel"><h2>客户数据分析</h2>
    <form action="/runs" method="post" enctype="multipart/form-data">
      <label>任务目标<textarea name="goal">分析客户数据并生成可复核报告。</textarea></label>
      <label>规则库文件<input type="file" name="rule_file" accept=".md,.yaml,.yml"></label>
      <label>数据集文件<input type="file" name="dataset_files" accept=".csv,.xlsx" multiple></label>
      <label>模型模式<select name="model_mode"><option value="model_auto">外部模型优先</option><option value="local">本地稳定模式</option><option value="model_required">强制外部模型</option></select></label>
      <label>模型供应商<input type="text" name="model_provider" value="openai"></label>
      <label>模型名称<input type="text" name="model_name" value=""></label>
      <button class="primary" type="submit">开始分析</button>
    </form></section>
    """
    return _page("客户数据分析", body)


@app.post("/runs")
async def create_run(
    background_tasks: BackgroundTasks,
    goal: str = Form(DEFAULT_GOAL),
    model_mode: str = Form(MODEL_RUN_MODE_AUTO),
    model_provider: str = Form("openai"),
    model_name: str = Form(""),
    rule_file: UploadFile = File(...),
    dataset_files: list[UploadFile] = File(...),
) -> RedirectResponse:
    context = store.create_run(goal)
    run_settings, model_config = build_run_settings(settings, mode=model_mode, provider=model_provider, model=model_name)
    context.model_config = model_config
    try:
        if Path(rule_file.filename or "").suffix.lower() not in ALLOWED_RULE_EXTENSIONS:
            raise ValueError("规则库文件格式不支持，请上传 .md/.yaml/.yml。")
        rule_bytes = await rule_file.read()
        store.save_bytes(context, data=rule_bytes, original_name=rule_file.filename or "rules.yaml", kind="rule_library", media_type=rule_file.content_type)
        for dataset in dataset_files:
            if Path(dataset.filename or "").suffix.lower() not in ALLOWED_DATASET_EXTENSIONS:
                raise ValueError("数据集文件格式不支持，请上传 .csv/.xlsx。")
            data = await dataset.read()
            store.save_bytes(context, data=data, original_name=dataset.filename or "dataset.csv", kind="dataset", media_type=dataset.content_type)
    except Exception as exc:
        context.status = "failed"
        context.error = str(exc)
        context.save_manifest()
        return RedirectResponse(f"/runs/{context.run_id}", status_code=303)
    context.save_manifest()
    background_tasks.add_task(run_agent_background, context.run_id, False, run_settings)
    return RedirectResponse(f"/runs/{context.run_id}/progress", status_code=303)


def run_agent_background(run_id: str, force_local: bool = False, run_settings=None) -> None:
    context = _rehydrate_run_context(run_id)
    try:
        if force_local:
            local_settings, _ = build_run_settings(run_settings or settings, mode=MODEL_RUN_MODE_LOCAL)
            active_settings = local_settings
        else:
            active_settings = run_settings or settings
        AgentRunner(context, ToolExecutor(context), active_settings).run()
    except Exception as exc:
        context.status = "failed"
        context.error = str(exc)
        context.emit_event("run_failed", title="运行失败", message=str(exc), payload={"error": str(exc)})
        context.save_manifest()


def _rehydrate_run_context(run_id: str) -> RunContext:
    manifest = _manifest(run_id)
    run_dir = store.get_run_dir(run_id)
    context = RunContext(
        run_id=str(manifest.get("run_id") or run_id),
        run_dir=run_dir,
        user_goal=str(manifest.get("user_goal") or ""),
        evidence=EvidenceStore(run_dir),
    )
    for file_id, file_data in (manifest.get("files") or {}).items():
        if not isinstance(file_data, dict):
            continue
        context.files[str(file_id)] = StoredFile(
            file_id=str(file_data.get("file_id") or file_id),
            kind=str(file_data.get("kind") or ""),
            original_name=str(file_data.get("original_name") or ""),
            path=Path(str(file_data.get("path") or "")),
            sha256=str(file_data.get("sha256") or ""),
            size_bytes=int(file_data.get("size_bytes") or 0),
            media_type=file_data.get("media_type"),
            rows=file_data.get("rows"),
            columns=file_data.get("columns"),
        )
    context.dataset_file_ids = [str(item) for item in manifest.get("dataset_file_ids") or []]
    context.rule_file_ids = [str(item) for item in manifest.get("rule_file_ids") or []]
    context.toolbox_manifest = list(manifest.get("toolbox_manifest") or [])
    context.agent_plan = list(manifest.get("agent_plan") or [])
    context.agent_trace = list(manifest.get("agent_trace") or [])
    context.validation_results = list(manifest.get("validation_results") or [])
    context.model_config = dict(manifest.get("model_config") or {})
    context.run_kind = str(manifest.get("run_kind") or "analysis")
    context.report_markdown_path = Path(str(manifest["report_markdown_path"])) if manifest.get("report_markdown_path") else None
    context.report_html_path = Path(str(manifest["report_html_path"])) if manifest.get("report_html_path") else None
    context.status = str(manifest.get("status") or "created")
    context.error = manifest.get("error")
    context.event_count = int(manifest.get("event_count") or 0)
    return context


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_page(run_id: str) -> str:
    manifest = _manifest(run_id)
    report_path = Path(str(manifest.get("report_html_path") or ""))
    actions = f"<div class='actions'><a class='button-link secondary' href='/api/runs/{run_id}'>JSON</a><a class='button-link secondary' href='/runs/{run_id}/progress'>运行进度</a></div>"
    if report_path.exists():
        return report_path.read_text(encoding="utf-8")
    body = f"<section class='panel'><h2>运行状态：{html.escape(str(manifest.get('status')))}</h2><p>{html.escape(str(manifest.get('error') or '报告尚未生成。'))}</p></section>"
    return _page("运行详情", body, actions)


@app.get("/api/runs/{run_id}")
def run_json(run_id: str) -> JSONResponse:
    manifest = _manifest(run_id)
    artifacts_dir = store.get_run_dir(run_id) / "artifacts"
    links = {"events": f"/api/runs/{run_id}/events"}
    if (artifacts_dir / "schema_mapping.json").exists():
        links["schema_mapping"] = f"/runs/{run_id}/schema-mapping"
    if (artifacts_dir / "schema_mapping_review.json").exists():
        links["schema_mapping_review"] = f"/runs/{run_id}/schema-mapping-review"
    if (artifacts_dir / "dataset_adaptation.json").exists():
        links["dataset_adaptation"] = f"/runs/{run_id}/dataset-adaptation"
    if (artifacts_dir / "mapping_execution_plan.json").exists():
        links["mapping_execution_plan"] = f"/runs/{run_id}/mapping-execution-plan"
    if (artifacts_dir / "mapping_regression_validation.json").exists():
        links["mapping_regression_validation"] = f"/runs/{run_id}/mapping-regression"
    if (artifacts_dir / "generalization_readiness.json").exists():
        links["generalization_readiness"] = f"/runs/{run_id}/generalization-readiness"
    return JSONResponse({**manifest, "links": links, "event_count": int(manifest.get("event_count") or 0)})


@app.get("/api/runs/{run_id}/events")
def run_events(run_id: str) -> Response:
    _manifest(run_id)
    path = store.get_run_dir(run_id) / "run_events.jsonl"
    chunks = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            chunks.append(f"event: run_event\ndata: {line}\n\n")
    return Response("".join(chunks), media_type="text/event-stream; charset=utf-8")


@app.get("/runs/{run_id}/progress", response_class=HTMLResponse)
def run_progress(run_id: str) -> str:
    _manifest(run_id)
    body = f"""
    <section class='panel'><h2>Agent 运行进度</h2><p class='muted'>Run ID: <code>{html.escape(run_id)}</code></p>
    <p>数据接口：<code>/api/runs/{html.escape(run_id)}</code></p>
    <pre id='events'></pre></section>
    <script>
    const runId = "{html.escape(run_id)}";
    const es = new EventSource(`/api/runs/${{runId}}/events`);
    es.addEventListener("run_event", ev => {{ document.getElementById("events").textContent += ev.data + "\\n"; }});
    </script>
    """
    return _page("Agent 运行进度", body)


@app.get("/runs/{run_id}/artifacts/{artifact_name}")
def artifact(run_id: str, artifact_name: str) -> Response:
    allowed = {
        "rule_validation.json", "rule_validation.md", "schema_review.json", "schema_review.md",
        "schema_mapping.json", "schema_mapping.md", "schema_suggestions.yaml", "schema_mapping_review.json",
        "schema_mapping_review.md", "dataset_adaptation.json", "dataset_adaptation.md",
        "mapping_execution_plan.json", "mapping_execution_plan.md", "mapping_regression_validation.json",
        "mapping_regression_validation.md", "generalization_readiness.json", "generalization_readiness.md",
        "review_decisions.json", "visualizations.html", "evidence_chain.json",
    }
    if artifact_name not in allowed:
        raise HTTPException(status_code=404, detail="artifact 不允许访问。")
    path = store.get_run_dir(run_id) / ("review_decisions.json" if artifact_name == "review_decisions.json" else f"artifacts/{artifact_name}")
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact 不存在。")
    media = "text/markdown; charset=utf-8" if path.suffix == ".md" else "application/json; charset=utf-8" if path.suffix == ".json" else "text/html; charset=utf-8"
    return Response(path.read_text(encoding="utf-8"), media_type=media)


@app.get("/runs/{run_id}/rule-validation", response_class=HTMLResponse)
def rule_validation_page(run_id: str) -> str:
    data = _load_artifact(run_id, "rule_validation.json")
    rows = "".join(f"<tr><td><code>{html.escape(str(r.get('rule_id','')))}</code><br>{html.escape(str(r.get('name','')))}</td><td>{html.escape(str(r.get('status','')))}</td><td>{html.escape(str(r.get('recommendation','')))}</td></tr>" for r in data.get("rules") or [])
    body = _metrics([("规则数", data.get("rule_count", 0)), ("可执行", data.get("ready_count", 0)), ("需复核", data.get("needs_review_count", 0))]) + f"<section class='panel'><h2>规则库校验</h2><div class='table-wrap'><table><tr><th>规则</th><th>状态</th><th>建议</th></tr>{rows}</table></div></section>"
    return _page("规则库校验", body, _artifact_links(run_id, ["rule_validation.json", "rule_validation.md"]))


@app.get("/runs/{run_id}/schema-review", response_class=HTMLResponse)
def schema_review_page(run_id: str) -> str:
    data = _load_artifact(run_id, "schema_review.json")
    rows = "".join(f"<tr><td>{html.escape(str(i.get('type','')))}</td><td><code>{html.escape(str(i.get('field','')))}</code></td><td>{html.escape(str(i.get('message','')))}</td></tr>" for i in data.get("review_items") or [])
    body = _metrics([("字段数", data.get("field_count", 0)), ("忽略字段", data.get("ignored_count", 0)), ("规则绑定", data.get("rule_bound_count", 0))]) + f"<section class='panel'><h2>Schema 复核</h2><div class='table-wrap'><table><tr><th>类型</th><th>字段</th><th>说明</th></tr>{rows}</table></div></section>"
    return _page("Schema 复核", body, _artifact_links(run_id, ["schema_review.json", "schema_review.md"]))


def _decision_forms(run_id: str, rule_field: str, dataset_field: str, semantic_role: str = "", display_name: str = "", unit: str = "unknown", statistic: str = "raw", return_to: str = "schema_mapping") -> str:
    if not rule_field or not dataset_field:
        return ""
    hidden = {
        "rule_field": rule_field,
        "dataset_field": dataset_field,
        "semantic_role": semantic_role,
        "display_name": display_name,
        "unit": unit,
        "statistic": statistic,
        "return_to": return_to,
    }
    inputs = "".join(f'<input type="hidden" name="{html.escape(k)}" value="{html.escape(str(v), quote=True)}">' for k, v in hidden.items())
    url = f"/runs/{html.escape(run_id)}/schema-mapping/decisions"
    return (
        f'<div class="decision-line"><form class="decision-form" method="post" action="{url}">{inputs}<input type="hidden" name="action" value="confirm"><input type="text" name="note" placeholder="确认备注"><button class="mini-button confirm" type="submit">确认</button></form>'
        f'<form class="decision-form" method="post" action="{url}">{inputs}<input type="hidden" name="action" value="reject"><input type="text" name="note" placeholder="驳回原因"><button class="mini-button reject" type="submit">驳回</button></form></div>'
    )


def _decisions_panel(run_id: str) -> str:
    decisions = load_review_decisions(store.get_run_dir(run_id))
    items = [item for item in decisions.get("decisions") or [] if isinstance(item, dict)]
    if not items:
        return "<section class='panel'><h2>人工复核意见</h2><p class='muted'>当前 run 还没有人工确认或驳回记录。确认后的意见只影响本次运行刷新；多数据集验证稳定后再推广到全局配置。</p></section>"
    rows = "".join(f"<tr><td>{'确认' if i.get('action') == 'confirm' else '驳回'}</td><td><code>{html.escape(str(i.get('rule_field','')))}</code> -> <code>{html.escape(str(i.get('dataset_field','')))}</code></td><td>{html.escape(str(i.get('note','')))}</td><td>{html.escape(str(i.get('created_at','')))}</td></tr>" for i in items[::-1])
    return f"<section class='panel'><h2>人工复核意见</h2><p class='muted'>这些意见保存在本次运行的 review_decisions.json 中，不自动写入全局配置。</p><div class='table-wrap'><table><tr><th>动作</th><th>映射</th><th>备注</th><th>时间</th></tr>{rows}</table></div></section>"


@app.get("/runs/{run_id}/schema-mapping", response_class=HTMLResponse)
def schema_mapping_page(run_id: str) -> str:
    data = _load_artifact(run_id, "schema_mapping.json")
    rows = []
    for item in data.get("rule_fields") or []:
        candidates = []
        for candidate in item.get("candidates") or []:
            field = str(candidate.get("field") or "")
            label = f"{field} score={candidate.get('score')} {candidate.get('match_method','')}"
            candidates.append(html.escape(label) + _decision_forms(run_id, str(item.get("rule_field") or ""), field, str(candidate.get("semantic_role") or item.get("semantic_role") or ""), str(candidate.get("display_name") or item.get("display_name") or ""), str(candidate.get("unit") or item.get("unit") or "unknown"), str(candidate.get("statistic") or "raw")))
        rows.append(f"<tr><td><code>{html.escape(str(item.get('rule_field','')))}</code><br>{html.escape(str(item.get('display_name','')))}</td><td>{html.escape(str(item.get('status','')))}</td><td>{'<br>'.join(candidates) or html.escape(str(item.get('selected_field') or '无候选'))}</td></tr>")
    body = _metrics([("字段数", data.get("field_count", 0)), ("规则字段", data.get("required_field_count", 0)), ("候选映射", data.get("candidate_mapping_count", 0))]) + _decisions_panel(run_id) + f"<section class='panel'><h2>Schema 语义映射</h2><div class='table-wrap'><table><tr><th>规则字段</th><th>状态</th><th>候选数据字段</th></tr>{''.join(rows)}</table></div></section>"
    return _page("Schema 语义映射", body, _artifact_links(run_id, ["schema_mapping.json", "schema_mapping.md", "schema_suggestions.yaml", "review_decisions.json"]))


@app.get("/runs/{run_id}/schema-mapping-review", response_class=HTMLResponse)
def schema_mapping_review_page(run_id: str) -> str:
    data = _load_artifact(run_id, "schema_mapping_review.json")
    rows = []
    for item in data.get("review_items") or []:
        rule_field = str(item.get("rule_field") or "")
        dataset_field = str(item.get("dataset_field") or "")
        forms = _decision_forms(run_id, rule_field, dataset_field, str(item.get("semantic_role") or ""), str(item.get("rule_name") or ""), str(item.get("candidate_unit") or item.get("rule_unit") or "unknown"), str(item.get("candidate_statistic") or "raw"), "schema_mapping_review") if dataset_field else "<span class='muted'>无候选字段</span>"
        rows.append(f"<tr><td><code>{html.escape(str(item.get('rule_id','')))}</code><br>{html.escape(str(item.get('rule_name','')))}</td><td><span class='badge warn'>{html.escape(_mapping_review_label(str(item.get('status',''))))}</span></td><td><code>{html.escape(rule_field)} -> {html.escape(dataset_field or '无候选')}</code></td><td>{html.escape('；'.join(str(x) for x in item.get('risk_flags') or []))}</td><td>{html.escape(str(item.get('recommendation','')))}</td><td>{forms}</td></tr>")
    body = _metrics([("复核状态", _mapping_review_label(str(data.get("status") or ""))), ("可考虑确认", data.get("confirm_suggestion_count", 0)), ("需专家复核", data.get("manual_review_count", 0)), ("不适用", data.get("not_applicable_count", 0))]) + f"<section class='panel'><h2>复核边界</h2><p class='muted'>{html.escape(str(data.get('anti_overfit_principle') or ''))}</p></section>" + _decisions_panel(run_id) + f"<section class='panel'><h2>映射候选复核辅助</h2><div class='table-wrap'><table><tr><th>规则</th><th>复核建议</th><th>候选映射</th><th>风险标签</th><th>建议动作</th><th>人工意见</th></tr>{''.join(rows)}</table></div></section>"
    return _page("映射候选复核辅助", body, _artifact_links(run_id, ["schema_mapping_review.json", "schema_mapping_review.md", "schema_mapping.json", "review_decisions.json"]))


def _mapping_review_label(status: str) -> str:
    return {
        "needs_human_review": "需人工复核",
        "manual_review_required": "需专家复核",
        "suggest_confirm_candidate": "可考虑确认",
        "not_applicable": "不适用",
        "already_safe": "已安全处理",
    }.get(status, status or "未知")


@app.post("/runs/{run_id}/schema-mapping/decisions")
def submit_schema_mapping_decision(
    run_id: str,
    action: str = Form(...),
    rule_field: str = Form(...),
    dataset_field: str = Form(...),
    semantic_role: str = Form(""),
    display_name: str = Form(""),
    unit: str = Form(""),
    statistic: str = Form(""),
    reviewer: str = Form("local_user"),
    note: str = Form(""),
    return_to: str = Form("schema_mapping"),
) -> RedirectResponse:
    run_dir = store.get_run_dir(run_id)
    if not (run_dir / "manifest.json").exists():
        raise HTTPException(status_code=404, detail="运行不存在。")
    decision = append_rule_mapping_decision(
        run_dir,
        action=action,
        rule_field=rule_field,
        dataset_field=dataset_field,
        semantic_role=semantic_role,
        display_name=display_name,
        unit=unit,
        statistic=statistic,
        reviewer=reviewer,
        note=note,
    )
    _refresh_review_artifacts(run_id, decision)
    target = "schema-mapping-review" if return_to == "schema_mapping_review" else "schema-mapping"
    return RedirectResponse(f"/runs/{run_id}/{target}", status_code=303)


def _refresh_review_artifacts(run_id: str, decision: dict[str, Any]) -> None:
    context = _rehydrate_run_context(run_id)
    executor = ToolExecutor(context)
    context.tool_results = _read_tool_results_from_events(run_id)
    for name, arguments in [
        ("profile_dataset", {}),
        ("parse_rule_library", {}),
        ("suggest_schema_metadata", {}),
        ("review_schema_metadata", {}),
        ("review_schema_mapping", {}),
        ("review_dataset_adaptation", {}),
        ("review_schema_mapping_candidates", {}),
        ("build_mapping_execution_plan", {}),
        ("validate_mapping_regression", {}),
        ("apply_rules", {}),
        ("review_generalization_readiness", {}),
        ("validate_analysis", {}),
        ("build_visualizations", {"max_items": 8}),
    ]:
        result = executor.execute(name, arguments, record=False)
        context.tool_results.append({"tool": name, "arguments": arguments, "result": result})
    report = executor.execute("write_report", {"final_summary": "人工复核意见已接入本次运行，相关产物已刷新。"}, record=False)
    context.tool_results.append({"tool": "write_report", "arguments": {}, "result": report})
    context.status = "complete"
    context.save_manifest()


def _read_tool_results_from_events(run_id: str) -> list[dict[str, Any]]:
    path = store.get_run_dir(run_id) / "run_events.jsonl"
    results = []
    if not path.exists():
        return results
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload") or {}
        if event.get("type") == "tool_finished" and payload.get("tool"):
            results.append({"tool": payload.get("tool"), "arguments": payload.get("arguments") or {}, "result": payload.get("result")})
    return results


def _generic_artifact_page(run_id: str, artifact_name: str, title: str, keywords: list[str]) -> str:
    data = _load_artifact(run_id, artifact_name)
    intro = "".join(f"<p>{html.escape(word)}</p>" for word in keywords)
    return _page(title, f"<section class='panel'><h2>{html.escape(title)}</h2>{intro}{_json_block(data)}</section>", _artifact_links(run_id, [artifact_name, artifact_name.replace('.json', '.md')]))


@app.get("/runs/{run_id}/dataset-adaptation", response_class=HTMLResponse)
def dataset_adaptation_page(run_id: str) -> str:
    return _generic_artifact_page(run_id, "dataset_adaptation.json", "数据适配审查", ["需映射确认"])


@app.get("/runs/{run_id}/mapping-execution-plan", response_class=HTMLResponse)
def mapping_execution_page(run_id: str) -> str:
    return _generic_artifact_page(run_id, "mapping_execution_plan.json", "映射执行计划", ["字段重写预案"])


@app.get("/runs/{run_id}/mapping-regression", response_class=HTMLResponse)
def mapping_regression_page(run_id: str) -> str:
    return _generic_artifact_page(run_id, "mapping_regression_validation.json", "映射回归验证", ["需映射确认"])


@app.get("/runs/{run_id}/generalization-readiness", response_class=HTMLResponse)
def generalization_page(run_id: str) -> str:
    return _generic_artifact_page(run_id, "generalization_readiness.json", "泛化验收门禁", ["候选映射保持人工确认状态"])


@app.get("/toolbox-ui", response_class=HTMLResponse)
def toolbox_ui() -> str:
    rows = "".join(f"<tr><td><code>{html.escape(str(tool.get('name','')))}</code></td><td>{html.escape(str(tool.get('label','')))}</td><td>{html.escape(str(tool.get('category','')))}</td></tr>" for tool in TOOLBOX_CATALOG)
    body = f"<section class='panel'><h2>分析能力中心</h2><div class='table-wrap'><table><tr><th>工具</th><th>名称</th><th>分类</th></tr>{rows}</table></div></section>"
    return _page("分析能力中心", body)


@app.get("/docs/mvp", response_class=HTMLResponse)
def docs_mvp() -> str:
    path = PROJECT_ROOT / "docs" / "MVP验收说明.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# Agent 工具箱 MVP 验收说明\n"
    body = f"<section class='panel'><h2>MVP 交付说明</h2><p><a href='/docs/mvp.md'>/docs/mvp.md</a></p></section>{markdown_to_html(text)}"
    return _page("MVP 交付说明", body)


@app.get("/docs/mvp.md")
def docs_mvp_markdown() -> Response:
    path = PROJECT_ROOT / "docs" / "MVP验收说明.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# Agent 工具箱 MVP 验收说明\n"
    return Response(text, media_type="text/markdown; charset=utf-8")


@app.get("/batch-ui", response_class=HTMLResponse)
def batch_ui() -> str:
    options = "".join(f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name in named_batch_dataset_dirs())
    body = f"""
    <section class='panel'><h2>批量验收</h2>
    <form action="/batches" method="post">
      <label>数据集族<select name="dataset_family"><option value="last_avg">last_avg</option><option value="customer_variants">customer_variants</option>{options}</select></label>
      <label>自定义数据集目录<input type="text" name="dataset_dir" value=""></label>
      <label>最大数据集数<input type="text" name="max_datasets" value=""></label>
      <button class="primary" type="submit">启动批量验收</button>
    </form></section>
    """
    return _page("批量验收", body)


@app.post("/batches")
def create_batch(background_tasks: BackgroundTasks, dataset_family: str = Form("last_avg"), dataset_dir: str = Form(""), max_datasets: str = Form("")) -> RedirectResponse:
    dirs = named_batch_dataset_dirs()
    selected_dir = Path(dataset_dir).expanduser() if dataset_dir.strip() else dirs.get(dataset_family, default_batch_dataset_dir())
    max_count = int(max_datasets) if str(max_datasets).strip().isdigit() else None
    manifest = create_batch_job(settings, rule_path=DEMO_RULE_PATH, dataset_dir=selected_dir, max_datasets=max_count)
    background_tasks.add_task(run_batch_job, settings, manifest["batch_id"])
    return RedirectResponse(f"/batches/{manifest['batch_id']}", status_code=303)


@app.get("/api/batches")
def batches_json() -> JSONResponse:
    return JSONResponse({"batches": list_batch_manifests(settings.storage_dir)})


@app.get("/api/batches/{batch_id}")
def batch_json(batch_id: str) -> JSONResponse:
    try:
        return JSONResponse(get_batch_manifest(settings.storage_dir, batch_id))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="批量验收记录不存在。") from None


def _batch_actions(batch_id: str) -> str:
    links = [
        ("批量汇总报告", f"/batches/{batch_id}/report"),
        ("数据适配", f"/batches/{batch_id}/dataset-adaptation"),
        ("映射复核辅助", f"/batches/{batch_id}/schema-mapping-review"),
        ("人工复核意见", f"/batches/{batch_id}/review-decisions"),
        ("映射回归", f"/batches/{batch_id}/mapping-regression"),
        ("泛化门禁", f"/batches/{batch_id}/generalization-readiness"),
        ("规则稳定性", f"/batches/{batch_id}/rule-analysis"),
        ("命中样例", f"/batches/{batch_id}/hit-samples"),
        ("JSON", f"/api/batches/{batch_id}"),
    ]
    return "<div class='actions'>" + "".join(f"<a class='button-link secondary' href='{href}'>{html.escape(label)}</a>" for label, href in links) + "</div>"


@app.get("/batches/{batch_id}", response_class=HTMLResponse)
def batch_detail(batch_id: str) -> str:
    manifest = get_batch_manifest(settings.storage_dir, batch_id)
    body = _batch_actions(batch_id) + f"<section class='panel'><h2>批次详情</h2>{_json_block(manifest)}</section>"
    return _page("批次详情", body)


def _batch_artifact_page(batch_id: str, build_fn, html_key: str, missing: str) -> str:
    try:
        build_fn(settings.storage_dir, batch_id)
        manifest = get_batch_manifest(settings.storage_dir, batch_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="批量验收记录不存在。") from None
    path = Path(str(manifest.get(html_key) or ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail=missing)
    return path.read_text(encoding="utf-8")


@app.get("/batches/{batch_id}/report", response_class=HTMLResponse)
def batch_report(batch_id: str) -> str:
    manifest = get_batch_manifest(settings.storage_dir, batch_id)
    path = Path(str(manifest.get("report_html_path") or ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail="批量报告不存在。")
    return path.read_text(encoding="utf-8")


@app.get("/batches/{batch_id}/rule-analysis", response_class=HTMLResponse)
def batch_rule_analysis_page(batch_id: str) -> str:
    return _batch_artifact_page(batch_id, build_batch_rule_analysis, "rule_analysis_html_path", "规则稳定性分析不存在。")


@app.get("/batches/{batch_id}/dataset-adaptation", response_class=HTMLResponse)
def batch_dataset_adaptation_page(batch_id: str) -> str:
    return _batch_artifact_page(batch_id, build_batch_dataset_adaptation, "dataset_adaptation_html_path", "批量数据适配审查不存在。")


@app.get("/batches/{batch_id}/schema-mapping-review", response_class=HTMLResponse)
def batch_schema_mapping_review_page(batch_id: str) -> str:
    return _batch_artifact_page(batch_id, build_batch_schema_mapping_review, "schema_mapping_review_html_path", "批量映射复核辅助不存在。")


@app.get("/batches/{batch_id}/review-decisions", response_class=HTMLResponse)
def batch_review_decisions_page(batch_id: str) -> str:
    return _batch_artifact_page(batch_id, build_batch_review_decisions, "review_decisions_html_path", "批量人工复核意见不存在。")


@app.get("/batches/{batch_id}/mapping-regression", response_class=HTMLResponse)
def batch_mapping_regression_page(batch_id: str) -> str:
    return _batch_artifact_page(batch_id, build_batch_mapping_regression, "mapping_regression_html_path", "批量映射回归验证不存在。")


@app.get("/batches/{batch_id}/generalization-readiness", response_class=HTMLResponse)
def batch_generalization_page(batch_id: str) -> str:
    return _batch_artifact_page(batch_id, build_batch_generalization_readiness, "generalization_readiness_html_path", "批量泛化门禁不存在。")


@app.get("/batches/{batch_id}/hit-samples", response_class=HTMLResponse)
def batch_hit_samples_page(batch_id: str) -> str:
    return _batch_artifact_page(batch_id, build_batch_hit_samples, "key_hit_samples_html_path", "关键命中样例不存在。")


@app.get("/api/batches/{batch_id}/rule-analysis")
def batch_rule_analysis_json(batch_id: str) -> JSONResponse:
    return JSONResponse(build_batch_rule_analysis(settings.storage_dir, batch_id))


@app.get("/api/batches/{batch_id}/dataset-adaptation")
def batch_dataset_adaptation_json(batch_id: str) -> JSONResponse:
    return JSONResponse(build_batch_dataset_adaptation(settings.storage_dir, batch_id))


@app.get("/api/batches/{batch_id}/schema-mapping-review")
def batch_schema_mapping_review_json(batch_id: str) -> JSONResponse:
    return JSONResponse(build_batch_schema_mapping_review(settings.storage_dir, batch_id))


@app.get("/api/batches/{batch_id}/review-decisions")
def batch_review_decisions_json(batch_id: str) -> JSONResponse:
    return JSONResponse(build_batch_review_decisions(settings.storage_dir, batch_id))


@app.get("/api/batches/{batch_id}/mapping-regression")
def batch_mapping_regression_json(batch_id: str) -> JSONResponse:
    return JSONResponse(build_batch_mapping_regression(settings.storage_dir, batch_id))


@app.get("/api/batches/{batch_id}/generalization-readiness")
def batch_generalization_json(batch_id: str) -> JSONResponse:
    return JSONResponse(build_batch_generalization_readiness(settings.storage_dir, batch_id))


@app.get("/api/batches/{batch_id}/hit-samples")
def batch_hit_samples_json(batch_id: str) -> JSONResponse:
    return JSONResponse(build_batch_hit_samples(settings.storage_dir, batch_id))


@app.get("/batches/{batch_id}/{artifact}.md")
def batch_markdown(batch_id: str, artifact: str) -> Response:
    mapping = {
        "schema-mapping-review": (build_batch_schema_mapping_review, "schema_mapping_review_markdown_path"),
        "review-decisions": (build_batch_review_decisions, "review_decisions_markdown_path"),
        "mapping-regression": (build_batch_mapping_regression, "mapping_regression_markdown_path"),
        "generalization-readiness": (build_batch_generalization_readiness, "generalization_readiness_markdown_path"),
        "dataset-adaptation": (build_batch_dataset_adaptation, "dataset_adaptation_markdown_path"),
        "hit-samples": (build_batch_hit_samples, "key_hit_samples_markdown_path"),
    }
    if artifact not in mapping:
        raise HTTPException(status_code=404, detail="批量 Markdown 不存在。")
    build_fn, key = mapping[artifact]
    build_fn(settings.storage_dir, batch_id)
    manifest = get_batch_manifest(settings.storage_dir, batch_id)
    path = Path(str(manifest.get(key) or ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail="批量 Markdown 不存在。")
    return Response(path.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")


@app.get("/review-promotions", response_class=HTMLResponse)
def review_promotions_page() -> str:
    report = build_review_promotion_report(settings.storage_dir)
    markdown = render_review_promotion_markdown(report)
    body = "<section class='panel'><h2>复核意见沉淀</h2><p>可进入全局审核的候选会在这里显示；系统不会自动写入全局配置。</p></section>" + markdown_to_html(markdown)
    return _page("复核意见沉淀", body)


@app.get("/api/review-promotions")
def review_promotions_json() -> JSONResponse:
    return JSONResponse(build_review_promotion_report(settings.storage_dir))


@app.get("/rule-versions", response_class=HTMLResponse)
def rule_versions_page(candidate_batch_id: str = "") -> str:
    body = f"<section class='panel'><h2>规则优化对比</h2><p class='muted'>候选批次：{html.escape(candidate_batch_id)}</p></section>"
    return _page("规则优化对比", body)


@app.get("/rule-versions/compare", response_class=HTMLResponse)
def rule_versions_compare(base_batch_id: str, candidate_batch_id: str) -> str:
    comparison = compare_batch_rule_versions(settings.storage_dir, base_batch_id, candidate_batch_id)
    return _page("规则版本对比", f"<section class='panel'>{_json_block(comparison)}</section>")
