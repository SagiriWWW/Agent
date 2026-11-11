"""Read-only, human-readable presentation of the existing run/event protocols."""

import html
import json
from typing import Any


STATUS_LABELS = {
    "created": "等待开始", "running": "正在运行",
    "complete": "运行已完成", "failed": "运行失败",
}

STYLE = """
<style>
.progress-view{--accent:#087f72;--line:#dce3e8;--muted:#596b78;--danger:#b42318;min-width:0}
.progress-view *{box-sizing:border-box;letter-spacing:0}
.progress-view [hidden]{display:none!important}
.progress-overview{padding:24px 0;border-bottom:1px solid var(--line)}
.progress-heading,.progress-section-heading{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
.progress-heading h2{font-size:23px;line-height:1.35;margin:0}
.progress-kicker{color:var(--muted);font-size:13px;margin:0 0 8px}
.progress-state{display:inline-flex;align-items:center;gap:8px;font-size:14px;font-weight:700;color:var(--accent)}
.progress-state::before{content:'';width:8px;height:8px;border-radius:50%;background:currentColor;flex:none}
.progress-state[data-status='failed']{color:var(--danger)}
.progress-state[data-status='created']{color:var(--muted)}
.progress-state[data-status='warning']{color:#926000}
.progress-goal{margin:14px 0 18px;overflow-wrap:anywhere;color:#344655;max-width:90ch}
.progress-facts{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:20px;margin:20px 0}
.progress-facts dt{font-size:13px;color:var(--muted);margin-bottom:4px}
.progress-facts dd{font-size:18px;font-weight:650;margin:0;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}
.progress-meter{width:100%;height:7px;display:block;accent-color:var(--accent)}
.progress-meter-note{display:flex;justify-content:space-between;gap:12px;font-size:13px;color:var(--muted);margin:8px 0 0}
.progress-actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.progress-actions .primary{background:var(--accent);border-color:var(--accent);color:white}
.progress-icon-button{width:38px;height:38px;min-width:38px;padding:0;font-size:23px}
.progress-view button:focus-visible,.progress-view a:focus-visible,.progress-view summary:focus-visible{outline:3px solid #138dce;outline-offset:3px}
.progress-notice{border-left:3px solid #d19a22;background:#fff8e8;padding:12px 16px;margin:16px 0;overflow-wrap:anywhere}
.progress-notice[data-level='error']{border-color:var(--danger);background:#fff1f0;color:#8f231a}
.progress-columns{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(280px,1fr);gap:36px;padding-top:28px}
.progress-columns>section{min-width:0}
.progress-section-heading{padding-bottom:14px;border-bottom:1px solid var(--line)}
.progress-section-heading h2{font-size:18px;margin:0}
.progress-section-heading span{color:var(--muted);font-size:13px}
.progress-steps,.progress-events{list-style:none;margin:0;padding:0}
.progress-step{display:grid;grid-template-columns:30px minmax(0,1fr);gap:12px;padding:16px 0;border-bottom:1px solid #e8edf0}
.progress-step-mark{width:28px;height:28px;border:1px solid #ccd7df;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:var(--muted)}
.progress-step[data-status='complete'] .progress-step-mark{background:#e7f5ee;border-color:#a8d8c3;color:#136c48}
.progress-step[data-status='running'] .progress-step-mark{background:#e8f7f5;border-color:#71beb5;color:var(--accent)}
.progress-step[data-status='failed'] .progress-step-mark{background:#ffefed;border-color:#e8b0a9;color:var(--danger)}
.progress-step-top{display:flex;align-items:baseline;justify-content:space-between;gap:12px}
.progress-step-name{font-size:14px;font-weight:700;overflow-wrap:anywhere}
.progress-step-state{font-size:12px;white-space:nowrap;flex:none;color:var(--muted)}
.progress-step[data-status='failed'] .progress-step-state{color:var(--danger)}
.progress-step-note{color:#526573;margin:5px 0 0;font-size:13px;overflow-wrap:anywhere;white-space:pre-wrap}
.progress-detail{margin-top:8px;font-size:13px;overflow-wrap:anywhere}
.progress-detail summary{cursor:pointer;color:#2563a8;width:fit-content;font-size:12px}
.progress-detail dl{margin:8px 0;background:#f4f7f9;padding:12px;border-radius:4px}
.progress-detail dt{font-weight:650;font-size:12px;color:var(--muted);margin-top:8px}
.progress-detail dt:first-child{margin-top:0}
.progress-detail dd{font-size:13px;margin:3px 0 0;white-space:pre-wrap;overflow-wrap:anywhere}
.progress-event{padding:14px 0;border-bottom:1px solid #e8edf0;min-width:0}
.progress-event-top{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.progress-event-title{font-size:13px;font-weight:650;overflow-wrap:anywhere}
.progress-event time{font-size:12px;color:var(--muted);flex:none;font-variant-numeric:tabular-nums}
.progress-event[data-error='true'] .progress-event-title{color:var(--danger)}
.progress-event p{font-size:13px;color:var(--muted);margin:5px 0 0;overflow-wrap:anywhere;white-space:pre-wrap}
.progress-view pre{font-size:12px;line-height:1.6;background:#f3f6f8;color:#263b49;border:1px solid var(--line);max-height:320px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;padding:12px}
.progress-empty{color:var(--muted);font-size:14px;padding:24px 0}
.progress-more{margin-top:14px;font-size:13px}
.progress-footer{margin-top:28px;padding:18px 0;border-top:1px solid var(--line);display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap;color:var(--muted);font-size:12px}
.progress-run-id{overflow-wrap:anywhere;min-width:0}
.progress-footer a{font-size:12px}
@media(max-width:760px){
 main:has(.progress-view){padding:16px 16px 32px}
 .progress-columns{grid-template-columns:minmax(0,1fr);gap:28px}
 .progress-facts{grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
 .progress-heading{align-items:flex-start}
 .progress-heading h2{font-size:21px}
 .progress-step{gap:10px}
 .progress-actions{width:100%}
 .progress-overview{padding-top:18px}
}
</style>
"""

SCRIPT = r"""
<script>
(() => {
  const initial = JSON.parse(document.getElementById('run-progress-data').textContent);
  const runId = initial.run_id;
  const root = document.getElementById('run-progress');
  const byId = id => document.getElementById(id);
  const events = new Map();
  const terminal = status => ['complete', 'failed'].includes(status);
  const statusLabels = {created: '等待开始', running: '正在运行', complete: '运行已完成', failed: '运行失败'};
  const stepLabels = {pending: '待执行', running: '执行中', complete: '已完成', failed: '失败'};
  let state = initial, timer, busy = false, expanded = false, stepVersion = '', eventVersion = '';
  let lastUpdated = '', failureCount = 0;
  const list = value => Array.isArray(value) ? value : [];
  const text = value => value == null ? '' : String(value);
  const short = (value, max = 180) => text(value).length > max ? text(value).slice(0, max) + '…' : text(value);
  function node(tag, className, content) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (content != null) element.textContent = content;
    return element;
  }
  function clock(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? '时间未知' : date.toLocaleTimeString('zh-CN', {hour12: false});
  }
  function detail(values, title = '查看详情') {
    const element = node('details', 'progress-detail');
    element.append(node('summary', '', title));
    const dl = node('dl');
    for (const [label, value] of values) {
      if (value == null || value === '') continue;
      dl.append(node('dt', '', label), node('dd', '', typeof value === 'object' ? JSON.stringify(value, null, 2) : text(value)));
    }
    element.append(dl);
    return element;
  }
  function getSteps() {
    const steps = new Map();
    for (const item of list(state.agent_plan)) {
      if (item && Number.isInteger(item.step) && item.step > 0) steps.set(item.step, {...item, status: 'pending'});
    }
    for (const event of events.values()) {
      const payload = event.payload || {};
      if (!Number.isInteger(payload.step) || payload.step <= 0) continue;
      if (!['step_started', 'step_finished', 'step_failed'].includes(event.type)) continue;
      const status = event.type === 'step_started' ? 'running' : event.type === 'step_failed' ? 'failed' : 'complete';
      steps.set(payload.step, {...steps.get(payload.step), ...payload, status, timestamp: event.timestamp});
    }
    for (const item of list(state.agent_trace)) {
      if (item && Number.isInteger(item.step) && item.step > 0) steps.set(item.step, {...steps.get(item.step), ...item});
    }
    return [...steps.values()].sort((a, b) => a.step - b.step);
  }
  function render() {
    const steps = getSteps();
    const complete = steps.filter(item => item.status === 'complete').length;
    const failed = steps.filter(item => item.status === 'failed').length;
    const finished = complete + failed;
    const active = steps.find(item => item.status === 'running');
    const labels = new Map(list(state.toolbox_manifest).filter(Boolean).map(item => [item.name, item.label]));
    const label = item => labels.get(item.action) || item.action || `步骤 ${item.step}`;
    const status = state.status || 'created';
    byId('progress-status').textContent = status === 'complete' && failed ? '已结束（有失败步骤）' : statusLabels[status] || '状态未知';
    byId('progress-status').dataset.status = status === 'complete' && failed ? 'warning' : status;
    root.dataset.status = status;
    byId('progress-count').textContent = steps.length ? `${finished} / ${steps.length}` : '尚未生成计划';
    byId('progress-failures').textContent = text(failed);
    byId('progress-current').textContent = terminal(status) ? '已结束' : active ? label(active) : status === 'created' ? '等待开始' : '等待下一步骤';
    byId('progress-mode').textContent = state.model_config?.mode_label || '未记录';
    byId('progress-meter').max = steps.length || 1;
    byId('progress-meter').value = finished;
    byId('progress-meter').setAttribute('aria-valuetext', `${finished} 个步骤已结束，共 ${steps.length} 个已知步骤`);
    byId('progress-step-note').textContent = `完成 ${complete} 步${failed ? ` · 失败 ${failed} 步` : ''}`;
    byId('progress-step-total').textContent = `${steps.length} 个步骤`;
    byId('progress-error').hidden = !state.error;
    byId('progress-error').textContent = state.error ? '运行失败：' + text(state.error) : '';
    byId('progress-report').hidden = !state.report_html_path;
    byId('progress-report').href = `/runs/${encodeURIComponent(runId)}`;
    const version = JSON.stringify(steps);
    if (version !== stepVersion) {
      stepVersion = version;
      const open = new Set([...byId('progress-steps').querySelectorAll('details[open]')].map(item => item.dataset.step));
      const rows = steps.map(item => {
        const row = node('li', 'progress-step');
        row.dataset.status = item.status;
        row.append(node('span', 'progress-step-mark', item.status === 'complete' ? '✓' : item.status === 'failed' ? '×' : text(item.step)));
        const body = node('div');
        const top = node('div', 'progress-step-top');
        top.append(node('span', 'progress-step-name', label(item)), node('span', 'progress-step-state', stepLabels[item.status] || '状态未知'));
        body.append(top);
        const note = item.observation || item.thought;
        if (note) body.append(node('p', 'progress-step-note', short(note)));
        const details = detail([['步骤', item.step], ['工具', item.action], ['执行目的', item.thought], ['执行结果', item.observation], ['输入参数', item.arguments]]);
        details.dataset.step = item.step;
        details.open = open.has(text(item.step));
        body.append(details); row.append(body); return row;
      });
      byId('progress-steps').replaceChildren(...rows);
    }
    byId('progress-no-steps').hidden = steps.length > 0;
    byId('progress-no-steps').textContent = terminal(status) ? '此运行没有保存步骤记录。' : '等待执行计划…';
    byId('progress-sync').textContent = lastUpdated ? `更新于 ${lastUpdated}${terminal(status) ? ' · 运行已结束' : ''}` : '正在读取运行状态…';
    renderEvents();
  }
  function renderEvents() {
    const all = [...events.values()].reverse();
    byId('progress-event-count').textContent = `${all.length} 条`;
    byId('progress-no-events').hidden = all.length > 0;
    const visible = expanded ? all : all.slice(0, 12);
    const version = JSON.stringify([expanded, visible]);
    if (version !== eventVersion) {
      eventVersion = version;
      const open = new Set([...byId('progress-events').querySelectorAll('details[open]')].map(item => item.dataset.event));
      byId('progress-events').replaceChildren(...visible.map(event => {
        const row = node('li', 'progress-event');
        row.dataset.eventId = text(event.id);
        row.dataset.error = text(event.type === 'run_failed' || event.type === 'step_failed');
        const top = node('div', 'progress-event-top');
        const time = node('time', '', clock(event.timestamp)); time.dateTime = text(event.timestamp);
        top.append(node('span', 'progress-event-title', event.title || event.type || '运行事件'), time);
        row.append(top);
        if (event.message && event.message !== event.title) row.append(node('p', '', short(event.message)));
        const details = node('details', 'progress-detail');
        details.dataset.event = text(event.id);
        details.open = open.has(text(event.id));
        details.append(node('summary', '', '原始记录'), node('pre', '', JSON.stringify(event, null, 2)));
        row.append(details); return row;
      }));
    }
    byId('progress-more').hidden = all.length <= 12;
    byId('progress-more').textContent = expanded ? '收起活动记录' : `显示全部 ${all.length} 条记录`;
  }
  function readEvents(source) {
    // The SSE endpoint returns a finite snapshot. Poll it without EventSource's
    // automatic reconnect, and deduplicate replayed events by their persisted ID.
    let malformed = 0;
    for (const block of source.replace(/\r\n/g, '\n').split('\n\n')) {
      const data = block.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
      if (!data) continue;
      try {
        const event = JSON.parse(data);
        if (!event || typeof event !== 'object' || Array.isArray(event) || !Number.isInteger(event.id) || event.id <= 0) throw new Error('invalid event');
        events.set(event.id, event);
      } catch (_) { malformed += 1; }
    }
    return malformed;
  }
  async function response(url) {
    const result = await fetch(url, {cache: 'no-store', signal: AbortSignal.timeout(10000)});
    if (!result.ok) throw new Error(result.status === 404 ? '运行记录不存在或不可访问' : `读取失败（HTTP ${result.status}）`);
    return result;
  }
  async function refresh() {
    if (busy) return;
    busy = true; clearTimeout(timer); byId('progress-refresh').disabled = true;
    const results = await Promise.allSettled([
      response(`/api/runs/${runId}`).then(result => result.json()),
      response(`/api/runs/${runId}/events`).then(result => result.text())
    ]);
    let error = '';
    if (results[0].status === 'fulfilled') {
      const value = results[0].value;
      if (value && value.run_id === runId && typeof value.status === 'string') state = value;
      else error = '运行状态格式异常';
    } else error = results[0].reason.message || '运行状态暂不可用';
    if (results[1].status === 'fulfilled') {
      if (readEvents(results[1].value)) error = error || '部分活动记录无法读取，已保留可用记录';
    } else error = error || '活动记录暂不可用，已保留上次结果';
    if (!error) { lastUpdated = clock(new Date().toISOString()); failureCount = 0; }
    else failureCount += 1;
    byId('progress-connection').hidden = !error;
    byId('progress-connection').textContent = error ? error + '。可点击刷新重试。' : '';
    render();
    busy = false; byId('progress-refresh').disabled = false;
    if (!terminal(state.status) && !document.hidden && failureCount < 5) timer = setTimeout(refresh, error ? 5000 : 2000);
  }
  byId('progress-refresh').addEventListener('click', refresh);
  byId('progress-more').addEventListener('click', () => { expanded = !expanded; renderEvents(); });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) clearTimeout(timer);
    else if (!terminal(state.status)) refresh();
  });
  window.addEventListener('pagehide', () => clearTimeout(timer));
  render(); refresh();
})();
</script>
"""


def progress_page_body(run_id: str, manifest: dict[str, Any]) -> str:
    fields = (
        "status", "error", "user_goal", "agent_plan", "agent_trace", "toolbox_manifest",
        "model_config", "created_at", "updated_at", "report_html_path",
    )
    state = {key: manifest.get(key) for key in fields}
    state["run_id"] = run_id
    # JSON lives in a script element, so HTML escaping alone would not be safe.
    data = (json.dumps(state, ensure_ascii=True)
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))
    safe_id = html.escape(run_id, quote=True)
    label = STATUS_LABELS.get(str(state["status"]), "状态未知")
    goal = html.escape(str(state["user_goal"] or "未填写分析目标"))
    report_hidden = "" if state["report_html_path"] else " hidden"
    return STYLE + f"""
<div class="progress-view" id="run-progress">
  <section class="progress-overview" aria-label="运行概览">
    <p class="progress-kicker">飞行遥测数据分析</p>
    <div class="progress-heading">
      <div><h2>本次分析</h2><p class="progress-state" id="progress-status" role="status">{label}</p></div>
      <div class="progress-actions">
        <a class="button-link primary" id="progress-report" href="/runs/{safe_id}"{report_hidden}>查看报告</a>
        <a class="button-link secondary" href='/runs/{safe_id}/evidence'>核验证据链</a>
        <button class="progress-icon-button" id="progress-refresh" type="button" title="刷新进度" aria-label="刷新进度">&#8635;</button>
      </div>
    </div>
    <p class="progress-goal">{goal}</p>
    <dl class="progress-facts">
      <div><dt>已结束 / 已知步骤</dt><dd id="progress-count">—</dd></div>
      <div><dt>当前步骤</dt><dd id="progress-current">—</dd></div>
      <div><dt>失败步骤</dt><dd id="progress-failures">—</dd></div>
      <div><dt>运行模式</dt><dd id="progress-mode">—</dd></div>
    </dl>
    <progress class="progress-meter" id="progress-meter" max="1" value="0" aria-label="已结束步骤"></progress>
    <div class="progress-meter-note"><span id="progress-step-note"></span><span id="progress-sync" aria-live="polite"></span></div>
    <div class="progress-notice" id="progress-error" data-level="error" role="alert" hidden></div>
    <div class="progress-notice" id="progress-connection" role="status" hidden></div>
  </section>
  <div class="progress-columns">
    <section aria-labelledby="progress-steps-title">
      <div class="progress-section-heading"><h2 id="progress-steps-title">执行步骤</h2><span id="progress-step-total"></span></div>
      <ol class="progress-steps" id="progress-steps"></ol>
      <p class="progress-empty" id="progress-no-steps">等待执行计划…</p>
    </section>
    <section aria-labelledby="progress-events-title">
      <div class="progress-section-heading"><h2 id="progress-events-title">活动记录</h2><span id="progress-event-count"></span></div>
      <ol class="progress-events" id="progress-events"></ol>
      <p class="progress-empty" id="progress-no-events">暂无活动记录</p>
      <button class="progress-more" id="progress-more" type="button" hidden>显示全部记录</button>
    </section>
  </div>
  <footer class="progress-footer">
    <span class="progress-run-id">运行 ID：{safe_id}</span>
    <a href="/api/runs/{safe_id}">查看运行 JSON</a>
  </footer>
  <noscript><p>浏览器已禁用 JavaScript，可通过运行详情查看结果。</p><a href="/runs/{safe_id}">查看运行详情</a></noscript>
</div>
<script id="run-progress-data" type="application/json">{data}</script>
""" + SCRIPT
