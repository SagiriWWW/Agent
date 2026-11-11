"""Read-only presentation of saved reports. No RunContext, tool execution or writes.

The original HTML is parsed as a tree, retained in expandable source sections,
and supplemented with tables from frozen source calls / completed event records.
"""
from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from aircraft_agent.telemetry.coverage import CLAIMS_FILE, COVERAGE_FILE, SNAPSHOT_FILE
from aircraft_agent.telemetry.schema import label_expression
from aircraft_agent.telemetry.visualizations import _render_chart

TABS = [('overview', '概览'), ('rules', '规则与质量'), ('associations', '关联分析'),
        ('charts', '图表'), ('adaptation', '适配复核'), ('evidence', '证据与过程')]
STATUS = {'evaluated': '已执行', 'passed': '通过', 'pass': '通过', 'ready': '可执行',
          'failed': '失败', 'error': '失败', 'not_run': '未执行', 'skipped': '未执行',
          'unparseable': '无法解析', 'unassessable': '不可评价', 'needs_review': '待复核',
          'blocked': '受阻', 'complete': '已完成', 'running': '运行中', 'created': '已创建',
          'verified': '引用已核验', 'high': '高', 'medium': '中', 'low': '低'}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def records(value: Any) -> list[dict]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def read_saved(root: Path, name: str) -> dict:
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        return {}
    try:
        return mapping(json.loads(path.read_text(encoding='utf-8')))
    except (OSError, ValueError):
        return {}


def state(value: Any) -> str:
    return STATUS.get(str(value), str(value) if value is not None else '缺失')


def number(value: Any, percent: bool = False) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return '不可评价'
    return f'{value:.2%}' if percent else f'{value:,g}'


@dataclass
class Node:
    tag: str = ''
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)

    def text(self):
        return ''.join(c.text() if isinstance(c, Node) else c for c in self.children)

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()

    def render(self):
        if self.tag in {'script', 'style', 'iframe', 'object', 'embed', 'link', 'meta'}:
            return ''
        body = ''.join(c.render() if isinstance(c, Node) else esc(c) for c in self.children)
        if not self.tag or self.tag in {'html', 'body', 'head'}:
            return body
        # Saved text and anchors survive; no executable legacy attributes or CSS.
        attrs = {k: v for k, v in self.attrs.items() if k in {'id', 'title', 'colspan', 'rowspan'}}
        if self.tag == 'a':
            href = str(self.attrs.get('href', ''))
            parts = urlsplit(href)
            if not parts.scheme and not parts.netloc:
                attrs['href'] = href
        attributes = ''.join(f' {k}="{esc(v)}"' for k, v in attrs.items())
        markup = f'<{self.tag}{attributes}>{body}</{self.tag}>'
        if self.tag == 'table':
            return '<div class="table-scroll" data-legacy-table tabindex="0" role="region" aria-label="原报告表格，可横向滚动">' + markup + '</div>'
        return markup


class SavedHTML(HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        node = Node(tag, dict(attrs))
        self.stack[-1].children.append(node)
        if tag not in {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'wbr'}:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                self.stack = self.stack[:i]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def group(title: str) -> str:
    if any(s in title for s in ['可视化', '图表']):
        return 'charts'
    if any(s in title for s in ['适配', '映射', '泛化', 'Schema', '语义']):
        return 'adaptation'
    if any(s in title for s in ['关联', '相关性']):
        return 'associations'
    if any(s in title for s in ['规则', '质量', '缺失', '离群']):
        return 'rules'
    return 'evidence'


def legacy_sections(source: str) -> dict[str, str]:
    tree = SavedHTML(source).root
    body = next((n for n in tree.walk() if n.tag == 'body'), tree)
    result = {key: [] for key, _ in TABS}

    def collect(node):
        heading = next((n for n in node.walk() if n.tag == 'h2'), None)
        if node.tag == 'section' and heading is not None:
            title = heading.text()
            result[group(title)].append(f'<details class="legacy"><summary>原报告 · {esc(title)}</summary>{node.render()}</details>')
        elif any(n.tag == 'h2' for n in node.walk()):
            for c in node.children:
                if isinstance(c, Node):
                    collect(c)
                elif c.strip():
                    result['evidence'].append(f'<p>{esc(c)}</p>')
        else:
            markup = node.render()
            if markup.strip():
                result['evidence'].append(f'<details class="legacy"><summary>原报告 · 补充信息</summary>{markup}</details>')
    for node in body.children:
        if isinstance(node, Node):
            collect(node)
        elif node.strip():
            result['evidence'].append(f'<p>{esc(node)}</p>')
    return {k: ''.join(v) for k, v in result.items()}


def json_detail(label: str, data: Any, anchor: str = '') -> str:
    attr = f' id="{esc(anchor)}"' if anchor else ''
    return f'<details{attr}><summary>{esc(label)}</summary><pre>{esc(json.dumps(data, ensure_ascii=False, indent=2))}</pre></details>'


def cell(value: Any, percent=False, numeric=False):
    raw = '' if value is None else value
    display = number(value, percent) if numeric or percent else ('缺失' if value is None else esc(value))
    return f'<td data-value="{esc(raw)}">{display}</td>'


def empty_result(result: dict, collection: str) -> str:
    if result.get('status') in {'failed', 'error'}:
        return '执行失败，无可评价结果。'
    if result.get('status') in {'not_run', 'skipped'}:
        return '未执行，无可评价结果。'
    if result.get('status') in {'unassessable', 'blocked'}:
        return state(result['status']) + '，无可评价结果。'
    if isinstance(result.get(collection), list):
        return '已保存结果为空（0 条）；需结合执行状态判断。'
    return '缺失：未保存结构化结果，执行状态未知；可展开原报告核对。'


def table(name, title, headers, rows, note='', statuses=False, empty='缺失：未保存结构化结果，执行状态未知；可展开原报告核对。'):
    if not rows:
        return f'<section class="result-block"><h2>{esc(title)}</h2><p class="empty">{esc(empty)}</p></section>'
    select = '<label>执行状态 <select data-filter><option value="">全部状态</option>' + ''.join(f'<option value="{esc(v)}">{esc(state(v))}</option>' for v in sorted({r[0] for r in rows})) + '</select></label>' if statuses else ''
    head = ''.join(f'<th scope="col" aria-sort="none"><button type="button" data-sort="{i}" title="点击排序">{esc(h)} ↕</button></th>' for i, h in enumerate(headers))
    rendered_rows = []
    for index, row in enumerate(rows):
        status, markup = row[:2]
        detail_id = f'{name}-detail-{index}'
        if len(row) > 2:
            markup += f'<td><button type="button" data-row-detail aria-expanded="false" aria-controls="{detail_id}">展开详情</button></td>'
        rendered_rows.append(f'<tr data-status="{esc(status)}">{markup}</tr>')
        if len(row) > 2:
            rendered_rows.append(f'<tr id="{detail_id}" data-detail hidden><td colspan="{len(headers)}">{row[2]}</td></tr>')
    body = ''.join(rendered_rows)
    return f'''<section class="result-block" data-report-table="{esc(name)}"><h2>{esc(title)}</h2><p class="muted">{esc(note)}</p>
    <div class="table-tools">{select}<span data-count aria-live="polite">共 {len(rows)} 条</span><button type="button" data-all>查看全部</button></div>
    <div class="table-scroll" tabindex="0" role="region" aria-label="{esc(title)}表格，可横向滚动"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div><p class="filter-empty" hidden>此状态下没有结果。</p></section>'''


def render_saved_report(run_id: str, run_dir: Path, manifest: dict, source: str) -> str:
    artifacts = run_dir / 'artifacts'
    def artifact_saved(name):
        return read_saved(run_dir, 'artifacts/' + name)
    catalog = artifact_saved(CLAIMS_FILE)
    coverage = artifact_saved(COVERAGE_FILE)
    visual = artifact_saved('visualizations.json')
    claims = records(catalog.get('claims'))
    calls = records(catalog.get('source_calls'))
    events = []
    event_path = run_dir / 'run_events.jsonl'
    if event_path.resolve().is_relative_to(run_dir.resolve()):
        try:
            for line in event_path.read_text(encoding='utf-8').splitlines():
                try:
                    e = mapping(json.loads(line))
                    if e.get('type') == 'tool_finished':
                        events.append(mapping(e.get('payload')))
                except ValueError:
                    continue
        except OSError:
            pass
    # Frozen catalog is authoritative for covered facts. Event outputs fill the
    # explicitly excluded categories; no live tool or evidence-store construction.
    sources = events + calls
    def latest(tool):
        return next((mapping(c.get('result')) for c in reversed(sources) if c.get('tool') == tool), {})

    def origin(tool, key=None):
        call = next((c for c in reversed(calls) if c.get('tool') == tool), {})
        matching = [c for c in claims if any(s.get('call_id') == call.get('call_id') for s in records(c.get('required_sources'))) and (key is None or c.get('key') == key)]
        links = ' '.join(f'<a href="#{quote(str(c.get("claim_id", "")), safe="")}">原结论：{esc(c.get("label", "查看来源"))}</a>' for c in matching)
        event = next((c for c in reversed(events) if c.get('tool') == tool), {})
        return json_detail('查看来源', {'来源工具': tool, '调用': call.get('call_id'), '证据引用': event.get('evidence_reference'), '报告快照': coverage.get('snapshot')}) + links

    schema = artifact_saved('schema_suggestions.json')
    def expr(value):
        return label_expression(str(value or '缺失'), schema)

    profile = mapping(latest('profile_dataset').get('profile'))
    rules_result = latest('apply_rules') or latest('apply_rule_library')
    rules = records(rules_result.get('rule_results'))
    association = latest('mine_association_rules')
    associations = records(association.get('rules'))
    correlations = records(latest('compute_correlations').get('correlations'))
    missing_result = latest('check_missing_values')
    missing = records(missing_result.get('columns_with_missing'))
    outlier_result = latest('detect_outliers')
    outliers = records(outlier_result.get('outliers'))
    reviews = records(artifact_saved('schema_review.json').get('review_items'))
    legacy = legacy_sections(source)
    panes = {key: '' for key, _ in TABS}
    base = f'/runs/{quote(run_id, safe="")}'
    def source_link(tool, key=None):
        return origin(tool, key)

    rule_rows = []
    for i, r in enumerate(rules):
        detail = f'<div class="row-detail"><h3>条件与建议</h3><p>条件：{esc(expr(r.get("condition")))}</p><p>样例索引：{esc(r.get("sample_indices", "缺失"))}</p><p>建议：{esc(r.get("recommendation", "未保存"))}</p><p>执行说明：{esc(r.get("message", "未保存"))}</p><p>规则来源：{esc(r.get("source_ref", "缺失"))}</p></div>'
        rule_rows.append((str(r.get('status', 'unassessable')), cell(r.get('name')) + cell(state(r.get('severity'))) + cell(state(r.get('status'))) + cell(r.get('matched_rows') if r.get('status') == 'evaluated' else None, numeric=True) + cell(r.get('match_rate') if r.get('status') == 'evaluated' else None, percent=True), detail + source_link("apply_rules", f"rule_{i}")))
    # Sort without manufacturing values for missing measurements.
    rule_rows.sort(key=lambda row: row[0])
    panes['rules'] += table('rules', '规则执行结果', ['规则名称', '严重程度', '执行状态', '命中行数', '命中比例', '详情与来源'], rule_rows, '默认按执行状态排序；点击列标题可排序。行数单位：行，比例以规则执行数据集为分母。', True, empty_result(rules_result, 'rule_results'))
    quality = []
    for r in missing:
        quality.append(('', cell(expr(r.get('column'))) + cell(r.get('missing_count'), numeric=True) + cell(r.get('missing_rate'), percent=True) + f'<td>{esc(r.get("recommendation", "未保存"))}</td>'))
    panes['rules'] += table('missing', '数据质量 · 缺失', ['字段', '缺失行数', '缺失比例', '建议'], quality, '保存的缺失检查结果；空结果不等同于检查通过。', empty=empty_result(missing_result, 'columns_with_missing')) + source_link('check_missing_values')
    quality = []
    for r in sorted(outliers, key=lambda r: -(r.get('outlier_rate') or 0)):
        quality.append(('', cell(expr(r.get('column'))) + cell(r.get('outlier_count'), numeric=True) + cell(r.get('outlier_rate'), percent=True) + '<td>' + json_detail('边界与样例', r) + '</td>'))
    panes['rules'] += table('outliers', '数据质量 · 离群提示', ['字段', '离群行数', '离群比例', '详情'], quality, f'方法：{outlier_result.get("method", "未保存")}；按离群比例降序。统计离群值仅是数据质量线索，不能直接称为故障。', empty=empty_result(outlier_result, 'outliers')) + source_link('detect_outliers')

    assoc_rows = []
    for i, r in sorted(enumerate(associations), key=lambda pair: -(pair[1].get('lift') or 0)):
        assoc_rows.append(('', cell(expr(r.get('antecedent'))) + cell(expr(r.get('consequent'))) + cell(r.get('support'), percent=True) + cell(r.get('confidence'), percent=True) + cell(r.get('lift'), numeric=True) + cell(r.get('matched_rows'), numeric=True) + '<td>' + json_detail('完整表达式与统计', r) + source_link('mine_association_rules', f'rule_{i}') + '</td>'))
    panes['associations'] += '<p class="notice">统计关联不代表因果；相关性保留正负方向，也不能据此判定物理故障。</p>'
    panes['associations'] += table('associations', '关联规则', ['前项', '后项', '支持度', '置信度', '提升度', '命中数', '详情与来源'], assoc_rows, f'按提升度降序；保存 {len(associations)} 条，挖掘上限 {mapping(association.get("parameters")).get("max_rules", association.get("max_rules", "未保存"))}；事务数 {association.get("transactions", "未保存")}，不是默认全量数据。支持度为事务占比，置信度为给定前项时后项的条件比例；提升度无量纲。', empty=empty_result(association, 'rules'))
    panes['associations'] += json_detail('挖掘范围与阈值', {k: v for k, v in association.items() if k != 'rules'})
    corr_rows = [('', cell(expr(r.get('field_a'))) + cell(expr(r.get('field_b'))) + cell(r.get('correlation'), numeric=True) + '<td>' + json_detail('原始字段与数值', r) + '</td>') for r in sorted(correlations, key=lambda r: -abs(r.get('correlation') or 0))]
    panes['associations'] += table('correlations', '数值字段相关性', ['字段一', '字段二', 'Pearson r（有符号）', '详情'], corr_rows, f'按 |r| 降序；保存 {len(correlations)} 对。正值同向，负值反向，r 无量纲；仅展示工具保存的范围。', empty=empty_result(latest('compute_correlations'), 'correlations')) + source_link('compute_correlations')

    charts = records(visual.get('charts'))
    panes['charts'] += f'<div class="section-title"><h2>结果图表</h2><a href="{base}/artifacts/visualizations.html">打开独立看板 ↗</a></div><p class="muted">来自已保存图表数据；图中为有限摘要，完整保存结果见规则与质量、关联分析。数值范围与单位见各图。百分比条形以 100% 为满幅，其他图以本图保存最大值为满幅；条形只对本图内可比。</p>'
    valid_charts = []
    for chart in charts:
        chart = dict(chart)
        chart['preserve_precision'] = True
        chart['rows'] = [r for r in records(chart.get('rows')) if isinstance(r.get('value'), (int, float)) and math.isfinite(r['value'])]
        valid_charts.append(_render_chart(chart))
    panes['charts'] += '<div class="chart-grid">' + ''.join(valid_charts) + '</div>' if valid_charts else '<p class="empty">未保存可用图表数据，请展开原报告图表或下载原报告。</p>'
    panes['charts'] += source_link('build_visualizations')

    panes['adaptation'] += '<h2>适配与人工复核</h2><p class="notice">适配检查通过仅表示既定检查项通过，不表示物理故障诊断通过或独立泛化验证通过。原产物中的自动诊断许可是历史门禁字段，不能扩大解释。</p>'
    panes['adaptation'] += table('reviews', '待复核清单', ['字段', '复核提示'], [('', cell(r.get('field')) + cell(r.get('message'))) for r in reviews], '来自保存的字段语义复核清单；忽略字段仍需确认用途。', empty=empty_result(artifact_saved('schema_review.json'), 'review_items'))
    for filename, title in [('rule_validation.json', '规则库校验'), ('schema_mapping.json', '字段映射'), ('dataset_adaptation.json', '数据适配'), ('mapping_execution_plan.json', '映射执行计划'), ('mapping_regression_validation.json', '映射回归'), ('generalization_readiness.json', '泛化门禁')]:
        d = artifact_saved(filename)
        panes['adaptation'] += f'<div class="review-line"><strong>{title}</strong><span>{esc(state(d.get("status"))) if d else "未保存"}</span>{json_detail("查看保存结果", d)}</div>'

    panes['evidence'] += '<h2>引用、快照与运行过程</h2><p class="notice">44/44 这样的覆盖率只表示既定范围的引用覆盖，不代表准确率或结论真实性。未覆盖内容、不可评价、未执行和失败均保留原状态。</p>'
    cov_rows = []
    for key, label in [('reference_coverage', '引用覆盖'), ('verified_coverage', '可验证引用覆盖')]:
        m = mapping(coverage.get(key))
        cov_rows.append(('', cell(label) + cell(f'{number(m.get("numerator"))}/{number(m.get("denominator"))}') + cell(m.get('ratio'), percent=True)))
    panes['evidence'] += table('coverage', '既定范围引用覆盖', ['项目', '已覆盖 / 范围内', '比例'], cov_rows, '这是保存时的覆盖记录；实时证据链校验请使用顶部入口。')
    validation = latest('validate_analysis')
    meta = read_saved(run_dir, 'run_metadata.json')
    snapshot = mapping(coverage.get('snapshot'))
    panes['evidence'] += '<dl class="timeline">' + ''.join(f'<dt>{esc(label)}</dt><dd>{number(v)} 节点</dd>' for label, v in [('结果校验时', validation.get('evidence_nodes')), ('报告渲染快照时', snapshot.get('node_count')), ('最终运行元数据记录时', mapping(meta.get('evidence')).get('node_count'))]) + '</dl>'
    panes['evidence'] += json_detail('覆盖范围与快照详情', coverage) + json_detail('运行标识与状态', {'run_id': run_id, 'status': manifest.get('status'), 'error': manifest.get('error'), 'model_config': manifest.get('model_config')})
    downloads = [ ('report.html', '原始 HTML 报告'), ('report.md', '原始文字报告'), (CLAIMS_FILE, '结论与来源'), (COVERAGE_FILE, '覆盖范围'), (SNAPSHOT_FILE, '报告快照'), ('visualizations.json', '图表数据')]
    panes['evidence'] += '<div id="downloads"><h3>产物下载</h3><div class="download-links">' + ''.join(f'<a href="{base}/artifacts/{name}?download=1" download>{title}</a>' for name, title in downloads if (artifacts/name).is_file() and (artifacts/name).resolve().is_relative_to(artifacts.resolve())) + '</div></div>'
    # The source catalog can have anchors absent from an old HTML revision.
    old_ids = {n.attrs.get('id') for n in SavedHTML(source).root.walk()}
    for c in claims:
        cid = c.get('claim_id')
        panes['evidence'] += json_detail('原结论 · ' + str(c.get('label', c.get('key', '未命名'))), c, str(cid) if cid and cid not in old_ids else '')
    panes['evidence'] += json_detail('全部已保存工具输出（含历史调用）', sources)

    evaluated = sum(r.get('status') == 'evaluated' for r in rules)
    metrics = [('数据行数', number(profile.get('rows'))), ('数据字段', number(profile.get('columns'))), ('规则执行', f'{evaluated} / {len(rules)}' if rules else '未保存'), ('字段待复核', str(len(reviews)) if artifact_saved('schema_review.json') else '未保存')]
    datasets = [mapping(v).get('original_name', '未命名') for v in mapping(manifest.get('files')).values() if mapping(v).get('kind') == 'dataset']
    panes['overview'] += '<div class="metrics" aria-label="关键指标">' + ''.join(f'<div><span>{label}</span><strong>{esc(value)}</strong></div>' for label, value in metrics) + '</div>'
    panes['overview'] += '<section class="overview-section"><h2>分析了什么</h2><p>' + esc('、'.join(datasets) or '数据集名称未保存') + '</p><p class="muted">规则命中、数据质量、字段统计关联及适配检查；基于本次保存结果。</p>' + source_link('profile_dataset', 'rows') + '</section>'
    measured = [r for r in rules if r.get('status') == 'evaluated' and isinstance(r.get('matched_rows'), (int, float)) and not isinstance(r.get('matched_rows'), bool) and math.isfinite(r['matched_rows'])]
    hits = [r for r in measured if r['matched_rows'] > 0]
    rules_available = isinstance(rules_result.get('rule_results'), list) and rules_result.get('status') not in {'failed', 'error', 'not_run', 'skipped', 'unassessable', 'blocked'}
    association_available = isinstance(association.get('rules'), list) and association.get('status') not in {'failed', 'error', 'not_run', 'skipped', 'unassessable', 'blocked'}
    rule_summary = f'可评价的 {len(measured)} 条已执行规则中，{len(hits)} 条存在命中。规则触发需要结合飞行阶段和原始曲线解释。' if rules_available and measured else ('尚无可评价的规则命中结果；请核对执行状态与缺失字段。' if rules_available and rules else '规则执行：' + empty_result(rules_result, 'rule_results'))
    panes['overview'] += f'<section class="overview-section"><h2>发现了什么</h2><p>{esc(rule_summary)}</p>'
    if hits:
        top = max(hits, key=lambda r: r['matched_rows'])
        panes['overview'] += f'<p>命中最多：{esc(top.get("name"))}，{number(top.get("matched_rows"))} 行（{number(top.get("match_rate"), True)}）。</p>'
    association_summary = f'关联挖掘保存 {len(associations)} 条规则；统计关联不代表因果。' if association_available else '关联挖掘：' + empty_result(association, 'rules')
    panes['overview'] += f'<p>{esc(association_summary)}</p><a href="#tab-rules">查看规则结果 →</a>　<a href="#tab-associations">查看关联结果 →</a></section>'
    panes['overview'] += '<section class="overview-section attention"><h2>哪些需要复核</h2><ul>'
    if reviews:
        for r in reviews[:3]:
            panes['overview'] += f'<li>{esc(r.get("field"))}：{esc(r.get("message"))}</li>'
        panes['overview'] += f'<li>当前摘要展示 {min(3,len(reviews))} / {len(reviews)} 项。<a href="#tab-adaptation">查看全部复核项 →</a></li>'
    else:
        panes['overview'] += '<li>待复核清单：' + esc(empty_result(artifact_saved('schema_review.json'), 'review_items')) + '</li>'
    if missing:
        panes['overview'] += f'<li>{len(missing)} 个字段存在缺失；<a href="#tab-rules">查看数据质量 →</a></li>'
    if manifest.get('error'):
        panes['overview'] += f'<li>运行错误：{esc(manifest["error"])}</li>'
    failed = [r for r in rules if r.get('status') != 'evaluated']
    if failed:
        panes['overview'] += f'<li>{len(failed)} 条规则未成功执行，不能当作零命中。<a href="#tab-rules">查看执行状态 →</a></li>'
    panes['overview'] += '</ul><p>适配通过不等于物理故障诊断或独立泛化验证通过。</p></section>'
    for key, _ in TABS:
        if legacy[key]:
            panes[key] += '<div class="legacy-group"><h3>原报告与来源</h3>' + legacy[key] + '</div>'
    nav = ''.join(f'<button role="tab" id="tab-{key}" aria-controls="pane-{key}" aria-selected="{str(key=="overview").lower()}" tabindex="{0 if key=="overview" else -1}">{title}</button>' for key, title in TABS)
    panels = ''.join(f'<section role="tabpanel" id="pane-{key}" aria-labelledby="tab-{key}" tabindex="0">{panes[key]}</section>' for key, _ in TABS)
    identity = '标准样例' if manifest.get('run_kind') == 'demo' else '数据分析'
    mode = state(mapping(manifest.get('model_config')).get('mode_label', '模式未保存'))
    css = Path(__file__).with_name('report_page.css').read_text()
    js = Path(__file__).with_name('report_page.js').read_text()
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>分析报告 · {identity}</title><link rel="icon" href="data:,"><style>{css}</style></head><body><main>
    <header class="report-header"><div><a class="back" href="/">工作台 / 分析结果</a><h1>分析报告 <span>{identity} · {esc(mode)}</span></h1><p class="muted">运行状态：{esc(state(manifest.get('status')))} · 记录时间 {esc(manifest.get('updated_at', '未保存'))}</p></div><nav aria-label="运行操作"><a href="{base}/progress">运行进度</a><a href='{base}/evidence'>证据校验</a><a href="#downloads">产物下载 ↓</a></nav></header>
    <div class="report-tabs" role="tablist" aria-label="报告内容">{nav}</div>{panels}<noscript><p>脚本未启用，全部内容按页签顺序展开。</p></noscript>
    </main><script>{js}</script></body></html>'''
