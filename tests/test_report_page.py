"""HTTP report presentation must preserve frozen artifacts and source anchors."""
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aircraft_agent.agent.runtime import RunStore
from aircraft_agent.interface import app as web
from aircraft_agent.telemetry.coverage import CLAIMS_FILE, COVERAGE_FILE


def seal(root):
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            if p.is_file() else None for p in root.rglob('*')}


@pytest.fixture
def saved(tmp_path, monkeypatch):
    store = RunStore(tmp_path / 'runs')
    ctx = store.create_run('saved')
    report = ctx.artifacts_dir / 'report.html'
    report.write_text('''<html><body><header>旧版宣传和运行信息</header>
    <section id="demo-section-3"><h2>可视化看板</h2><p>旧图说明</p></section>
    <section id="demo-section-17"><h2>规则命中摘要</h2><ul><li id="claim-rule">原始规则来源</li></ul></section>
    <section id="legacy"><h2>未知旧章节</h2><p>不利结果不能删除 &lt;script&gt;</p></section></body></html>''')
    ctx.report_html_path = report
    ctx.status = 'complete'
    ctx.save_manifest()
    calls = [dict(call_id='call1', tool='apply_rules', result={'rule_results': [
        dict(rule_id='R1', name='<img onerror=alert(1)>', status='evaluated', severity='high', matched_rows=12, match_rate=.25, condition='x < 2', sample_indices=[3], recommendation='复核'),
        dict(rule_id='R2', name='失败规则', status='failed', matched_rows=None, match_rate=None),
    ]})]
    (ctx.artifacts_dir / CLAIMS_FILE).write_text(json.dumps({'source_calls': calls, 'claims': [dict(claim_id='claim-rule', category='rule_execution', key='rule_0', value=calls[0]['result']['rule_results'][0], required_sources=[{'call_id':'call1'}])]}))
    (ctx.artifacts_dir / COVERAGE_FILE).write_text(json.dumps({'snapshot': {'node_count': 41}, 'reference_coverage': {'numerator': 1,'denominator': 2}}))
    monkeypatch.setattr(web, 'store', store)
    return TestClient(web.app), ctx


def test_saved_report_uses_tabs_and_keeps_sources_read_only(saved):
    client, ctx = saved
    before = seal(ctx.run_dir)
    page = client.get(f'/runs/{ctx.run_id}').text
    assert 'role="tablist"' in page
    for name in ['概览', '规则与质量', '关联分析', '图表', '适配复核', '证据与过程']:
        assert name in page
    assert 'data-report-table="rules"' in page
    assert 'data-value="12"' in page
    assert 'data-value="0.25"' in page
    assert 'id="claim-rule"' in page and 'id="demo-section-3"' in page
    assert '不利结果不能删除' in page
    assert '&lt;img onerror=alert(1)&gt;' in page
    assert '<img onerror' not in page
    assert '不可评价' in page
    assert '准确率' in page and '统计关联不代表因果' in page
    assert seal(ctx.run_dir) == before


def test_legacy_no_structured_results_retains_all_text(saved):
    client, ctx = saved
    (ctx.artifacts_dir / CLAIMS_FILE).unlink()
    page = client.get(f'/runs/{ctx.run_id}').text
    assert '未保存结构化结果' in page
    assert '旧版宣传和运行信息' in page
    assert '不利结果不能删除' in page


def test_dashboard_alias_and_download_boundary(saved):
    client, ctx = saved
    (ctx.artifacts_dir / 'visualizations.html').write_text('<p>saved dashboard</p>')
    response = client.get(f'/runs/{ctx.run_id}/visualizations', follow_redirects=False)
    assert response.status_code == 307
    assert response.headers['location'] == f'/runs/{ctx.run_id}/artifacts/visualizations.html'
    download = client.get(f'/runs/{ctx.run_id}/artifacts/report.html?download=1')
    assert download.status_code == 200
    assert 'attachment' in download.headers['content-disposition']
    assert client.get(f'/runs/{ctx.run_id}/artifacts/manifest.json').status_code == 404
    outside = ctx.run_dir.parent / 'outside.html'
    outside.write_text('private outside')
    manifest = json.loads((ctx.run_dir / 'manifest.json').read_text())
    manifest['report_html_path'] = str(outside)
    (ctx.run_dir / 'manifest.json').write_text(json.dumps(manifest))
    assert client.get(f'/runs/{ctx.run_id}').status_code == 404


@pytest.mark.parametrize('data', [None, [], {'charts': [None]}, {'source_calls': [None], 'claims': [None]}])
def test_missing_or_malformed_optional_artifacts_are_explicit(saved, data):
    client, ctx = saved
    (ctx.artifacts_dir / CLAIMS_FILE).write_text(json.dumps(data))
    (ctx.artifacts_dir / 'visualizations.json').write_text(json.dumps(data))
    page = client.get(f'/runs/{ctx.run_id}')
    assert page.status_code == 200
    assert '未保存' in page.text


def test_failed_measurements_do_not_render_as_successful_zero(saved):
    client, ctx = saved
    p = ctx.artifacts_dir / CLAIMS_FILE
    data = json.loads(p.read_text())
    data['source_calls'][0]['result']['rule_results'][1].update(matched_rows=0, match_rate=0)
    p.write_text(json.dumps(data))
    page = client.get(f'/runs/{ctx.run_id}').text
    # A failed evaluation is not an observed zero-hit measurement.
    from aircraft_agent.interface.report_page import SavedHTML
    row = next(n for n in SavedHTML(page).root.walk() if n.tag == 'tr' and n.attrs.get('data-status') == 'failed')
    cells = [n for n in row.children if getattr(n, 'tag', '') == 'td']
    assert cells[3].text() == '不可评价'
    assert cells[4].text() == '不可评价'


def test_saved_empty_and_failed_results_are_distinct(saved):
    client, ctx = saved
    p = ctx.artifacts_dir / CLAIMS_FILE
    data = json.loads(p.read_text())
    data['source_calls'][0]['result'] = {'rule_results': [], 'status': 'failed', 'error': '保存的执行错误'}
    p.write_text(json.dumps(data))
    failed = client.get(f'/runs/{ctx.run_id}').text
    assert '执行失败，无可评价结果' in failed
    data['source_calls'][0]['result'] = {'rule_results': [], 'status': 'evaluated'}
    p.write_text(json.dumps(data))
    empty = client.get(f'/runs/{ctx.run_id}').text
    assert '已保存结果为空（0 条）' in empty


def test_legacy_markup_cannot_execute_scripts_or_escape_artifact_boundary(saved):
    client, ctx = saved
    ctx.report_html_path.write_text('<section id="old"><h2>旧格式</h2><p>保留文本</p><script>alert(1)</script><a href="javascript:alert(1)">链接</a><img src=x onerror=alert(2)></section>')
    response = client.get(f'/runs/{ctx.run_id}')
    assert response.status_code == 200
    assert '保留文本' in response.text
    assert 'javascript:alert' not in response.text and 'onerror=alert(2)' not in response.text
    external = ctx.run_dir.parent / 'external.json'
    external.write_text('{"private":"must not serve"}')
    (ctx.artifacts_dir / 'visualizations.json').symlink_to(external)
    assert client.get(f'/runs/{ctx.run_id}/artifacts/visualizations.json').status_code == 404


def test_saved_chart_preserves_numeric_precision_and_full_labels(saved):
    client, ctx = saved
    (ctx.artifacts_dir/'visualizations.json').write_text(json.dumps({'charts':[
        {'title':'保存图表','unit':'%','kind':'percent','rows':[{'label':'长字段'*60,'value':32.81}]}
    ]}))
    page = client.get(f'/runs/{ctx.run_id}').text
    assert '32.81%' in page
    assert '长字段'*60 in page


def test_overview_does_not_turn_missing_or_failed_findings_into_zero(saved):
    client, ctx = saved
    p = ctx.artifacts_dir / CLAIMS_FILE
    p.unlink()
    from aircraft_agent.interface.report_page import SavedHTML
    page = client.get(f'/runs/{ctx.run_id}').text
    overview = next(n for n in SavedHTML(page).root.walk() if n.attrs.get('id')=='pane-overview').text()
    assert '0 条存在命中' not in overview and '保存 0 条规则' not in overview
    p.write_text(json.dumps({'source_calls':[{'tool':'apply_rules','result':{'status':'failed','rule_results':[]}}]}))
    page = client.get(f'/runs/{ctx.run_id}').text
    overview = next(n for n in SavedHTML(page).root.walk() if n.attrs.get('id')=='pane-overview').text()
    assert '失败' in overview and '0 条存在命中' not in overview


def test_association_limit_uses_saved_parameters(saved):
    client, ctx = saved
    p = ctx.artifacts_dir / CLAIMS_FILE
    p.write_text(json.dumps({'source_calls':[{'tool':'mine_association_rules','result':{'rules':[{'lift':2}], 'parameters':{'max_rules':15}}}]}))
    page = client.get(f'/runs/{ctx.run_id}').text
    assert '挖掘上限 15' in page


def test_legacy_tables_have_contained_scroll(saved):
    client, ctx = saved
    ctx.report_html_path.write_text('<section id="old"><h2>旧格式</h2><table><tr><td>历史明细</td></tr></table></section>')
    page = client.get(f'/runs/{ctx.run_id}').text
    assert 'data-legacy-table' in page


def test_overview_all_unevaluable_rules_never_claim_zero_hits(saved):
    client, ctx = saved
    p=ctx.artifacts_dir / CLAIMS_FILE
    data={'source_calls':[{'tool':'apply_rules','result':{'rule_results':[{'name':'不可解析规则','status':'unparseable'}]}}]}
    p.write_text(json.dumps(data))
    from aircraft_agent.interface.report_page import SavedHTML
    page=client.get(f'/runs/{ctx.run_id}').text
    overview=next(n for n in SavedHTML(page).root.walk() if n.attrs.get('id')=='pane-overview').text()
    assert '0 条存在命中' not in overview
    assert '尚无可评价的规则命中结果' in overview
