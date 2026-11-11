/* All behavior is local to this read-only response. No mutation API calls. */
(() => {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const panels = [...document.querySelectorAll('[role="tabpanel"]')];
  function activate(tab, focus=false) {
    tabs.forEach(t => { const yes=t===tab; t.setAttribute('aria-selected', String(yes));t.tabIndex=yes?0:-1; });
    panels.forEach(p => {p.hidden=p.id!==tab.getAttribute('aria-controls');});
    if(focus) tab.focus();
    const nav=tab.parentElement;
    nav.scrollTo({left:Math.max(0,tab.offsetLeft-nav.offsetLeft-12)});
  }
  function route() {
    let id;try{id=decodeURIComponent(location.hash.slice(1));}catch{ id=''; }
    const target=document.getElementById(id);
    const tab=target?.getAttribute('role')==='tab' ? target : tabs.find(t=>t.getAttribute('aria-controls')===target?.closest('[role="tabpanel"]')?.id) || tabs[0];
    activate(tab);
    if(target===tab) window.scrollTo({top:0});
    if(target && target!==tab) {
      for(let n=target;n;n=n.parentElement) if(n.tagName==='DETAILS') n.open=true;
      // A source anchor may be inside a table's collapsed summary.
      const block=target.closest('[data-report-table]');if(block){block.dataset.expanded='true';block.dispatchEvent(new Event('report-refresh'));}
      requestAnimationFrame(()=>target.scrollIntoView({block:'start'}));
    }
  }
  tabs.forEach((tab,i) => {
    tab.addEventListener('click',()=>{location.hash=tab.id;});
    tab.addEventListener('keydown',e=>{
      const indexes={ArrowRight:(i+1)%tabs.length,ArrowLeft:(i+tabs.length-1)%tabs.length,Home:0,End:tabs.length-1};
      if(e.key in indexes){e.preventDefault();const next=tabs[indexes[e.key]];location.hash=next.id;activate(next,true);}
    });
  });
  document.querySelectorAll('[data-report-table]').forEach(block=>{
    const body=block.querySelector('tbody'); let rows=[...body.querySelectorAll('tr[data-status]')];
    const filter=block.querySelector('[data-filter]');const all=block.querySelector('[data-all]');
    const count=block.querySelector('[data-count]');let sortColumn=-1,ascending=true;
    function refresh(){
      const selected=rows.filter(r=>!filter?.value||r.dataset.status===filter.value);
      const expanded=block.dataset.expanded==='true';
      rows.forEach(r=>{r.hidden=true;const detail=document.getElementById(r.querySelector('[data-row-detail]')?.getAttribute('aria-controls'));if(detail)detail.hidden=true;});
      selected.forEach((r,i)=>{r.hidden=!expanded&&i>=5;const toggle=r.querySelector('[data-row-detail]');const detail=document.getElementById(toggle?.getAttribute('aria-controls'));if(detail)detail.hidden=r.hidden||toggle.getAttribute('aria-expanded')!=='true';});
      count.textContent=`显示 ${expanded?selected.length:Math.min(5,selected.length)} / ${selected.length} 条（已保存 ${rows.length} 条）`;
      all.hidden=selected.length<=5;all.textContent=expanded?'收起摘要':'查看全部';all.setAttribute('aria-expanded',String(expanded));
      block.querySelector('.filter-empty').hidden=selected.length>0;
    }
    block.querySelectorAll('[data-row-detail]').forEach(toggle=>toggle.addEventListener('click',()=>{toggle.setAttribute('aria-expanded',String(toggle.getAttribute('aria-expanded')!=='true'));toggle.textContent=toggle.getAttribute('aria-expanded')==='true'?'收起详情':'展开详情';refresh();}));
    all.addEventListener('click',()=>{block.dataset.expanded=String(block.dataset.expanded!=='true');refresh();});
    filter?.addEventListener('change',()=>{block.dataset.expanded='false';refresh();});
    block.addEventListener('report-refresh',refresh);
    block.querySelectorAll('[data-sort]').forEach(button=>button.addEventListener('click',()=>{
      const col=Number(button.dataset.sort);ascending=sortColumn===col?!ascending:true;sortColumn=col;
      block.querySelectorAll('th').forEach(th=>th.setAttribute('aria-sort','none'));button.closest('th').setAttribute('aria-sort',ascending?'ascending':'descending');
      rows.sort((a,b)=>{
        const x=a.cells[col].dataset.value ?? a.cells[col].textContent.trim();
        const y=b.cells[col].dataset.value ?? b.cells[col].textContent.trim();
        if(x==='')return y===''?0:1;if(y==='')return -1;
        const n=Number(x),m=Number(y);const cmp=Number.isFinite(n)&&Number.isFinite(m)?n-m:x.localeCompare(y,'zh-CN',{numeric:true});return ascending?cmp:-cmp;
      });rows.forEach(r=>{const detail=document.getElementById(r.querySelector('[data-row-detail]')?.getAttribute('aria-controls'));body.append(r);if(detail)body.append(detail);});refresh();
    }));refresh();
  });
  window.addEventListener('hashchange',route);window.addEventListener('popstate',route);route();
})();
