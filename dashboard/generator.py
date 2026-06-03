# dashboard/generator.py
import json
import os
import webbrowser
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")

META_COLS = {
    "player_name", "team", "game", "commence_time",
    "pinnacle_odds", "pinnacle_prob", "sharp_anchor",
    "best_retail_odds", "best_retail_decimal", "best_retail_book",
    "ev_pct", "composite_score", "composite_z",
    "kelly_units", "stake_usd",
    # Quality / anchor metadata — string columns, not book odds
    "anchor_quality", "over_only",
    # Simulation columns — must NOT appear as book columns in the EV table
    "sim_prob", "sim_edge", "convergence",
    # Z-score columns — display-only scaling, not book odds
    "pin_prob_z", "sim_prob_z",
    # WAT scoring columns — display-only, not book odds
    "bet_score", "bet_grade",
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>HR Dashboard</title>
  <style>
    *{box-sizing:border-box}
    body{font-family:system-ui,sans-serif;padding:20px;background:#f8f9fa;margin:0}
    h1{color:#222;margin-bottom:4px}
    .meta{color:#666;font-size:.9em;margin-bottom:16px}

    /* ── Tabs ── */
    .tab-nav{display:flex;gap:0;margin-bottom:20px;border-bottom:2px solid #dee2e6}
    .tab-btn{padding:10px 22px;border:none;background:transparent;cursor:pointer;
             font-size:.95em;font-weight:500;color:#6c757d;border-bottom:2px solid transparent;
             margin-bottom:-2px;transition:color .15s,border-color .15s}
    .tab-btn:hover{color:#343a40}
    .tab-btn.active{color:#0d6efd;border-bottom-color:#0d6efd;font-weight:600}
    .tab-pane{display:none}
    .tab-pane.active{display:block}

    /* ── EV table ── */
    .controls{margin-bottom:12px;display:flex;align-items:center;gap:12px}
    table{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.1);font-size:.9em}
    th{background:#343a40;color:#fff;padding:10px 8px;cursor:pointer;text-align:left;white-space:nowrap;user-select:none}
    th:hover{background:#495057}
    td{padding:8px;border-bottom:1px solid #dee2e6;white-space:nowrap}
    .odds-pct{display:block;font-size:.78em;color:#c0392b;line-height:1.1}
    tr.positive-ev{background:#d4edda}
    tr.strong-play{background:#28a745!important;color:#fff;font-weight:700}
    tr.negative-ev td{color:#aaa}
    tr.hidden{display:none}
    .anchor-pin{color:#0d6efd;font-size:.78em;font-weight:600}
    .anchor-bol{color:#fd7e14;font-size:.78em;font-weight:600}

    /* ── Parlay builder ── */
    #parlay-builder{margin-top:32px;background:#fff;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,.1);border-radius:4px}
    #parlay-builder h2{margin-top:0}
    .parlay-leg{font-size:.95em;margin:2px 0}
    #parlay-stats{margin-top:12px;line-height:1.8}
    .stat-label{font-weight:600}
    #suggested-parlays{margin-top:0;background:#fff;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,.1);border-radius:4px}
    #suggested-parlays h2{margin-top:0;color:#495057}
    .parlay-card{border:1px solid #dee2e6;border-radius:6px;padding:14px 16px;margin-bottom:12px;background:#fafafa}
    .parlay-card.all-same-book{border-left:4px solid #0d6efd}
    .parlay-card.has-bol{border-left:4px solid #fd7e14}
    .parlay-card-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
    .parlay-ev{font-size:1.1em;font-weight:700;color:#28a745}
    .parlay-meta{color:#6c757d;font-size:.85em}
    .parlay-legs-list{margin:0;padding-left:18px;font-size:.9em;line-height:1.8}
    .flag-bol{background:#fff3cd;color:#856404;border-radius:3px;padding:1px 5px;font-size:.78em;margin-left:4px}
    .flag-book{background:#cfe2ff;color:#0a58ca;border-radius:3px;padding:1px 5px;font-size:.78em;margin-left:4px}

    /* ── Simulation ── */
    #sim-section{background:#fff;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,.1);border-radius:4px}
    #sim-section h2{margin-top:0;color:#495057}
    .sim-summary{display:flex;gap:16px;margin-bottom:16px}
    .sim-box{flex:1;border-radius:6px;padding:12px 16px;text-align:center}
    .sim-box-agree{background:#d4edda;border:1px solid #c3e6cb}
    .sim-box-bullish{background:#cfe2ff;border:1px solid #b6d4fe}
    .sim-box-bearish{background:#f8d7da;border:1px solid #f5c6cb}
    .sim-box-count{font-size:2em;font-weight:700}
    .sim-box-label{font-size:.85em;color:#495057}
    #sim-table{border-collapse:collapse;width:100%;font-size:.9em}
    #sim-table th{background:#343a40;color:#fff;padding:10px 8px;cursor:pointer;text-align:left;white-space:nowrap;user-select:none}
    #sim-table th:hover{background:#495057}
    #sim-table td{padding:8px;border-bottom:1px solid #dee2e6;white-space:nowrap}
    tr.sim-bullish-row{background:#d4edda}
    tr.sim-agree-row{background:#fff3cd}
    tr.sim-bearish-row td{color:#999}

    /* ── DFS tab ── */
    .dfs-section{background:#fff;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,.1);border-radius:4px;margin-bottom:24px}
    .dfs-section h2{margin-top:0;color:#495057;font-size:1.15em}
    .dfs-section h3{margin:18px 0 8px;color:#343a40;font-size:1em}
    .dfs-meta-bar{display:flex;gap:24px;margin-bottom:16px;font-size:.88em;color:#6c757d}
    .dfs-meta-bar strong{color:#343a40}
    .dfs-table{border-collapse:collapse;width:100%;font-size:.88em;margin-bottom:8px}
    .dfs-table th{background:#343a40;color:#fff;padding:8px 8px;cursor:pointer;text-align:left;white-space:nowrap;user-select:none}
    .dfs-table th:hover{background:#495057}
    .dfs-table td{padding:7px 8px;border-bottom:1px solid #dee2e6;white-space:nowrap}
    tr.dfs-conv{background:#d4edda;font-weight:600}
    tr.dfs-lev-high{background:#fff3cd}
    .dfs-no-data{color:#999;font-style:italic;padding:24px;text-align:center}
    .conv-star{color:#28a745;font-weight:700;margin-right:2px}
    .hr-ev-pos{color:#155724;font-weight:600}
    .hr-ev-neg{color:#721c24}
    .lev-pos{color:#0a58ca;font-weight:600}
    .lev-neg{color:#6c757d}
    .stack-team{font-weight:700;font-size:1em}
    .dfs-tab-nav{display:flex;gap:0;margin-bottom:16px;border-bottom:1px solid #dee2e6}
    .dfs-tab-btn{padding:7px 18px;border:none;background:transparent;cursor:pointer;
                 font-size:.88em;font-weight:500;color:#6c757d;border-bottom:2px solid transparent;
                 margin-bottom:-1px}
    .dfs-tab-btn.active{color:#0d6efd;border-bottom-color:#0d6efd;font-weight:600}
    .dfs-pane{display:none}
    .dfs-pane.active{display:block}
  </style>
</head>
<body>
  <h1>&#9968; Home Run Dashboard</h1>
  <p class="meta">__N_PLAYERS__ players &nbsp;|&nbsp; __N_POSITIVE__ +EV plays &nbsp;|&nbsp; __TIMESTAMP__</p>

  <!-- ── Tab navigation ── -->
  <div class="tab-nav">
    <button class="tab-btn active" data-tab="ev">&#128202; EV Analysis</button>
    <button class="tab-btn" data-tab="dfs">&#127919; DFS</button>
    <button class="tab-btn" data-tab="sim">&#128300; Simulation</button>
    <button class="tab-btn" data-tab="parlays">&#127922; Parlays</button>
  </div>

  <!-- ── EV Analysis Tab ── -->
  <div id="tab-ev" class="tab-pane active">
    <div class="controls">
      <label>Min EV%: <input type="range" id="ev-filter" min="-100" max="100" value="-100" step="1" oninput="applyFilter(+this.value)"></label>
      <span id="ev-label">-100%</span>
    </div>
    <table id="player-table">
      <thead><tr>
        <th></th>
        <th onclick="sortBy('player')">Player</th>
        <th onclick="sortBy('team')">Team</th>
        <th onclick="sortBy('time_sort')">Time (ET)</th>
        <th onclick="sortBy('pinnacle_pct')">Pin %</th>
        <th onclick="sortBy('pin_prob_z')">Pin Z</th>
        <th onclick="sortBy('sharp_anchor')">Anchor</th>
        __BOOK_HEADERS__
        <th onclick="sortBy('best_retail_odds')">Best Retail</th>
        <th onclick="sortBy('ev_pct')">EV%</th>
        <th onclick="sortBy('stake_units')">Stake</th>
        <th onclick="sortBy('composite_z')">Composite Z</th>
        <th onclick="sortBy('bet_score')">Score</th>
        <th>Grade</th>
      </tr></thead>
      <tbody id="table-body"></tbody>
    </table>
  </div>

  <!-- ── DFS Tab ── -->
  <div id="tab-dfs" class="tab-pane">
    __DFS_SECTION__
  </div>

  <!-- ── Simulation Tab ── -->
  <div id="tab-sim" class="tab-pane">
    __SIM_SECTION__
  </div>

  <!-- ── Parlays Tab ── -->
  <div id="tab-parlays" class="tab-pane">
    <div id="suggested-parlays">
      <h2>Suggested Longshot Parlays</h2>
      <p style="color:#6c757d;font-size:.9em">Same-book &middot; +EV legs at +500 to +1500 &middot; no same-game &middot; ranked by combined EV</p>
      <div id="parlay-cards"></div>
    </div>
    <div id="parlay-builder" style="margin-top:24px">
      <h2>Manual Parlay Builder</h2>
      <p style="color:#6c757d;font-size:.88em">Check players in the <strong>EV Analysis</strong> tab to add legs here.</p>
      <div id="parlay-legs"><em style="color:#999">No legs selected yet</em></div>
      <div id="parlay-stats"></div>
    </div>
  </div>

  <script>
    // ── Tab switching ──────────────────────────────────────────────────────────
    function showTab(tabId){
      document.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
      document.getElementById('tab-'+tabId).classList.add('active');
      document.querySelector('.tab-btn[data-tab="'+tabId+'"]').classList.add('active');
    }
    document.querySelectorAll('.tab-btn').forEach(btn=>{
      btn.addEventListener('click',()=>showTab(btn.dataset.tab));
    });

    // ── DFS sub-tab switching ──────────────────────────────────────────────────
    function showDfsPane(paneId,btn){
      document.querySelectorAll('.dfs-pane').forEach(p=>p.classList.remove('active'));
      document.querySelectorAll('.dfs-tab-btn').forEach(b=>b.classList.remove('active'));
      document.getElementById(paneId).classList.add('active');
      btn.classList.add('active');
    }

    // ── Core data ─────────────────────────────────────────────────────────────
    const DATA=__DATA__;
    const SIM_DATA=__SIM_DATA__;
    const DFS_DATA=__DFS_DATA__;
    let simSortKey='sim_edge',simSortDir=-1;
    const BOOKS=__BOOK_NAMES__;
    const PARLAYS=__PARLAYS__;
    let sortKey='composite_z',sortDir=-1,minEv=-100;
    const legs={};

    // ── Formatters ────────────────────────────────────────────────────────────
    function fmtOdds(v){return v==null?'--':v>0?'+'+v:''+v}
    function fmtPct(v){return(v>=0?'+':'')+v.toFixed(2)+'%'}
    function fmtZ(v){if(v==null)return'--';return(v>=0?'+':'')+v.toFixed(2)}
    function impliedPct(v){if(v==null)return'--';const d=v>0?(v/100)+1:(100/Math.abs(v))+1;return(100/d).toFixed(1)+'%'}
    function fmtBook(v){if(v==null)return'<td>--</td>';return`<td>${fmtOdds(v)}<span class="odds-pct">${impliedPct(v)}</span></td>`}
    function fmtAnchor(a){
      if(!a||a==='pinnacle')return'<span class="anchor-pin">PIN</span>';
      return'<span class="anchor-bol">BOL</span>';
    }

    // ── EV Table ──────────────────────────────────────────────────────────────
    function rowCls(r){
      if(r.composite_z>=1.5)return 'strong-play';
      if(r.ev_pct>0)return 'positive-ev';
      return 'negative-ev';
    }
    function renderTable(){
      const sorted=[...DATA].sort((a,b)=>{
        const av=a[sortKey],bv=b[sortKey];
        if(typeof av==='string')return sortDir*av.localeCompare(bv);
        if(av==null&&bv==null)return 0;
        if(av==null)return 1;
        if(bv==null)return -1;
        return sortDir*(av-bv);
      });
      document.getElementById('table-body').innerHTML=sorted.map(r=>`
        <tr class="${rowCls(r)}${r.ev_pct<minEv?' hidden':''}">
          <td><input type="checkbox" ${legs[r.player+'|'+r.game]?'checked':''} onchange="toggleLeg('${r.player}','${r.game}',this)"></td>
          <td>${r.player}</td><td>${r.team}</td><td>${r.time}</td>
          <td>${r.pinnacle_pct.toFixed(1)}%</td>
          <td>${fmtZ(r.pin_prob_z)}</td>
          <td>${fmtAnchor(r.sharp_anchor)}</td>
          ${BOOKS.map(b=>fmtBook(r[b])).join('')}
          <td>${fmtOdds(r.best_retail_odds)}</td>
          <td>${fmtPct(r.ev_pct)}</td>
          <td>${r.stake}</td>
          <td>${fmtZ(r.composite_z)}</td>
          ${(()=>{const gradeColors={'Strong':'#28a745','Solid':'#17a2b8','Marginal':'#ffc107','Skip':'#6c757d'};const g=r.bet_grade||'';const c=gradeColors[g]||'#6c757d';const s=r.bet_score!=null?r.bet_score:'--';return`<td style="font-weight:700;color:${c}">${s}</td><td style="color:${c}">${g||'--'}</td>`;})()}
        </tr>`).join('');
    }

    // ── Simulation Table ──────────────────────────────────────────────────────
    function sortSim(k){if(simSortKey===k)simSortDir*=-1;else{simSortKey=k;simSortDir=-1;}renderSimTable();}
    function renderSimTable(){
      const tbody=document.getElementById('sim-table-body');
      if(!tbody)return;
      if(!SIM_DATA||!SIM_DATA.length){
        tbody.innerHTML='<tr><td colspan="12" style="color:#999;font-style:italic;text-align:center">No simulation data.</td></tr>';
        return;
      }
      const sorted=[...SIM_DATA].sort((a,b)=>{
        const av=a[simSortKey],bv=b[simSortKey];
        if(typeof av==='string')return simSortDir*av.localeCompare(bv);
        if(av==null&&bv==null)return 0;if(av==null)return 1;if(bv==null)return -1;
        return simSortDir*(av-bv);
      });
      tbody.innerHTML=sorted.filter(r=>r.ev_pct>=minEv).map(r=>{
        let cls='';
        if(r.sim_edge>3&&r.ev_pct>0)cls='sim-bullish-row';
        else if(Math.abs(r.sim_edge)<=3&&r.ev_pct>0)cls='sim-agree-row';
        else if(r.sim_edge<-3)cls='sim-bearish-row';
        const edgeStr=(r.sim_edge>=0?'+':'')+r.sim_edge.toFixed(1)+'%';
        const convBadge=r.convergence==='AGREE'?'<span style="color:#155724;font-weight:600">&#10003;AGREE</span>':'<span style="color:#721c24">DIVERGE</span>';
        const simZStr=r.sim_prob_z!=null?fmtZ(r.sim_prob_z):'--';
        const pinZStr=r.pin_prob_z!=null?fmtZ(r.pin_prob_z):'--';
        return`<tr class="${cls}">
          <td>${r.player}</td><td>${r.team}</td><td>${r.game}</td>
          <td>${r.sim_prob.toFixed(1)}%</td><td>${simZStr}</td>
          <td>${r.pin_prob.toFixed(1)}%</td><td>${pinZStr}</td>
          <td>${edgeStr}</td><td>${convBadge}</td>
          <td>${fmtOdds(r.best_retail_odds)}</td><td>${fmtPct(r.ev_pct)}</td>
          <td>${r.stake}</td>
        </tr>`;
      }).join('');
    }

    // ── DFS Tables ────────────────────────────────────────────────────────────
    function fmtLev(v){
      if(v==null)return'--';
      const s=(v>=0?'+':'')+v.toFixed(2);
      return v>=0?`<span class="lev-pos">${s}</span>`:`<span class="lev-neg">${s}</span>`;
    }
    function fmtHrEv(v){
      if(v==null)return'--';
      const s=(v>=0?'+':'')+v.toFixed(1)+'%';
      return v>0?`<span class="hr-ev-pos">${s}</span>`:`<span class="hr-ev-neg">${s}</span>`;
    }
    function fmtHrOdds(v){
      if(v==null)return'--';
      return v>0?'+'+v:''+v;
    }
    function renderDfsConvergences(){
      const tbody=document.getElementById('dfs-conv-body');
      if(!tbody)return;
      if(!DFS_DATA||!DFS_DATA.convergences||!DFS_DATA.convergences.length){
        tbody.innerHTML='<tr><td colspan="9" class="dfs-no-data">No convergence plays (need Lev ≥ 0 and +EV HR prop)</td></tr>';return;
      }
      tbody.innerHTML=_dfsSorted(DFS_DATA.convergences,'conv').map((r,i)=>`
        <tr class="${i<3?'dfs-conv':''}">
          <td><span class="conv-star">&#9733;</span>${r.player}</td>
          <td>${r.team}</td><td>${r.pos}</td>
          <td>${fmtLev(r.leverage)}</td>
          <td>${r.pts}</td>
          <td>${fmtHrEv(r.hr_ev_pct)}</td>
          <td>${r.hr_pin_pct!=null?r.hr_pin_pct+'%':'--'}</td>
          <td>${fmtHrOdds(r.hr_odds)}${r.hr_book?` <small style="color:#6c757d">@ ${r.hr_book}</small>`:''}</td>
          <td>${r.conv_score.toFixed(2)}</td>
        </tr>`).join('');
    }
    function renderDfsLeverages(){
      const tbody=document.getElementById('dfs-lev-body');
      if(!tbody)return;
      if(!DFS_DATA||!DFS_DATA.leverages||!DFS_DATA.leverages.length){
        tbody.innerHTML='<tr><td colspan="8" class="dfs-no-data">No DFS data loaded. Drop your CSV at data/dfs_projections.csv and re-run.</td></tr>';return;
      }
      tbody.innerHTML=_dfsSorted(DFS_DATA.leverages,'lev').map((r,i)=>`
        <tr class="${i<5?'dfs-lev-high':''}">
          <td>${r.player}</td><td>${r.pos}</td><td>${r.team}</td>
          <td>${r.pts}</td>
          <td>${r.own}%</td>
          <td>$${(r.sal/1000).toFixed(1)}k</td>
          <td>${fmtLev(r.leverage)}</td>
          <td>${r.hr_ev_pct!=null?fmtHrEv(r.hr_ev_pct):'--'}</td>
        </tr>`).join('');
    }
    function renderDfsStacks(){
      const tbody=document.getElementById('dfs-stack-body');
      if(!tbody)return;
      if(!DFS_DATA||!DFS_DATA.stacks||!DFS_DATA.stacks.length){
        tbody.innerHTML='<tr><td colspan="7" class="dfs-no-data">No stack data.</td></tr>';return;
      }
      tbody.innerHTML=_dfsSorted(DFS_DATA.stacks,'stack').map((r,i)=>`
        <tr>
          <td><span class="stack-team">${r.team}</span></td>
          <td>${r.players}</td>
          <td>${r.total_pts}</td>
          <td>${r.avg_pts!=null?r.avg_pts:'--'}</td>
          <td>${r.avg_own}%</td>
          <td>${fmtLev(r.avg_leverage)}</td>
          <td style="max-width:280px;white-space:normal;font-size:.85em">${r.top_players}</td>
        </tr>`).join('');
    }
    function renderDfsCrossovers(){
      const tbody=document.getElementById('dfs-cross-body');
      if(!tbody)return;
      if(!DFS_DATA||!DFS_DATA.crossovers||!DFS_DATA.crossovers.length){
        tbody.innerHTML='<tr><td colspan="8" class="dfs-no-data">No HR crossover plays found.</td></tr>';return;
      }
      tbody.innerHTML=_dfsSorted(DFS_DATA.crossovers,'cross').map(r=>`
        <tr>
          <td>${r.player}</td><td>${r.team}</td><td>${r.pos}</td>
          <td>${r.pts}</td>
          <td>${fmtLev(r.leverage)}</td>
          <td>${r.hr_pin_pct!=null?r.hr_pin_pct+'%':'--'}</td>
          <td>${fmtHrEv(r.hr_ev_pct)}</td>
          <td>${fmtHrOdds(r.hr_odds)}${r.hr_book?` <small style="color:#6c757d">@ ${r.hr_book}</small>`:''}</td>
        </tr>`).join('');
    }
    // ── DFS sort helpers ─────────────────────────────────────────────────────
    const dfsSortState={
      conv: {key:'conv_score',dir:-1},
      lev:  {key:'leverage',  dir:-1},
      stack:{key:'total_pts', dir:-1},
      cross:{key:'hr_ev_pct', dir:-1},
    };
    function sortDfsBy(tbl,key){
      const s=dfsSortState[tbl];
      if(s.key===key)s.dir*=-1;else{s.key=key;s.dir=-1;}
      if(tbl==='conv')renderDfsConvergences();
      else if(tbl==='lev')renderDfsLeverages();
      else if(tbl==='stack')renderDfsStacks();
      else if(tbl==='cross')renderDfsCrossovers();
    }
    function _dfsSorted(arr,tbl){
      const s=dfsSortState[tbl];
      if(!s||!arr||!arr.length)return arr;
      return[...arr].sort((a,b)=>{
        const av=a[s.key],bv=b[s.key];
        if(av==null&&bv==null)return 0;
        if(av==null)return 1;if(bv==null)return-1;
        if(typeof av==='string')return s.dir*av.localeCompare(bv);
        return s.dir*(av-bv);
      });
    }
    function renderAllDfs(){
      renderDfsConvergences();renderDfsLeverages();renderDfsStacks();renderDfsCrossovers();
    }

    // ── Sort helpers ──────────────────────────────────────────────────────────
    function sortBy(k){if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=-1;}renderTable();}
    function applyFilter(v){minEv=v;document.getElementById('ev-label').textContent=v+'%';renderTable();}

    // ── Parlay builder ────────────────────────────────────────────────────────
    function americanToDecimal(o){return o>0?(o/100)+1:(100/Math.abs(o))+1}
    function decimalToAmerican(d){return d>=2?'+'+(Math.round((d-1)*100)):''+Math.round(-100/(d-1))}
    function toggleLeg(player,game,cb){
      const key=player+'|'+game;
      if(cb.checked)legs[key]=DATA.find(d=>d.player===player&&d.game===game);
      else delete legs[key];
      updateParlay();
    }
    function updateParlay(){
      const sel=Object.values(legs);
      const legsDiv=document.getElementById('parlay-legs');
      const statsDiv=document.getElementById('parlay-stats');
      if(!sel.length){
        legsDiv.innerHTML='<em style="color:#999">No legs selected yet</em>';
        statsDiv.innerHTML='';return;
      }
      legsDiv.innerHTML=sel.map(l=>`<div class="parlay-leg">${l.player} &mdash; ${fmtOdds(l.best_retail_odds)}</div>`).join('');
      const pDec=sel.reduce((a,l)=>a*americanToDecimal(l.best_retail_odds),1);
      const pProb=sel.reduce((a,l)=>a*(l.pinnacle_pct/100),1);
      const pEv=(pProb*pDec)-1;
      const pComp=pEv*pProb;
      const comps=DATA.map(d=>d.ev_pct/100*(d.pinnacle_pct/100));
      const mean=comps.reduce((a,b)=>a+b,0)/comps.length;
      const std=Math.sqrt(comps.map(x=>(x-mean)**2).reduce((a,b)=>a+b,0)/comps.length);
      const pZ=std>0?fmtZ((pComp-mean)/std):'--';
      const gameCounts={};
      sel.forEach(l=>{if(l.game)gameCounts[l.game]=(gameCounts[l.game]||0)+1});
      const dupGames=Object.entries(gameCounts).filter(e=>e[1]>=2);
      let warn='';
      if(dupGames.length){
        const desc=dupGames.map(e=>`${e[1]} legs in ${e[0]}`).join('; ');
        warn=`<div style="background:#fff3cd;border:1px solid #ffc107;padding:8px 10px;border-radius:4px;margin-bottom:8px;color:#856404"><strong>&#9888; Same-game correlation:</strong> ${desc}. HR outcomes within one game are correlated.</div>`;
      }
      statsDiv.innerHTML=warn+`
        <div><span class="stat-label">Legs:</span> ${sel.length}</div>
        <div><span class="stat-label">Combined Probability:</span> ${(pProb*100).toFixed(2)}%</div>
        <div><span class="stat-label">Combined Odds:</span> ${decimalToAmerican(pDec)}</div>
        <div><span class="stat-label">Combined EV%:</span> ${fmtPct(pEv*100)}</div>
        <div><span class="stat-label">Composite Z:</span> ${pZ}</div>`;
    }

    // ── Suggested parlays ─────────────────────────────────────────────────────
    function renderParlays(){
      const div=document.getElementById('parlay-cards');
      if(!PARLAYS||!PARLAYS.length){
        div.innerHTML='<p style="color:#999;font-style:italic">No qualifying same-book parlays for today.</p>';
        return;
      }
      div.innerHTML=PARLAYS.map((p,i)=>{
        const cls=['parlay-card',p.all_same_book?'all-same-book':'',p.has_betonline_anchor?'has-bol':''].filter(Boolean).join(' ');
        const book=p.book||(p.books&&p.books[0])||'?';
        const bolFlag=p.has_betonline_anchor?'<span class="flag-bol">BOL anchor</span>':'';
        const legsHtml=p.legs.map((leg,j)=>{
          const o=p.leg_odds[j];
          return`<li>${leg} &nbsp;<strong>${o>0?'+'+o:''+o}</strong><em style="color:#6c757d"> (EV ${p.leg_ev_pcts[j]>=0?'+':''}${p.leg_ev_pcts[j]}%)</em></li>`;
        }).join('');
        return`<div class="${cls}">
  <div class="parlay-card-header">
    <span><strong>#${i+1} &nbsp; ${p.n_legs}-leg</strong> &nbsp; <span class="flag-book">${book}</span> &nbsp; ${bolFlag}</span>
    <span class="parlay-ev">EV +${p.combined_ev_pct}%</span>
  </div>
  <div class="parlay-meta">Odds ${p.combined_american} &nbsp;|&nbsp; Hit prob ${p.combined_prob_pct}%</div>
  <ul class="parlay-legs-list">${legsHtml}</ul>
</div>`;
      }).join('');
    }

    // ── Init ──────────────────────────────────────────────────────────────────
    renderTable();
    renderParlays();
    renderSimTable();
    renderAllDfs();
  </script>
</body>
</html>"""


_SIM_COLS = {"sim_prob", "sim_edge", "convergence"}

_SIM_TABLE_HTML = """<div id="sim-section">
  <h2>Simulation Analysis</h2>
  <div class="sim-summary">
    <div class="sim-box sim-box-agree">
      <div class="sim-box-count">{n_agree}</div>
      <div class="sim-box-label">&#127823; Convergence plays<br><small>+EV &amp; |sim edge| &lt;3%</small></div>
    </div>
    <div class="sim-box sim-box-bullish">
      <div class="sim-box-count">{n_bullish}</div>
      <div class="sim-box-label">&#128309; Sim bullish<br><small>sim &gt; pin by &gt;3%</small></div>
    </div>
    <div class="sim-box sim-box-bearish">
      <div class="sim-box-count">{n_bearish}</div>
      <div class="sim-box-label">&#128308; Sim bearish<br><small>sim &lt; pin by &gt;3%</small></div>
    </div>
  </div>
  <p style="color:#6c757d;font-size:.9em">Sorted by sim edge &darr; &nbsp;|&nbsp; Green=bullish+EV &nbsp; Yellow=convergence+EV &nbsp; Gray=bearish</p>
  <table id="sim-table">
    <thead><tr>
      <th onclick="sortSim('player')">Player</th>
      <th onclick="sortSim('team')">Team</th>
      <th onclick="sortSim('game')">Game</th>
      <th onclick="sortSim('sim_prob')">Sim %</th>
      <th onclick="sortSim('sim_prob_z')">Sim Z</th>
      <th onclick="sortSim('pin_prob')">Pin %</th>
      <th onclick="sortSim('pin_prob_z')">Pin Z</th>
      <th onclick="sortSim('sim_edge')">Sim Edge</th>
      <th>Signal</th>
      <th onclick="sortSim('best_retail_odds')">Best Retail</th>
      <th onclick="sortSim('ev_pct')">EV%</th>
      <th>Stake</th>
    </tr></thead>
    <tbody id="sim-table-body"></tbody>
  </table>
</div>"""

_SIM_UNAVAILABLE_HTML = """<div id="sim-section">
  <h2>Simulation Analysis</h2>
  <p style="color:#6c757d;font-style:italic">Simulation data unavailable for this slate.
  Check <code>data/sim_unmatched.log</code> for details.</p>
</div>"""

_DFS_SECTION_HTML = """<div class="dfs-section">
  <h2>&#127919; DFS Analysis</h2>
  <div class="dfs-meta-bar">
    <span><strong>{total_players}</strong> players in slate</span>
    <span><strong>{active_hitters}</strong> active hitters</span>
    <span><strong>{hr_matches}</strong> HR prop matches</span>
    <span style="color:#6c757d">Leverage = Pts z &minus; Own z &nbsp;|&nbsp; Conv Score = Lev + HR EV/15</span>
  </div>

  <div class="dfs-tab-nav">
    <button class="dfs-tab-btn active" onclick="showDfsPane('dfs-conv-pane',this)">&#11088; Convergence</button>
    <button class="dfs-tab-btn" onclick="showDfsPane('dfs-lev-pane',this)">&#128200; Leverage</button>
    <button class="dfs-tab-btn" onclick="showDfsPane('dfs-stack-pane',this)">&#127968; Stacks</button>
    <button class="dfs-tab-btn" onclick="showDfsPane('dfs-cross-pane',this)">&#128279; HR Crossover</button>
  </div>

  <div id="dfs-conv-pane" class="dfs-pane active">
    <p style="color:#6c757d;font-size:.85em;margin-bottom:8px">Players with high DFS leverage AND a +EV HR prop today &mdash; the strongest multi-format signals.</p>
    <table class="dfs-table">
      <thead><tr>
        <th onclick="sortDfsBy('conv','player')">Player</th>
        <th onclick="sortDfsBy('conv','team')">Team</th>
        <th onclick="sortDfsBy('conv','pos')">Pos</th>
        <th onclick="sortDfsBy('conv','leverage')">DFS Lev</th>
        <th onclick="sortDfsBy('conv','pts')">Pts</th>
        <th onclick="sortDfsBy('conv','hr_ev_pct')">HR EV%</th>
        <th onclick="sortDfsBy('conv','hr_pin_pct')">HR Pin%</th>
        <th>Best Odds</th>
        <th onclick="sortDfsBy('conv','conv_score')">Conv Score</th>
      </tr></thead>
      <tbody id="dfs-conv-body"></tbody>
    </table>
  </div>

  <div id="dfs-lev-pane" class="dfs-pane">
    <p style="color:#6c757d;font-size:.85em;margin-bottom:8px">Top {top_leverage} hitters sorted by leverage score (high projection, low ownership). Yellow = top 5.</p>
    <table class="dfs-table">
      <thead><tr>
        <th onclick="sortDfsBy('lev','player')">Player</th>
        <th onclick="sortDfsBy('lev','pos')">Pos</th>
        <th onclick="sortDfsBy('lev','team')">Team</th>
        <th onclick="sortDfsBy('lev','pts')">Pts</th>
        <th onclick="sortDfsBy('lev','own')">Own%</th>
        <th onclick="sortDfsBy('lev','sal')">Salary</th>
        <th onclick="sortDfsBy('lev','leverage')">Leverage</th>
        <th onclick="sortDfsBy('lev','hr_ev_pct')">HR EV%</th>
      </tr></thead>
      <tbody id="dfs-lev-body"></tbody>
    </table>
  </div>

  <div id="dfs-stack-pane" class="dfs-pane">
    <p style="color:#6c757d;font-size:.85em;margin-bottom:8px">Teams ranked by total projected points. Higher avg leverage = lower-owned stack.</p>
    <table class="dfs-table">
      <thead><tr>
        <th onclick="sortDfsBy('stack','team')">Team</th>
        <th onclick="sortDfsBy('stack','players')">Hitters</th>
        <th onclick="sortDfsBy('stack','total_pts')">Total Pts</th>
        <th onclick="sortDfsBy('stack','avg_pts')">Avg Pts</th>
        <th onclick="sortDfsBy('stack','avg_own')">Avg Own%</th>
        <th onclick="sortDfsBy('stack','avg_leverage')">Avg Lev</th>
        <th>Top 3 Players</th>
      </tr></thead>
      <tbody id="dfs-stack-body"></tbody>
    </table>
  </div>

  <div id="dfs-cross-pane" class="dfs-pane">
    <p style="color:#6c757d;font-size:.85em;margin-bottom:8px">All hitters with a HR prop today, sorted by HR EV%.</p>
    <table class="dfs-table">
      <thead><tr>
        <th onclick="sortDfsBy('cross','player')">Player</th>
        <th onclick="sortDfsBy('cross','team')">Team</th>
        <th onclick="sortDfsBy('cross','pos')">Pos</th>
        <th onclick="sortDfsBy('cross','pts')">DFS Pts</th>
        <th onclick="sortDfsBy('cross','leverage')">DFS Lev</th>
        <th onclick="sortDfsBy('cross','hr_pin_pct')">HR Pin%</th>
        <th onclick="sortDfsBy('cross','hr_ev_pct')">HR EV%</th>
        <th>Best Odds</th>
      </tr></thead>
      <tbody id="dfs-cross-body"></tbody>
    </table>
  </div>
</div>"""

_DFS_UNAVAILABLE_HTML = """<div class="dfs-section">
  <h2>&#127919; DFS Analysis</h2>
  <p class="dfs-no-data">No DFS projections loaded.<br>
  Drop a DraftKings-style CSV at <code>data/dfs_projections.csv</code> and re-run the dashboard.<br>
  Required columns: <em>Player, Pos, Tm, Opp, Lineup, Sal, Val, Own, Pts, Own z, Pts z</em></p>
</div>"""


def generate_dashboard(
    final_df: pd.DataFrame,
    output_path: str = "hr_dashboard.html",
    open_browser: bool = True,
    *,
    parlays: list | None = None,
    dfs_data: dict | None = None,
) -> str:
    book_cols = [c for c in final_df.columns if c not in META_COLS]

    # --- Per-slate z-scores for scaling context ---
    _df = final_df.copy()

    _pin_std = _df["pinnacle_prob"].std(ddof=0)
    _pin_mean = _df["pinnacle_prob"].mean()
    _df["pin_prob_z"] = (
        (_df["pinnacle_prob"] - _pin_mean) / _pin_std
        if _pin_std > 0 else 0.0
    )

    _df["sim_prob_z"] = float("nan")
    if "sim_prob" in _df.columns:
        _sim_mask = _df["sim_prob"].notna()
        if _sim_mask.sum() >= 2:
            _sv = _df.loc[_sim_mask, "sim_prob"]
            _sim_std = _sv.std(ddof=0)
            _df.loc[_sim_mask, "sim_prob_z"] = (
                (_sv - _sv.mean()) / _sim_std if _sim_std > 0 else 0.0
            )

    records = []
    for _, row in _df.iterrows():
        record = {
            "player": row["player_name"],
            "team": row.get("team", ""),
            "game": row.get("game", ""),
            "time": row["commence_time"].astimezone(ET).strftime("%I:%M %p ET"),
            "time_sort": row["commence_time"].timestamp(),
            "pinnacle_pct": round(row["pinnacle_prob"] * 100, 2),
            "pin_prob_z": round(float(row["pin_prob_z"]), 2),
            "sharp_anchor": row.get("sharp_anchor", "pinnacle"),
            "best_retail_odds": int(row["best_retail_odds"]),
            "ev_pct": round(row["ev_pct"] * 100, 2),
            "stake_units": round(row["kelly_units"], 1),
            "stake": (
                f'{row["kelly_units"]:g}u (${row["stake_usd"]:,.0f})'
                if row["kelly_units"] > 0 else "0u"
            ),
            "composite_z": round(row["composite_z"], 2),
            "bet_score": int(row["bet_score"]) if pd.notna(row.get("bet_score")) else None,
            "bet_grade": str(row["bet_grade"]) if pd.notna(row.get("bet_grade")) else None,
        }
        for col in book_cols:
            val = row.get(col)
            record[col] = int(val) if pd.notna(val) else None
        records.append(record)

    # Build simulation section
    if _SIM_COLS.issubset(_df.columns) and _df["sim_prob"].notna().any():
        sim_records = []
        for _, row in _df.iterrows():
            if pd.isna(row.get("sim_prob")):
                continue
            _spz = row.get("sim_prob_z")
            sim_records.append({
                "player": row["player_name"],
                "team": row.get("team", ""),
                "game": row.get("game", ""),
                "sim_prob": round(float(row["sim_prob"]) * 100, 1),
                "sim_prob_z": round(float(_spz), 2) if pd.notna(_spz) else None,
                "pin_prob": round(float(row["pinnacle_prob"]) * 100, 1),
                "pin_prob_z": round(float(row["pin_prob_z"]), 2),
                "sim_edge": round(float(row["sim_edge"]) * 100, 1),
                "convergence": row["convergence"],
                "best_retail_odds": int(row["best_retail_odds"]),
                "ev_pct": round(float(row["ev_pct"]) * 100, 2),
                "stake": (
                    f'{row["kelly_units"]:g}u (${row["stake_usd"]:,.0f})'
                    if row["kelly_units"] > 0 else "0u"
                ),
            })
        n_agree = sum(
            1 for r in sim_records if r["convergence"] == "AGREE" and r["ev_pct"] > 0
        )
        n_bullish = sum(1 for r in sim_records if r["sim_edge"] > 3)
        n_bearish = sum(1 for r in sim_records if r["sim_edge"] < -3)
        sim_section_html = _SIM_TABLE_HTML.format(
            n_agree=n_agree, n_bullish=n_bullish, n_bearish=n_bearish
        )
    else:
        sim_records = []
        sim_section_html = _SIM_UNAVAILABLE_HTML

    # Build DFS section
    if dfs_data is not None:
        from agents.dfs import _TOP_LEVERAGE
        meta = dfs_data.get("meta", {})
        dfs_section_html = _DFS_SECTION_HTML.format(
            total_players=meta.get("total_players", 0),
            active_hitters=meta.get("active_hitters", 0),
            hr_matches=meta.get("hr_matches", 0),
            top_leverage=_TOP_LEVERAGE,
        )
        dfs_json = json.dumps(dfs_data)
    else:
        dfs_section_html = _DFS_UNAVAILABLE_HTML
        dfs_json = "null"

    timestamp = datetime.now(ET).strftime("%Y-%m-%d %I:%M %p ET")
    n_players = len(_df)
    n_positive = int((_df["ev_pct"] > 0).sum())
    book_headers = "".join(f'<th onclick="sortBy(\'{b}\')">{b}</th>' for b in book_cols)

    html = (
        HTML_TEMPLATE
        .replace("__DATA__", json.dumps(records))
        .replace("__BOOK_NAMES__", json.dumps(book_cols))
        .replace("__BOOK_HEADERS__", book_headers)
        .replace("__PARLAYS__", json.dumps(parlays or []))
        .replace("__TIMESTAMP__", timestamp)
        .replace("__N_PLAYERS__", str(n_players))
        .replace("__N_POSITIVE__", str(n_positive))
        .replace("__SIM_DATA__", json.dumps(sim_records))
        .replace("__SIM_SECTION__", sim_section_html)
        .replace("__DFS_SECTION__", dfs_section_html)
        .replace("__DFS_DATA__", dfs_json)
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    if open_browser:
        abs_path = os.path.abspath(output_path).replace("\\", "/")
        webbrowser.open(f"file:///{abs_path}")

    return html
