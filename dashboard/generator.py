# dashboard/generator.py
import json
import os
import webbrowser
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")

META_COLS = {
    "player_name", "game", "commence_time",
    "pinnacle_odds", "pinnacle_prob",
    "best_retail_odds", "best_retail_decimal",
    "ev_pct", "composite_score", "composite_z",
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>HR Dashboard</title>
  <style>
    body{font-family:system-ui,sans-serif;padding:20px;background:#f8f9fa;margin:0}
    h1{color:#222;margin-bottom:4px}
    .meta{color:#666;font-size:.9em;margin-bottom:16px}
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
    #parlay-builder{margin-top:32px;background:#fff;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,.1);border-radius:4px}
    #parlay-builder h2{margin-top:0}
    .parlay-leg{font-size:.95em;margin:2px 0}
    #parlay-stats{margin-top:12px;line-height:1.8}
    .stat-label{font-weight:600}
  </style>
</head>
<body>
  <h1>Home Run Dashboard</h1>
  <p class="meta">__N_PLAYERS__ players &nbsp;|&nbsp; __N_POSITIVE__ +EV plays &nbsp;|&nbsp; __TIMESTAMP__</p>
  <div class="controls">
    <label>Min EV%: <input type="range" id="ev-filter" min="-100" max="100" value="-100" step="1" oninput="applyFilter(+this.value)"></label>
    <span id="ev-label">-100%</span>
  </div>
  <table id="player-table">
    <thead><tr>
      <th></th>
      <th onclick="sortBy('player')">Player</th>
      <th onclick="sortBy('game')">Game</th>
      <th onclick="sortBy('time_sort')">Time (ET)</th>
      <th onclick="sortBy('pinnacle_pct')">Pin %</th>
      __BOOK_HEADERS__
      <th onclick="sortBy('best_retail_odds')">Best Retail</th>
      <th onclick="sortBy('ev_pct')">EV%</th>
      <th onclick="sortBy('composite_z')">Composite Z</th>
    </tr></thead>
    <tbody id="table-body"></tbody>
  </table>
  <div id="parlay-builder">
    <h2>Parlay Builder</h2>
    <div id="parlay-legs"><em style="color:#999">Select players above to build a parlay</em></div>
    <div id="parlay-stats"></div>
  </div>
  <script>
    const DATA=__DATA__;
    const BOOKS=__BOOK_NAMES__;
    let sortKey='composite_z',sortDir=-1,minEv=-100;
    const legs={};
    function fmtOdds(v){return v==null?'--':v>0?'+'+v:''+v}
    function fmtPct(v){return(v>=0?'+':'')+v.toFixed(2)+'%'}
    function fmtZ(v){return(v>=0?'+':'')+v.toFixed(2)}
    function impliedPct(v){if(v==null)return'--';const d=v>0?(v/100)+1:(100/Math.abs(v))+1;return(100/d).toFixed(1)+'%'}
    function fmtBook(v){if(v==null)return'<td>--</td>';return`<td>${fmtOdds(v)}<span class="odds-pct">${impliedPct(v)}</span></td>`}
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
          <td>${r.player}</td><td>${r.game}</td><td>${r.time}</td>
          <td>${r.pinnacle_pct.toFixed(1)}%</td>
          ${BOOKS.map(b=>fmtBook(r[b])).join('')}
          <td>${fmtOdds(r.best_retail_odds)}</td>
          <td>${fmtPct(r.ev_pct)}</td>
          <td>${fmtZ(r.composite_z)}</td>
        </tr>`).join('');
    }
    function sortBy(k){if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=-1;}renderTable();}
    function applyFilter(v){minEv=v;document.getElementById('ev-label').textContent=v+'%';renderTable();}
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
        legsDiv.innerHTML='<em style="color:#999">Select players above to build a parlay</em>';
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
      statsDiv.innerHTML=`
        <div><span class="stat-label">Legs:</span> ${sel.length}</div>
        <div><span class="stat-label">Combined Probability:</span> ${(pProb*100).toFixed(2)}%</div>
        <div><span class="stat-label">Combined Odds:</span> ${decimalToAmerican(pDec)}</div>
        <div><span class="stat-label">Combined EV%:</span> ${fmtPct(pEv*100)}</div>
        <div><span class="stat-label">Composite Z:</span> ${pZ}</div>`;
    }
    renderTable();
  </script>
</body>
</html>"""


def generate_dashboard(
    final_df: pd.DataFrame,
    output_path: str = "hr_dashboard.html",
    open_browser: bool = True,
) -> None:
    book_cols = [c for c in final_df.columns if c not in META_COLS]

    records = []
    for _, row in final_df.iterrows():
        record = {
            "player": row["player_name"],
            "game": row["game"],
            "time": row["commence_time"].astimezone(ET).strftime("%I:%M %p ET"),
            "time_sort": row["commence_time"].timestamp(),
            "pinnacle_pct": round(row["pinnacle_prob"] * 100, 2),
            "best_retail_odds": int(row["best_retail_odds"]),
            "ev_pct": round(row["ev_pct"] * 100, 2),
            "composite_z": round(row["composite_z"], 2),
        }
        for col in book_cols:
            val = row.get(col)
            record[col] = int(val) if pd.notna(val) else None
        records.append(record)

    timestamp = datetime.now(ET).strftime("%Y-%m-%d %I:%M %p ET")
    n_players = len(final_df)
    n_positive = int((final_df["ev_pct"] > 0).sum())
    book_headers = "".join(f'<th onclick="sortBy(\'{b}\')">{b}</th>' for b in book_cols)

    html = (
        HTML_TEMPLATE
        .replace("__DATA__", json.dumps(records))
        .replace("__BOOK_NAMES__", json.dumps(book_cols))
        .replace("__BOOK_HEADERS__", book_headers)
        .replace("__TIMESTAMP__", timestamp)
        .replace("__N_PLAYERS__", str(n_players))
        .replace("__N_POSITIVE__", str(n_positive))
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    if open_browser:
        abs_path = os.path.abspath(output_path).replace("\\", "/")
        webbrowser.open(f"file:///{abs_path}")
