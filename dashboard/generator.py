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
    # Simulation columns — must NOT appear as book columns in the EV table
    "sim_prob", "sim_edge", "convergence",
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
    .anchor-pin{color:#0d6efd;font-size:.78em;font-weight:600}
    .anchor-bol{color:#fd7e14;font-size:.78em;font-weight:600}
    #parlay-builder{margin-top:32px;background:#fff;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,.1);border-radius:4px}
    #parlay-builder h2{margin-top:0}
    .parlay-leg{font-size:.95em;margin:2px 0}
    #parlay-stats{margin-top:12px;line-height:1.8}
    .stat-label{font-weight:600}
    #suggested-parlays{margin-top:32px;background:#fff;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,.1);border-radius:4px}
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
    #sim-section{margin-top:32px;background:#fff;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,.1);border-radius:4px}
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
      <th onclick="sortBy('team')">Team</th>
      <th onclick="sortBy('time_sort')">Time (ET)</th>
      <th onclick="sortBy('pinnacle_pct')">Pin %</th>
      <th onclick="sortBy('sharp_anchor')">Anchor</th>
      __BOOK_HEADERS__
      <th onclick="sortBy('best_retail_odds')">Best Retail</th>
      <th onclick="sortBy('ev_pct')">EV%</th>
      <th onclick="sortBy('stake_units')">Stake</th>
      <th onclick="sortBy('composite_z')">Composite Z</th>
    </tr></thead>
    <tbody id="table-body"></tbody>
  </table>
  __SIM_SECTION__
  <div id="suggested-parlays">
    <h2>Suggested Longshot Parlays</h2>
    <p style="color:#6c757d;font-size:.9em">Same-book &middot; +EV legs at +500 to +1500 &middot; no same-game &middot; ranked by combined EV</p>
    <div id="parlay-cards"></div>
  </div>
  <div id="parlay-builder">
    <h2>Manual Parlay Builder</h2>
    <div id="parlay-legs"><em style="color:#999">Select players above to build a parlay</em></div>
    <div id="parlay-stats"></div>
  </div>
  <script>
    const DATA=__DATA__;
    const SIM_DATA=__SIM_DATA__;
    let simSortKey='sim_edge',simSortDir=-1;
    const BOOKS=__BOOK_NAMES__;
    const PARLAYS=__PARLAYS__;
    let sortKey='composite_z',sortDir=-1,minEv=-100;
    const legs={};
    function fmtOdds(v){return v==null?'--':v>0?'+'+v:''+v}
    function fmtPct(v){return(v>=0?'+':'')+v.toFixed(2)+'%'}
    function fmtZ(v){return(v>=0?'+':'')+v.toFixed(2)}
    function impliedPct(v){if(v==null)return'--';const d=v>0?(v/100)+1:(100/Math.abs(v))+1;return(100/d).toFixed(1)+'%'}
    function fmtBook(v){if(v==null)return'<td>--</td>';return`<td>${fmtOdds(v)}<span class="odds-pct">${impliedPct(v)}</span></td>`}
    function fmtAnchor(a){
      if(!a||a==='pinnacle')return'<span class="anchor-pin">PIN</span>';
      return'<span class="anchor-bol">BOL</span>';
    }
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
          <td>${fmtAnchor(r.sharp_anchor)}</td>
          ${BOOKS.map(b=>fmtBook(r[b])).join('')}
          <td>${fmtOdds(r.best_retail_odds)}</td>
          <td>${fmtPct(r.ev_pct)}</td>
          <td>${r.stake}</td>
          <td>${fmtZ(r.composite_z)}</td>
        </tr>`).join('');
    }
    function sortSim(k){if(simSortKey===k)simSortDir*=-1;else{simSortKey=k;simSortDir=-1;}renderSimTable();}
    function renderSimTable(){
      const tbody=document.getElementById('sim-table-body');
      if(!tbody)return;
      if(!SIM_DATA||!SIM_DATA.length){
        if(tbody)tbody.innerHTML='<tr><td colspan="10" style="color:#999;font-style:italic;text-align:center">No simulation data.</td></tr>';
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
        return`<tr class="${cls}">
          <td>${r.player}</td><td>${r.team}</td><td>${r.game}</td>
          <td>${r.sim_prob.toFixed(1)}%</td><td>${r.pin_prob.toFixed(1)}%</td>
          <td>${edgeStr}</td><td>${convBadge}</td>
          <td>${fmtOdds(r.best_retail_odds)}</td><td>${fmtPct(r.ev_pct)}</td>
          <td>${r.stake}</td>
        </tr>`;
      }).join('');
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
      const gameCounts={};
      sel.forEach(l=>{if(l.game)gameCounts[l.game]=(gameCounts[l.game]||0)+1});
      const dupGames=Object.entries(gameCounts).filter(e=>e[1]>=2);
      let warn='';
      if(dupGames.length){
        const desc=dupGames.map(e=>`${e[1]} legs in ${e[0]}`).join('; ');
        warn=`<div style="background:#fff3cd;border:1px solid #ffc107;padding:8px 10px;border-radius:4px;margin-bottom:8px;color:#856404"><strong>&#9888; Same-game correlation:</strong> ${desc}. HR outcomes within one game are correlated; the combined probability and EV below assume independence and understate the true risk on these legs.</div>`;
      }
      statsDiv.innerHTML=warn+`
        <div><span class="stat-label">Legs:</span> ${sel.length}</div>
        <div><span class="stat-label">Combined Probability:</span> ${(pProb*100).toFixed(2)}%</div>
        <div><span class="stat-label">Combined Odds:</span> ${decimalToAmerican(pDec)}</div>
        <div><span class="stat-label">Combined EV%:</span> ${fmtPct(pEv*100)}</div>
        <div><span class="stat-label">Composite Z:</span> ${pZ}</div>`;
    }
    function renderParlays(){
      const div=document.getElementById('parlay-cards');
      if(!PARLAYS||!PARLAYS.length){
        div.innerHTML='<p style="color:#999;font-style:italic">No qualifying same-book parlays for today (need 3+ +EV legs at +500-+1500 within one sportsbook, across different games).</p>';
        return;
      }
      div.innerHTML=PARLAYS.map((p,i)=>{
        const cls=[
          'parlay-card',
          p.all_same_book?'all-same-book':'',
          p.has_betonline_anchor?'has-bol':'',
        ].filter(Boolean).join(' ');
        const book=p.book||(p.books&&p.books[0])||'?';
        const bolFlag=p.has_betonline_anchor?'<span class="flag-bol">BOL anchor</span>':'';
        const legsHtml=p.legs.map((leg,j)=>{
          const o=p.leg_odds[j];
          const odds_str=o>0?'+'+o:''+o;
          return`<li>${leg} &nbsp;<strong>${odds_str}</strong><em style="color:#6c757d"> (EV ${p.leg_ev_pcts[j]>=0?'+':''}${p.leg_ev_pcts[j]}%)</em></li>`;
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
    renderTable();
    renderParlays();
    renderSimTable();
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
      <th onclick="sortSim('pin_prob')">Pin %</th>
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


def generate_dashboard(
    final_df: pd.DataFrame,
    output_path: str = "hr_dashboard.html",
    open_browser: bool = True,
    *,
    parlays: list | None = None,
) -> None:
    book_cols = [c for c in final_df.columns if c not in META_COLS]

    records = []
    for _, row in final_df.iterrows():
        record = {
            "player": row["player_name"],
            "team": row.get("team", ""),
            "game": row.get("game", ""),
            "time": row["commence_time"].astimezone(ET).strftime("%I:%M %p ET"),
            "time_sort": row["commence_time"].timestamp(),
            "pinnacle_pct": round(row["pinnacle_prob"] * 100, 2),
            "sharp_anchor": row.get("sharp_anchor", "pinnacle"),
            "best_retail_odds": int(row["best_retail_odds"]),
            "ev_pct": round(row["ev_pct"] * 100, 2),
            "stake_units": round(row["kelly_units"], 1),
            "stake": (
                f'{row["kelly_units"]:g}u (${row["stake_usd"]:,.0f})'
                if row["kelly_units"] > 0 else "0u"
            ),
            "composite_z": round(row["composite_z"], 2),
        }
        for col in book_cols:
            val = row.get(col)
            record[col] = int(val) if pd.notna(val) else None
        records.append(record)

    # Build simulation section
    if _SIM_COLS.issubset(final_df.columns) and final_df["sim_prob"].notna().any():
        sim_records = []
        for _, row in final_df.iterrows():
            if pd.isna(row.get("sim_prob")):
                continue
            sim_records.append({
                "player": row["player_name"],
                "team": row.get("team", ""),
                "game": row.get("game", ""),
                "sim_prob": round(float(row["sim_prob"]) * 100, 1),
                "pin_prob": round(float(row["pinnacle_prob"]) * 100, 1),
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

    timestamp = datetime.now(ET).strftime("%Y-%m-%d %I:%M %p ET")
    n_players = len(final_df)
    n_positive = int((final_df["ev_pct"] > 0).sum())
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
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    if open_browser:
        abs_path = os.path.abspath(output_path).replace("\\", "/")
        webbrowser.open(f"file:///{abs_path}")
