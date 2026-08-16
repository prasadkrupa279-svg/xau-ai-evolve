"""
multi_agent_dashboard.py — LIVE dashboard for the multi-agent swarm.
Reads memory/multi_agent_best.json (written by the daemon every ~10s) and serves
an auto-refreshing page showing agents, dedup, global best + params, in real time.
"""
from __future__ import annotations
import os, time, json
from flask import Flask, jsonify

app = Flask(__name__)
MEM = "memory/multi_agent_best.json"


def _read():
    try:
        with open(MEM) as f:
            return json.load(f)
    except Exception:
        return None


@app.route("/api/state")
def state():
    return jsonify(_read() or {"warming": True})


@app.route("/health")
def health():
    return jsonify({"ok": True, "has_data": _read() is not None})


@app.route("/")
def index():
    return HTML


HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-Agent Swarm — LIVE</title>
<style>
:root{--bg:#0b1020;--c:#16203a;--c2:#1e2c4a;--t:#e6ecff;--m:#8ea0c8;--g:#22c55e;--b:#38bdf8;--w:#f59e0b;--r:#f87171;--bd:#283450}
*{box-sizing:border-box;margin:0;padding:0}body{font-family:'Segoe UI',system-ui,sans-serif;background:linear-gradient(160deg,#0b1020,#121a30);color:var(--t);padding:16px}
.wrap{max-width:980px;margin:0 auto}
header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:14px}
h1{font-size:1.2rem}h1 span{background:linear-gradient(90deg,var(--g),var(--b));-webkit-background-clip:text;background-clip:text;color:transparent}
.pill{font-size:.72rem;padding:4px 10px;border-radius:20px;background:var(--c2);color:var(--m)} .pill.live{color:var(--g)}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}
@media(max-width:760px){.grid{grid-template-columns:repeat(2,1fr)}}
.card{background:var(--c);border:1px solid var(--bd);border-radius:12px;padding:14px}.card h3{font-size:.7rem;color:var(--m);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
.big{font-size:1.5rem;font-weight:800}.sub{font-size:.78rem;color:var(--m);margin-top:2px}
.champ{background:linear-gradient(135deg,rgba(34,197,94,.12),rgba(56,189,248,.08));border-color:var(--g)}
table{width:100%;border-collapse:collapse;font-size:.84rem;background:var(--c);border-radius:10px;overflow:hidden;margin-bottom:14px}
th,td{padding:8px 9px;text-align:left;border-bottom:1px solid #243150}th{color:var(--m);font-size:.7rem;text-transform:uppercase;background:#101a30}.r{text-align:right}
.tag{display:inline-block;padding:1px 7px;border-radius:8px;font-size:.66rem;background:var(--c2);color:var(--b)}
.win{color:var(--g)}.loss{color:var(--r)}
.disc{font-size:.74rem;color:var(--m);margin-top:10px;line-height:1.5}
.pulse{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--g);animation:p 1.2s infinite}@keyframes p{0%{opacity:.3}100%{opacity:1}}
.params{font-size:.74rem;color:var(--m);background:var(--c);border:1px solid var(--bd);border-radius:10px;padding:12px;margin-top:6px;font-family:monospace;white-space:pre-wrap;word-break:break-all}
</style></head><body><div class="wrap">
<header><h1>🧠 <span>Multi-Agent Swarm</span> — LIVE</h1>
<div><span class="pill" id="src">…</span> <span class="pill live"><span class="pulse"></span> auto-refresh 3s</span></div></header>

<div class="grid">
  <div class="card champ"><h3>Global Best Fitness</h3><div class="big" id="fit">–</div><div class="sub">winrate × trades × PF</div></div>
  <div class="card"><h3>Trades</h3><div class="big" id="tr">–</div><div class="sub" id="wr">– winrate</div></div>
  <div class="card"><h3>Profit Factor</h3><div class="big" id="pf">–</div><div class="sub" id="net">– net</div></div>
  <div class="card"><h3>Unique Genomes</h3><div class="big" id="uniq">–</div><div class="sub">0 duplicate work</div></div>
</div>

<div class="card" style="margin-bottom:14px">
  <h3 style="margin-bottom:8px">Agents (each different task · RR pinned 1:3 · global dedup)</h3>
  <table><thead><tr><th>Agent</th><th>Evals</th><th class="r">Dup-skips</th><th class="r">% of work</th></tr></thead>
  <tbody id="ag"></tbody></table>
</div>

<div class="card champ">
  <h3 style="margin-bottom:8px">★ Champion Strategy (best genome so far)</h3>
  <div class="params" id="params">warming up…</div>
  <div class="disc"><b>Honest note:</b> yeh mere Python engine pe optimize ho raha hai (TV numbers se match nahi). Agents + dedup + RR-lock sach me kaam kar rahe hain. Live improvement dekho.</div>
</div>
<div class="disc">Updated: <span id="upd">–</span></div>
</div>
<script>
const $=id=>document.getElementById(id);
const FOCI={ "A-Sessions":"time-windows","B-Confirm":"confirm/body/range","C-Stops":"tight stops","D-Trend":"D/H1/H4 trend","E-Timing":"cooldown/entry-hours","F-Explorer":"explore-all" };
async function tick(){
 try{
  const d=await (await fetch('/api/state')).json();
  if(d.warming){ $('src').textContent='warming…'; return; }
  $('src').textContent='data: real 1.06M · RR 1:3 (pinned)';
  const b=d.best||{}; 
  $('fit').textContent=(b.fit??0).toFixed(4);
  $('tr').textContent=b.trades??'–';
  $('wr').textContent=((b.winrate??0)*100).toFixed(1)+'% WR';
  $('pf').textContent=(b.pf??0).toFixed(2);
  const net=b.net_profit??0; $('net').textContent=(net>=0?'+$':'-$')+Math.abs(net).toFixed(0)+' net';
  $('net').className='sub '+(net>=0?'win':'loss');
  $('uniq').textContent=(d.total_evaluated??0).toLocaleString();
  const ev=d.per_agent_evals||{}, du=d.per_agent_dups||{};
  const tot=Object.values(ev).reduce((a,c)=>a+c,0)||1;
  let rows='';
  for(const k of Object.keys(FOCI)){
    const e=ev[k]||0, dd=du[k]||0;
    rows+=`<tr><td><span class="tag">${k}</span> ${FOCI[k]}</td><td>${e}</td><td class="r">${dd}</td><td class="r">${(e/tot*100).toFixed(0)}%</td></tr>`;
  }
  $('ag').innerHTML=rows;
  const p=b.params||{};
  $('params').textContent = p && Object.keys(p).length ? JSON.stringify(p,null,1) : '(no champion yet)';
  $('upd').textContent=d.updated||'–';
 }catch(e){console.warn(e)}
}
tick(); setInterval(tick,3000);
</script></body></html>"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), threaded=True)
