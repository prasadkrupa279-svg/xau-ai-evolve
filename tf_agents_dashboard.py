"""
tf_agents_dashboard.py — LIVE dashboard: one row per AI agent (= per timeframe).
Shows: agent name, timeframe, RR, trades, winrate, PF, net — auto-refresh 3s.
Reads memory/tf_agents.json (written by tf_agents_daemon.py every ~10s).
"""
from __future__ import annotations
import os, json
from flask import Flask, jsonify

app = Flask(__name__)
MEM = "memory/tf_agents.json"


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
<title>AI Agents per Timeframe — LIVE</title>
<style>
:root{--bg:#0b1020;--c:#16203a;--c2:#1e2c4a;--t:#e6ecff;--m:#8ea0c8;--g:#22c55e;--b:#38bdf8;--w:#f59e0b;--r:#f87171;--bd:#283450}
*{box-sizing:border-box;margin:0;padding:0}body{font-family:'Segoe UI',system-ui,sans-serif;background:linear-gradient(160deg,#0b1020,#121a30);color:var(--t);padding:16px}
.wrap{max-width:1000px;margin:0 auto}
header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:14px}
h1{font-size:1.2rem}h1 span{background:linear-gradient(90deg,var(--g),var(--b));-webkit-background-clip:text;background-clip:text;color:transparent}
.pill{font-size:.72rem;padding:4px 10px;border-radius:20px;background:var(--c2);color:var(--m)} .pill.live{color:var(--g)}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}
@media(max-width:760px){.cards{grid-template-columns:repeat(2,1fr)}}
.card{background:var(--c);border:1px solid var(--bd);border-radius:12px;padding:13px}.card h3{font-size:.66rem;color:var(--m);text-transform:uppercase;margin-bottom:5px}.big{font-size:1.35rem;font-weight:800}
table{width:100%;border-collapse:collapse;font-size:.85rem;background:var(--c);border-radius:10px;overflow:hidden}
th,td{padding:9px 10px;text-align:left;border-bottom:1px solid #243150}th{color:var(--m);font-size:.66rem;text-transform:uppercase;background:#101a30;position:sticky;top:0}.r{text-align:right}
.name{font-weight:700;color:var(--b)}.tf{font-family:monospace;color:var(--w)}
.win{color:var(--g)}.loss{color:var(--r)}.pfok{color:var(--g)}.pfbad{color:var(--r)}
.spin{display:inline-block;width:9px;height:9px;border:2px solid var(--m);border-top-color:var(--t);border-radius:50%;animation:s 1s linear infinite;vertical-align:middle}@keyframes s{to{transform:rotate(360deg)}}
.pulse{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--g);animation:p 1.2s infinite;margin-right:4px}@keyframes p{0%{opacity:.3}100%{opacity:1}}
.disc{font-size:.74rem;color:var(--m);margin-top:12px;line-height:1.5}
</style></head><body><div class="wrap">
<header><h1>🤖 <span>AI Agents</span> — per Timeframe</h1>
<div><span class="pill" id="src">…</span> <span class="pill live"><span class="pulse"></span>auto-refresh 3s</span></div></header>

<div class="cards">
  <div class="card"><h3>Agents online</h3><div class="big" id="n">–</div></div>
  <div class="card"><h3>Total trades (sum)</h3><div class="big" id="tt">–</div></div>
  <div class="card"><h3>Best winrate</h3><div class="big" id="bwr">–</div></div>
  <div class="card"><h3>Total net (sum)</h3><div class="big" id="tnet">–</div></div>
</div>

<table><thead><tr>
  <th>Agent name</th><th>Timeframe</th><th>RR</th><th class="r">Trades</th>
  <th>Win rate</th><th class="r">PF</th><th class="r">Net</th><th class="r">Evals</th>
</tr></thead><tbody id="rows"><tr><td colspan="8"><span class="spin"></span> warming up…</td></tr></tbody></table>

<div class="disc"><b>Honest note:</b> har agent apne TF pe apna strategy evolve kar raha hai (RR original pinned). Numbers mere Python engine ke hain (TV se match nahi) — lekin agents + live improvement sach me chal raha hai.</div>
<div class="disc">Updated: <span id="upd">–</span></div>
</div>
<script>
const $=id=>document.getElementById(id);
const ORDER=["1m","3m","5m","15m","30m","45m","1H","2H","3H","4H","1D"];
async function tick(){
 try{
  const d=await (await fetch('/api/state')).json();
  if(d.warming){return;}
  $('src').textContent='real 1.06M data · RR original per TF';
  const A=d.agents||{};
  const keys=ORDER.filter(k=>A[k]);
  $('n').textContent=(d.total_agents||keys.length)+' agents';
  let tt=0,bwr=0,tnet=0;
  keys.forEach(k=>{tt+=A[k].trades||0; bwr=Math.max(bwr,A[k].winrate||0); tnet+=A[k].net_profit||0;});
  $('tt').textContent=tt.toLocaleString();
  $('bwr').textContent=(bwr*100).toFixed(1)+'%';
  $('tnet').textContent=(tnet>=0?'+$':'-$')+Math.abs(tnet).toFixed(0);
  $('tnet').className='big '+(tnet>=0?'win':'loss');
  let rows='';
  keys.forEach(k=>{
    const a=A[k]; const wr=(a.winrate*100).toFixed(1)+'%';
    const pf=a.pf; const net=a.net_profit;
    rows+=`<tr>
      <td class="name">${a.name}</td><td class="tf">${a.tf}</td><td>1:${a.rr}</td>
      <td class="r">${a.trades}</td><td>${wr}</td>
      <td class="r ${pf>=1?'pfok':'pfbad'}">${pf.toFixed(2)}</td>
      <td class="r ${net>=0?'win':'loss'}">${net>=0?'+':'-'}$${Math.abs(net).toFixed(0)}</td>
      <td class="r">${a.evals}</td></tr>`;
  });
  $('rows').innerHTML=rows||'<tr><td colspan=8>gathering…</td></tr>';
  $('upd').textContent=d.updated||'–';
 }catch(e){console.warn(e)}
}
tick(); setInterval(tick,3000);
</script></body></html>"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), threaded=True)
