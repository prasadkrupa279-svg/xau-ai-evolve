"""
dashboard.py
============
Render web dashboard. ONE process does three things:
  1. loads data + computes 37 votes once,
  2. runs the evolution daemon in a background thread (24/7 generations),
  3. serves an honest dashboard + JSON API.

Env:
  DATA_DIR, MEMORY_PATH, PORT, GEN_SLEEP, ENABLE_PAPER (1/0),
  RENDER_EXTERNAL_URL (used for self keep-alive pings).
"""
from __future__ import annotations
import os, json, time, threading, traceback

from flask import Flask, jsonify, request
import requests

from data_tools import ensure_data
from indicators_lib import compute_votes
from evolution import EvolutionEngine
from live_tracker import PaperTracker
from realtime_alerter import start_alerter, build_feed

DATA_DIR = os.environ.get("DATA_DIR", "data")
MEMORY_PATH = os.environ.get("MEMORY_PATH", "memory/global_ai_memory.json")
ENABLE_PAPER = os.environ.get("ENABLE_PAPER", "0") == "1"
GEN_SLEEP = float(os.environ.get("GEN_SLEEP", "0.5"))

app = Flask(__name__)

# global runtime
RT = {
    "ready": False, "warming": "loading data...", "df_bars": 0, "data_source": None,
    "engine": None, "tracker": None, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
}
_last = {"gen": 0, "best_fit": 0.0, "mut_rate": 0.0, "stagnant": 0, "champion": None, "ts": None}


def _evolution_loop(df, votes):
    eng = EvolutionEngine(df, votes, memory_path=MEMORY_PATH)
    RT["engine"] = eng
    RT["ready"] = True
    RT["warming"] = None
    print(f"[dashboard] evolution ready. data={RT['data_source']} bars={len(df):,}", flush=True)
    if ENABLE_PAPER:
        RT["tracker"] = PaperTracker("memory/paper_results.json")
        start_alerter(eng.consensus_signal, RT["tracker"], feed=build_feed(float(df["close"].iloc[-1])))
        print("[dashboard] paper trading ENABLED.", flush=True)
    while True:
        try:
            r = eng.step()
            _last.update(gen=r["gen"], best_fit=r["best_fit"], mut_rate=r["mut_rate"],
                         stagnant=r["stagnant"], champion=r["champion"],
                         ts=time.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as e:
            print(f"[dashboard] gen error: {e}\n{traceback.format_exc()}", flush=True)
            time.sleep(5)
        time.sleep(GEN_SLEEP)


def _init():
    def worker():
        try:
            df, src = ensure_data(DATA_DIR)
            RT["df_bars"] = len(df); RT["data_source"] = src
            RT["warming"] = f"computing 37 indicators on {len(df):,} bars..."
            votes, _ = compute_votes(df)
            _evolution_loop(df, votes)
        except Exception as e:
            RT["warming"] = f"FATAL: {e}"
            print(f"[dashboard] init fatal: {e}\n{traceback.format_exc()}", flush=True)
    threading.Thread(target=worker, daemon=True, name="evolution").start()


def _keepalive():
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return
    url = url.rstrip("/")
    def loop():
        while True:
            try:
                requests.get(f"{url}/ping", timeout=20)
            except Exception:
                pass
            time.sleep(600)
    threading.Thread(target=loop, daemon=True, name="keepalive").start()


@app.route("/ping")
@app.route("/health")
def health():
    return jsonify({"ok": True, "ready": RT["ready"], "data_source": RT["data_source"]})


@app.route("/api/state")
def state():
    eng = RT["engine"]
    payload = {
        "ready": RT["ready"],
        "warming": RT["warming"],
        "started": RT["started"],
        "data_source": RT["data_source"],
        "bars": RT["df_bars"],
        "gen": _last["gen"],
        "best_fit": _last["best_fit"],
        "mut_rate": _last["mut_rate"],
        "stagnant": _last["stagnant"],
        "champion": _last["champion"],
        "leaderboard": (eng.leaderboard() if eng else []),
        "consensus": (eng.consensus_signal() if eng else {"signal": "flat"}),
        "paper": (RT["tracker"].stats() if RT["tracker"] else None),
        "updated": _last["ts"],
        "ts_now": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return jsonify(payload)


@app.route("/")
def index():
    return HTML


HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>XAUUSD AI Evolution — Dashboard</title>
<style>
:root{--bg:#0b1020;--c:#16203a;--c2:#1e2c4a;--t:#e6ecff;--m:#8ea0c8;--g:#22c55e;--b:#38bdf8;--w:#f59e0b;--r:#f87171;--bd:#283450}
*{box-sizing:border-box;margin:0;padding:0}body{font-family:'Segoe UI',system-ui,sans-serif;background:linear-gradient(160deg,#0b1020,#121a30);color:var(--t);padding:16px}
.wrap{max-width:1100px;margin:0 auto}header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:14px}
h1{font-size:1.25rem}h1 span{background:linear-gradient(90deg,var(--g),var(--b));-webkit-background-clip:text;background-clip:text;color:transparent}
.pill{font-size:.72rem;padding:4px 10px;border-radius:20px;background:var(--c2);color:var(--m)} .pill.live{color:var(--g)} .pill.warm{color:var(--w)}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}
@media(max-width:760px){.grid{grid-template-columns:repeat(2,1fr)}}
.card{background:var(--c);border:1px solid var(--bd);border-radius:12px;padding:14px}.card h3{font-size:.72rem;color:var(--m);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
.big{font-size:1.5rem;font-weight:800}.sub{font-size:.78rem;color:var(--m);margin-top:2px}
.champ{background:linear-gradient(135deg,rgba(34,197,94,.12),rgba(56,189,248,.08));border-color:var(--g)}
table{width:100%;border-collapse:collapse;font-size:.82rem}th,td{padding:7px 6px;text-align:left;border-bottom:1px solid var(--bd)}
th{color:var(--m);font-size:.68rem;text-transform:uppercase}.r{text-align:right}
.tag{display:inline-block;padding:1px 7px;border-radius:8px;font-size:.68rem;font-weight:700}
.buy{background:rgba(34,197,94,.18);color:var(--g)}.sell{background:rgba(248,113,113,.18);color:var(--r)}.flat{background:var(--c2);color:var(--m)}
.slA{color:var(--b)}.slB{color:var(--w)}
.disc{font-size:.72rem;color:var(--m);margin-top:14px;border-top:1px solid var(--bd);padding-top:10px;line-height:1.5}
.bar{height:6px;background:var(--c2);border-radius:4px;overflow:hidden;margin-top:4px}.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--g),var(--b))}
spin{display:inline-block;width:10px;height:10px;border:2px solid var(--m);border-top-color:var(--t);border-radius:50%;animation:s 1s linear infinite}@keyframes s{to{transform:rotate(360deg)}}
</style></head><body><div class="wrap">
<header><h1>🥇 <span>XAUUSD AI Evolution</span></h1><div><span id="src" class="pill">…</span> <span id="gen" class="pill">warming</span></div></header>

<div id="warm" class="card" style="margin-bottom:14px"><spin></spin> <span id="warmt">starting…</span></div>
<div id="main" style="display:none">
  <div class="grid">
    <div class="card champ"><h3>Champion winrate</h3><div class="big" id="c_wr">–</div><div class="sub">SL-guard · 1:5 RR · no trailing</div></div>
    <div class="card"><h3>Profit factor</h3><div class="big" id="c_pf">–</div><div class="sub" id="c_tr">– trades</div></div>
    <div class="card"><h3>Net (backtest)</h3><div class="big" id="c_net">–</div><div class="sub" id="c_sl">–</div></div>
    <div class="card"><h3>Consensus</h3><div class="big"><span id="cs" class="tag flat">FLAT</span></div><div class="sub" id="csd">–</div></div>
  </div>

  <div class="card" style="margin-bottom:14px"><h3>Evolution fitness</h3><div class="big" id="fit">–</div>
    <div class="bar"><i id="fitbar" style="width:0%"></i></div>
    <div class="sub" id="fitmeta">–</div></div>

  <div class="card">
    <h3>AI Agents Leaderboard (50/50 SL split: <span class="slA">blue=$0.5</span> / <span class="slB">amber=$1.0</span>)</h3>
    <table><thead><tr><th>#</th><th>key</th><th>SL</th><th>RR</th><th class="r">trades</th><th class="r">winrate</th><th class="r">PF</th><th class="r">net</th><th class="r">fit</th></tr></thead>
    <tbody id="lb"></tbody></table>
  </div>

  <div id="paperCard" class="card" style="margin-top:14px;display:none">
    <h3>Paper Trading (live feed, SL-guard)</h3>
    <div class="grid" style="margin:0"><div><div class="sub">trades</div><div class="big" id="p_tr">–</div></div>
    <div><div class="sub">winrate</div><div class="big" id="p_wr">–</div></div>
    <div><div class="sub">PF</div><div class="big" id="p_pf">–</div></div>
    <div><div class="sub">net</div><div class="big" id="p_net">–</div></div></div>
  </div>
</div>

<div class="disc">
<b>Honesty rules.</b> All numbers come from a real no-look-ahead backtest (closed-bar votes, fill at next open, SL-guard first).
1:5 RR makes ~16–34% winrate mathematically honest; <b>75%+ at 1:5 is impossible</b> and will never be claimed here.
Data source labelled clearly — <b>SYNTHETIC</b> until you drop real CSVs into <code>data/</code>. Paper/demo only; LIVE trading hard-blocked.
Not financial advice. Past performance ≠ future results. Trading is risky.
</div>
</div>
<script>
const $=id=>document.getElementById(id);
function fmt(n,d=2){return n==null?'–':Number(n).toLocaleString('en-IN',{maximumFractionDigits:d,minimumFractionDigits:d})}
async function tick(){
 try{
  const s=await (await fetch('/api/state')).json();
  $('src').textContent=(s.data_source||'?').toUpperCase()+' · '+ (s.bars?fmt(s.bars,0)+' bars':'');
  if(!s.ready){ $('warm').style.display='block'; $('warmt').textContent=s.warming||'warming…'; $('gen').textContent='warming'; $('gen').className='pill warm'; return; }
  $('warm').style.display='none'; $('main').style.display='block';
  $('gen').textContent='gen '+s.gen+' · fit '+s.best_fit; $('gen').className='pill live';
  const c=s.champion||{};
  $('c_wr').textContent=c.winrate==null?'–':(c.winrate*100).toFixed(2)+'%';
  $('c_pf').textContent=fmt(c.pf); $('c_tr').textContent=fmt(c.trades,0)+' trades';
  $('c_net').textContent='$'+fmt(c.net_profit); $('c_sl').textContent='SL $'+c.sl+' · 1:'+(c.rr||5);
  $('fit').textContent=s.best_fit; $('fitmeta').textContent='mut '+s.mut_rate+' · stagnant '+s.stagnant+' · '+s.updated;
  const maxf=Math.max(0.02,s.best_fit); $('fitbar').style.width=Math.min(100,s.best_fit/maxf*100)+'%';
  const cs=s.consensus||{}; $('cs').textContent=(cs.signal||'flat').toUpperCase();
  $('cs').className='tag '+(cs.signal||'flat'); $('csd').textContent='score '+cs.score+' · '+cs.agreement+' agree · '+cs.voters+' agents';
  const lb=$('lb'); lb.innerHTML='';
  (s.leaderboard||[]).forEach((a,i)=>{
   const sl=Number(a.sl); const cls=Math.abs(sl-0.5)<1e-6?'slA':'slB';
   const tr=document.createElement('tr');
   tr.innerHTML=`<td>${i+1}</td><td style="font-family:monospace">${a.key}</td><td class="${cls}">$${sl}</td><td>1:${a.rr}</td><td class="r">${fmt(a.trades,0)}</td><td class="r">${(a.winrate*100).toFixed(2)}%</td><td class="r">${fmt(a.pf)}</td><td class="r">$${fmt(a.net)}</td><td class="r">${a.fit}</td>`;
   lb.appendChild(tr);
  });
  if(s.paper){ $('paperCard').style.display='block';
   $('p_tr').textContent=fmt(s.paper.trades,0); $('p_wr').textContent=(s.paper.winrate*100).toFixed(1)+'%';
   $('p_pf').textContent=fmt(s.paper.pf); $('p_net').textContent='$'+fmt(s.paper.net); }
 }catch(e){console.warn(e)}
}
tick(); setInterval(tick,4000);
</script></body></html>"""

_init()
_keepalive()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
