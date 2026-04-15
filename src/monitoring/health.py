"""
FastAPI app exposing /health, /ready, /metrics, and /ml/* endpoints.

/health         always 200 if the process is alive (used by Docker HEALTHCHECK)
/ready          200 only if WS feeds are connected, DB is reachable, and not halted
/metrics        Prometheus exposition format
/ml/dashboard   Browser dashboard — live signal decisions + model stats
/ml/stats       JSON: model version, sample count, win rate, feature importances
/ml/decisions   JSON: last 50 ML decisions from the database
/ml/stream      Server-Sent Events: one event per new ML decision (live feed)
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, AsyncIterator

import uvicorn
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.monitoring.prom_metrics import REGISTRY
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.data.storage import Storage
    from src.exchanges.base import ExchangeConnector
    from src.ml.model import MLSignalFilter
    from src.risk.kill_switch import KillSwitch

log = get_logger(__name__)


class LiveBroadcaster:
    """Push real-time events to all connected WebSocket clients."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[str]] = set()

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        self._subscribers.discard(q)

    def push(self, event: dict) -> None:
        """Non-blocking push to all subscribers. Full queues are dropped."""
        payload = json.dumps(event)
        dead: set[asyncio.Queue[str]] = set()
        for q in self._subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.add(q)
        self._subscribers -= dead

# ─────────────────────── inline dashboard HTML ───────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cryprobotik — Dashboard</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #c9d1d9; }
h1 { padding: 14px 24px; font-size: 1.1rem; border-bottom: 1px solid #21262d;
     color: #58a6ff; letter-spacing: 1px; display: flex; align-items: center; gap: 8px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#3fb950; flex-shrink:0; }
.page { padding: 14px 24px; display: flex; flex-direction: column; gap: 14px; }

/* ── P&L banner ── */
.pnl-banner { background:#161b22; border:1px solid #21262d; border-radius:8px;
              display:flex; flex-wrap:wrap; gap:0; }
.pnl-cell { flex:1; min-width:110px; padding:10px 16px; border-right:1px solid #21262d;
            display:flex; flex-direction:column; gap:3px; }
.pnl-cell:last-child { border-right:none; }
.pnl-label { font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:.8px; }
.pnl-val { font-size:1.05rem; font-weight:700; color:#58a6ff; }
.pos { color:#3fb950 !important; }
.neg { color:#f85149 !important; }
.warn { color:#d29922 !important; }

/* ── Two column layout ── */
.cols { display:grid; grid-template-columns: 320px 1fr; gap:14px; }

/* ── Cards ── */
.card { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:14px; }
.card h2 { font-size:0.78rem; text-transform:uppercase; letter-spacing:1px;
           color:#8b949e; margin-bottom:10px; }
.stat-row { display:flex; justify-content:space-between; padding:4px 0;
            font-size:0.87rem; border-bottom:1px solid #21262d; }
.stat-row:last-child { border-bottom:none; }
.sv { color:#58a6ff; font-weight:600; }

/* ── Tables ── */
table { width:100%; border-collapse:collapse; font-size:0.8rem; }
th { text-align:left; padding:5px 8px; color:#8b949e; border-bottom:1px solid #21262d; font-weight:500; }
td { padding:4px 8px; border-bottom:1px solid #0d1117; }
tr:hover td { background:#1c2128; }
tbody tr:first-child td { animation: flash .35s ease; }
@keyframes flash { from { background:#1c3a5e; } to { background:transparent; } }

/* ── Feature bars ── */
.bar-row { display:flex; align-items:center; gap:6px; margin-bottom:3px; font-size:0.76rem; }
.bar-lbl { width:120px; color:#8b949e; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.bar-track { flex:1; background:#21262d; border-radius:2px; height:8px; }
.bar-fill { height:8px; border-radius:2px; background:#58a6ff; transition:width .4s; }
.bar-pct { width:40px; text-align:right; color:#c9d1d9; }

/* ── Equity mini-chart ── */
#equity-canvas { width:100%; height:60px; display:block; margin-top:8px; }
</style>
</head>
<body>
<h1><span class="dot" id="dot"></span>Cryprobotik — Live Trading Dashboard</h1>

<div class="page">

  <!-- P&L Banner -->
  <div class="pnl-banner">
    <div class="pnl-cell">
      <span class="pnl-label">Balance</span>
      <span class="pnl-val" id="b-balance">—</span>
    </div>
    <div class="pnl-cell">
      <span class="pnl-label">Total P&amp;L</span>
      <span class="pnl-val" id="b-pnl">—</span>
    </div>
    <div class="pnl-cell">
      <span class="pnl-label">Fees</span>
      <span class="pnl-val" id="b-fees">—</span>
    </div>
    <div class="pnl-cell">
      <span class="pnl-label">Trades</span>
      <span class="pnl-val" id="b-trades">—</span>
    </div>
    <div class="pnl-cell">
      <span class="pnl-label">Win Rate</span>
      <span class="pnl-val" id="b-wr">—</span>
    </div>
    <div class="pnl-cell">
      <span class="pnl-label">Regime</span>
      <span class="pnl-val" id="b-regime">—</span>
    </div>
    <div class="pnl-cell">
      <span class="pnl-label">ML Status</span>
      <span class="pnl-val" id="b-ml">—</span>
    </div>
  </div>

  <!-- Two columns -->
  <div class="cols">

    <!-- Left: Model + Features -->
    <div style="display:flex;flex-direction:column;gap:14px">
      <div class="card">
        <h2>ML Model</h2>
        <div id="ml-stats"><div class="stat-row"><span>Status</span><span class="sv warn">Loading…</span></div></div>
      </div>
      <div class="card">
        <h2>Feature Importances</h2>
        <div id="imp-container"><span style="color:#8b949e;font-size:0.8rem">Training in progress…</span></div>
      </div>
      <div class="card">
        <h2>Equity Curve</h2>
        <canvas id="equity-canvas"></canvas>
        <div id="equity-info" style="font-size:0.75rem;color:#8b949e;margin-top:4px">No trades yet</div>
      </div>
    </div>

    <!-- Right: Signal feed + Trade history -->
    <div style="display:flex;flex-direction:column;gap:14px">
      <div class="card">
        <h2>Live Signal Feed <span id="feed-count" style="color:#8b949e;font-weight:normal;font-size:0.8rem"></span></h2>
        <table>
          <thead><tr>
            <th>Time</th><th>Symbol</th><th>Side</th>
            <th>Conf</th><th>Score</th><th>Verdict</th><th>Regime</th>
          </tr></thead>
          <tbody id="feed-body"></tbody>
        </table>
      </div>
      <div class="card">
        <h2>Trade History <span id="trade-count" style="color:#8b949e;font-weight:normal;font-size:0.8rem"></span></h2>
        <table>
          <thead><tr>
            <th>Time</th><th>Symbol</th><th>Dir</th><th>Exit Price</th><th>PnL</th><th>Fee</th>
          </tr></thead>
          <tbody id="trade-body"></tbody>
        </table>
      </div>
    </div>

  </div><!-- /cols -->
</div><!-- /page -->

<script>
'use strict';
const MAX_FEED = 60;
let feedCount = 0;
let equityPoints = [10000];

// ────────────────────────────────────────────
// P&L Banner
// ────────────────────────────────────────────
async function refreshPnL() {
  try {
    const [statsRes, mlRes] = await Promise.all([
      fetch('/trading/stats'), fetch('/ml/stats')
    ]);
    const s = await statsRes.json();
    const m = await mlRes.json();

    // Banner
    setText('b-balance', '$' + fmt2(s.paper_balance || 10000));
    const pnl = s.total_pnl || 0;
    const pnlEl = document.getElementById('b-pnl');
    pnlEl.textContent = (pnl >= 0 ? '+' : '') + '$' + fmt2(Math.abs(pnl));
    pnlEl.className = 'pnl-val ' + (pnl > 0 ? 'pos' : pnl < 0 ? 'neg' : '');
    setText('b-fees', '-$' + fmt2(s.total_fees || 0));
    setText('b-trades', (s.closed_trades || 0) + ' / ' + (s.total_fills || 0));
    const wr = s.win_rate;
    const wrEl = document.getElementById('b-wr');
    wrEl.textContent = wr != null ? (wr * 100).toFixed(1) + '%' : '—';
    wrEl.className = 'pnl-val ' + (wr != null ? (wr >= 0.5 ? 'pos' : 'neg') : '');
    const reg = s.regimes_seen && s.regimes_seen[0] ? s.regimes_seen[0].regime.replace(/_/g,' ') : '—';
    setText('b-regime', reg);

    // ML status in banner
    const mlEl = document.getElementById('b-ml');
    if (m.cold_start) {
      mlEl.textContent = 'COLD (' + m.n_samples + '/10)';
      mlEl.className = 'pnl-val warn';
    } else {
      mlEl.textContent = 'v' + m.model_version + ' (' + (m.n_samples || 0) + ' samples)';
      mlEl.className = 'pnl-val pos';
    }

    // Dot color
    document.getElementById('dot').style.background = '#3fb950';

    // Render ML stats
    renderML(m);

    // Update equity
    if (s.paper_balance) drawEquity(s);
  } catch(e) {
    document.getElementById('dot').style.background = '#f85149';
  }
}

// ────────────────────────────────────────────
// ML Model card
// ────────────────────────────────────────────
function renderML(d) {
  const wr = d.win_rate != null ? (d.win_rate * 100).toFixed(1) + '%' : '—';
  document.getElementById('ml-stats').innerHTML =
    row('Status', d.cold_start ? '<span class="sv warn">COLD START</span>' : '<span class="sv pos">TRAINED</span>') +
    row('Model version', 'v' + d.model_version) +
    row('Samples', d.n_samples + ' (need ' + (d.cold_start ? 10 - d.n_samples : 'more') + ')') +
    row('Profitable', d.n_profitable + ' / ' + d.n_samples) +
    row('Win rate', wr) +
    row('Threshold', (d.accept_threshold * 100).toFixed(0) + '%') +
    row('Open positions', d.pending_positions);

  if (d.feature_importances) {
    const entries = Object.entries(d.feature_importances).sort((a,b)=>b[1]-a[1]).slice(0,10);
    const max = entries[0] ? entries[0][1] : 1;
    document.getElementById('imp-container').innerHTML = entries.map(([k,v]) =>
      '<div class="bar-row">' +
      '<span class="bar-lbl" title="' + k + '">' + k + '</span>' +
      '<div class="bar-track"><div class="bar-fill" style="width:' + (v/max*100).toFixed(1) + '%"></div></div>' +
      '<span class="bar-pct">' + (v*100).toFixed(1) + '%</span>' +
      '</div>'
    ).join('');
  }
}

// ────────────────────────────────────────────
// Trade History
// ────────────────────────────────────────────
async function loadTrades() {
  try {
    const r = await fetch('/trading/fills');
    const fills = await r.json();
    const closed = fills.filter(f => f.realized_pnl != null && f.realized_pnl !== 0);
    document.getElementById('trade-count').textContent = '(' + closed.length + ' closed trades)';
    const tbody = document.getElementById('trade-body');
    tbody.innerHTML = closed.slice(0, 30).map(f => {
      const pnl = f.realized_pnl;
      const cls = pnl > 0 ? 'pos' : pnl < 0 ? 'neg' : '';
      const side = f.side === 'buy' ? 'CLOSE↑' : 'CLOSE↓';
      const sideColor = f.side === 'buy' ? '#3fb950' : '#f85149';
      return '<tr>' +
        '<td>' + fmtTime(f.ts) + '</td>' +
        '<td>' + f.symbol.split('/')[0] + '</td>' +
        '<td style="color:' + sideColor + '">' + side + '</td>' +
        '<td>' + parseFloat(f.price).toFixed(4) + '</td>' +
        '<td class="' + cls + '">' + (pnl >= 0 ? '+' : '') + '$' + fmt2(Math.abs(pnl)) + '</td>' +
        '<td style="color:#8b949e">$' + fmt2(f.fee) + '</td>' +
        '</tr>';
    }).join('');
  } catch(e) {}
}

// ────────────────────────────────────────────
// Equity mini-chart
// ────────────────────────────────────────────
function drawEquity(s) {
  // Rebuild from fills history on each call — simple running sum
  const canvas = document.getElementById('equity-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || 400;
  const H = 60;
  canvas.width = W;
  canvas.height = H;

  // Use s.paper_balance as the latest point
  const latest = s.paper_balance || 10000;
  if (equityPoints[equityPoints.length - 1] !== latest) {
    equityPoints.push(latest);
    if (equityPoints.length > 200) equityPoints.shift();
  }
  const pts = equityPoints;
  const mn = Math.min(...pts) - 50;
  const mx = Math.max(10000, Math.max(...pts)) + 50;

  ctx.clearRect(0, 0, W, H);
  // Grid line at 10000
  const base = H - ((10000 - mn) / (mx - mn)) * H;
  ctx.strokeStyle = '#21262d';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(0, base); ctx.lineTo(W, base); ctx.stroke();
  ctx.setLineDash([]);

  // Equity line
  ctx.strokeStyle = latest >= 10000 ? '#3fb950' : '#f85149';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = (i / (pts.length - 1 || 1)) * W;
    const y = H - ((p - mn) / (mx - mn)) * H;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  const pct = ((latest - 10000) / 10000 * 100).toFixed(2);
  document.getElementById('equity-info').textContent =
    '$' + fmt2(latest) + '  (' + (latest >= 10000 ? '+' : '') + pct + '%)  ' +
    pts.length + ' data points';
}

// ────────────────────────────────────────────
// SSE live signal feed
// ────────────────────────────────────────────
function startSSE() {
  const es = new EventSource('/ml/stream');
  es.onopen = () => document.getElementById('dot').style.background = '#3fb950';
  es.onmessage = e => {
    try { addFeedRow(JSON.parse(e.data)); } catch(_) {}
  };
  es.onerror = () => {
    document.getElementById('dot').style.background = '#d29922';
    es.close();
    setTimeout(startSSE, 4000);
  };
}

function addFeedRow(d) {
  if (!d.symbol) return;
  feedCount++;
  document.getElementById('feed-count').textContent = '(' + feedCount + ' total)';
  const tbody = document.getElementById('feed-body');
  const verdict = d.cold_start
    ? '<span class="warn">COLD</span>'
    : d.accepted
      ? '<span class="pos">✓ PASS</span>'
      : '<span class="neg">✗ SKIP</span>';
  const tr = document.createElement('tr');
  tr.innerHTML =
    '<td>' + fmtTime(d.ts) + '</td>' +
    '<td>' + (d.symbol || '').split('/')[0] + '</td>' +
    '<td style="color:' + (d.side==='buy'?'#3fb950':'#f85149') + '">' + (d.side||'').toUpperCase() + '</td>' +
    '<td>' + ((d.confidence||0)*100).toFixed(0) + '%</td>' +
    '<td>' + ((d.ml_score||0)*100).toFixed(0) + '%</td>' +
    '<td>' + verdict + '</td>' +
    '<td>' + (d.regime||'—').replace(/_/g,' ') + '</td>';
  tbody.insertBefore(tr, tbody.firstChild);
  while (tbody.rows.length > MAX_FEED) tbody.deleteRow(tbody.rows.length - 1);
}

async function loadHistory() {
  try {
    const r = await fetch('/ml/decisions');
    const rows = await r.json();
    rows.slice().reverse().forEach(addFeedRow);
  } catch(e) {}
}

// ────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
function fmt2(n) { return parseFloat(n).toFixed(2); }
function fmtTime(iso) {
  try { return new Date(iso).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'}); }
  catch(_) { return '—'; }
}
function row(label, val) {
  return '<div class="stat-row"><span>' + label + '</span><span class="sv">' + val + '</span></div>';
}

// ────────────────────────────────────────────
// Boot — load everything immediately
// ────────────────────────────────────────────
refreshPnL();
loadTrades();
loadHistory();
startSSE();
setInterval(refreshPnL, 5000);
setInterval(loadTrades, 15000);
</script>
</body>
</html>"""


# ─────────────────────── Telegram Mini App HTML ───────────────────────

_MINIAPP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Cryprobotik Live</title>
<script src="https://telegram.org/js/telegram-web-app.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#17212b;--bg2:#232e3c;--bg3:#1c2733;--border:#2b5278;
  --text:#e8e8e8;--hint:#708499;--green:#3ddc84;--red:#f85149;
  --blue:#58a6ff;--yellow:#ffa500;--accent:#2b5278;}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--text);
  font-family:-apple-system,'Segoe UI',Arial,sans-serif;font-size:13px;
  overflow-x:hidden}
.app{display:flex;flex-direction:column;height:100%;max-height:100vh}

/* ── Header ── */
.hdr{background:var(--bg2);border-bottom:1px solid var(--border);
  padding:10px 14px;display:flex;align-items:center;gap:10px;flex-shrink:0}
.hdr-dot{width:8px;height:8px;border-radius:50%;background:var(--hint);flex-shrink:0}
.hdr-dot.live{background:var(--green)}
.hdr-title{font-weight:700;font-size:15px;color:var(--blue)}
.hdr-mode{background:var(--accent);color:#fff;font-size:10px;
  padding:2px 6px;border-radius:3px;font-weight:600;letter-spacing:.5px}
.hdr-right{margin-left:auto;display:flex;gap:6px;align-items:center}
.badge{font-size:11px;background:var(--bg3);border:1px solid var(--border);
  padding:2px 8px;border-radius:10px;white-space:nowrap}
.pos{color:var(--green)}.neg{color:var(--red)}.warn{color:var(--yellow)}

/* ── Equity bar ── */
.eq-bar{display:flex;gap:0;flex-shrink:0;border-bottom:1px solid var(--border);
  background:var(--bg3)}
.eq-cell{flex:1;padding:6px 10px;border-right:1px solid var(--border);
  display:flex;flex-direction:column;gap:1px}
.eq-cell:last-child{border-right:none}
.eq-lbl{font-size:9px;text-transform:uppercase;letter-spacing:.7px;color:var(--hint)}
.eq-val{font-size:13px;font-weight:700;color:var(--blue)}

/* ── Tabs ── */
.tabs{display:flex;background:var(--bg2);border-bottom:1px solid var(--border);
  flex-shrink:0}
.tab{flex:1;padding:8px 4px;text-align:center;font-size:11px;
  color:var(--hint);cursor:pointer;border-bottom:2px solid transparent;
  transition:color .15s,border-color .15s}
.tab.active{color:var(--blue);border-bottom-color:var(--blue)}

/* ── Tab panels ── */
.panels{flex:1;overflow:hidden;position:relative}
.panel{position:absolute;inset:0;overflow-y:auto;padding:10px;display:none;
  flex-direction:column;gap:8px}
.panel.active{display:flex}

/* ── Card ── */
.card{background:var(--bg2);border:1px solid var(--border);border-radius:8px;
  padding:10px}
.card-title{font-size:10px;text-transform:uppercase;letter-spacing:.8px;
  color:var(--hint);margin-bottom:8px}

/* ── Signal feed ── */
.feed-row{display:flex;align-items:center;gap:6px;padding:5px 0;
  border-bottom:1px solid var(--bg3);font-size:11px}
.feed-row:last-child{border-bottom:none}
.feed-time{color:var(--hint);width:38px;flex-shrink:0}
.feed-sym{width:48px;flex-shrink:0;font-weight:600}
.feed-side{width:28px;flex-shrink:0}
.feed-conf{color:var(--hint);width:30px;flex-shrink:0;text-align:right}
.feed-regime{color:var(--hint);flex:1;font-size:10px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.verdict{font-size:10px;font-weight:600;padding:1px 5px;border-radius:3px;
  flex-shrink:0}
.v-pass{background:#1a3a1a;color:var(--green)}
.v-cold{background:#2a2a1a;color:var(--yellow)}
.v-skip{background:#3a1a1a;color:var(--red)}

/* ── Positions ── */
.pos-row{display:flex;align-items:center;justify-content:space-between;
  padding:6px 0;border-bottom:1px solid var(--bg3);font-size:12px}
.pos-row:last-child{border-bottom:none}
.pos-sym{font-weight:700}
.pos-detail{color:var(--hint);font-size:10px}

/* ── ML stats ── */
.stat-row{display:flex;justify-content:space-between;align-items:center;
  padding:4px 0;border-bottom:1px solid var(--bg3);font-size:12px}
.stat-row:last-child{border-bottom:none}
.stat-val{color:var(--blue);font-weight:600}

/* ── Regime grid ── */
.regime-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}
.regime-cell{background:var(--bg3);border:1px solid var(--border);
  border-radius:6px;padding:6px 8px;font-size:11px}
.regime-name{color:var(--hint);font-size:9px;text-transform:uppercase;
  letter-spacing:.5px}
.regime-syms{margin-top:2px;font-weight:600;color:var(--text);
  word-break:break-all}

/* ── Chart ── */
#equity-chart{width:100%;height:120px}
.fill-row{display:flex;justify-content:space-between;align-items:center;
  padding:5px 0;border-bottom:1px solid var(--bg3);font-size:11px}
.fill-row:last-child{border-bottom:none}

/* ── Empty state ── */
.empty{color:var(--hint);font-size:12px;padding:20px 0;text-align:center}
</style>
</head>
<body>
<div class="app">

  <!-- Header -->
  <div class="hdr">
    <div class="hdr-dot" id="dot"></div>
    <span class="hdr-title">Cryprobotik</span>
    <span class="hdr-mode" id="mode-badge">PAPER</span>
    <div class="hdr-right">
      <span class="badge" id="ml-badge">ML: cold</span>
      <span class="badge" id="ks-badge" style="display:none">HALTED</span>
    </div>
  </div>

  <!-- Equity bar -->
  <div class="eq-bar">
    <div class="eq-cell">
      <span class="eq-lbl">Balance</span>
      <span class="eq-val" id="eq-balance">—</span>
    </div>
    <div class="eq-cell">
      <span class="eq-lbl">PnL</span>
      <span class="eq-val" id="eq-pnl">—</span>
    </div>
    <div class="eq-cell">
      <span class="eq-lbl">Wins</span>
      <span class="eq-val" id="eq-wr">—</span>
    </div>
    <div class="eq-cell">
      <span class="eq-lbl">Drawdown</span>
      <span class="eq-val" id="eq-dd">—</span>
    </div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <div class="tab active" onclick="switchTab('signals')">📡 Signals</div>
    <div class="tab" onclick="switchTab('positions')">📊 Positions</div>
    <div class="tab" onclick="switchTab('ml')">🤖 ML</div>
    <div class="tab" onclick="switchTab('regimes')">🌍 Regime</div>
  </div>

  <!-- Panels -->
  <div class="panels">

    <!-- Signals tab -->
    <div class="panel active" id="panel-signals">
      <div class="card">
        <div class="card-title">Live Signal Feed <span id="feed-count" style="color:var(--hint)"></span></div>
        <div id="feed-list"><div class="empty">Waiting for first signal…</div></div>
      </div>
      <div class="card">
        <div class="card-title">Recent Fills</div>
        <div id="fills-list"><div class="empty">No fills yet</div></div>
      </div>
    </div>

    <!-- Positions tab -->
    <div class="panel" id="panel-positions">
      <div class="card">
        <div class="card-title">Open Positions</div>
        <div id="pos-list"><div class="empty">No open positions</div></div>
      </div>
      <div class="card">
        <div class="card-title">Today's Stats</div>
        <div id="today-stats">
          <div class="stat-row"><span>Trades</span><span class="stat-val" id="s-trades">0</span></div>
          <div class="stat-row"><span>Win Rate</span><span class="stat-val" id="s-wr">—</span></div>
          <div class="stat-row"><span>Net PnL</span><span class="stat-val" id="s-pnl">—</span></div>
          <div class="stat-row"><span>Total Fees</span><span class="stat-val" id="s-fees">—</span></div>
        </div>
      </div>
    </div>

    <!-- ML tab -->
    <div class="panel" id="panel-ml">
      <div class="card">
        <div class="card-title">Model State</div>
        <div id="ml-stats-list">
          <div class="stat-row"><span>Status</span><span class="stat-val warn" id="ml-status">Cold Start</span></div>
          <div class="stat-row"><span>Version</span><span class="stat-val" id="ml-ver">—</span></div>
          <div class="stat-row"><span>Samples</span><span class="stat-val" id="ml-samples">0 / 50</span></div>
          <div class="stat-row"><span>Win Rate</span><span class="stat-val" id="ml-wr">—</span></div>
          <div class="stat-row"><span>Threshold</span><span class="stat-val" id="ml-thr">60%</span></div>
          <div class="stat-row"><span>Pending</span><span class="stat-val" id="ml-pending">0</span></div>
        </div>
      </div>
      <div class="card">
        <div class="card-title">Equity Curve</div>
        <canvas id="equity-chart"></canvas>
        <div style="text-align:center;color:var(--hint);font-size:10px;margin-top:4px" id="chart-info">No data</div>
      </div>
      <div class="card">
        <div class="card-title">Top Feature Importances</div>
        <div id="features-list"><div class="empty">Model not trained yet</div></div>
      </div>
    </div>

    <!-- Regime tab -->
    <div class="panel" id="panel-regimes">
      <div class="card">
        <div class="card-title">Current Regimes</div>
        <div class="regime-grid" id="regime-grid"><div class="empty" style="grid-column:1/-1">Waiting for bar close…</div></div>
      </div>
    </div>

  </div><!-- /panels -->
</div><!-- /app -->

<script>
'use strict';

// ── Telegram WebApp init ──────────────────────────
const tg = window.Telegram && window.Telegram.WebApp;
if (tg) { tg.ready(); tg.expand(); }

// ── Tab switching ─────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i)=>{
    const names=['signals','positions','ml','regimes'];
    t.classList.toggle('active', names[i]===name);
  });
  document.querySelectorAll('.panel').forEach(p=>{
    p.classList.toggle('active', p.id==='panel-'+name);
  });
}

// ── State ─────────────────────────────────────────
let signals = [];
let equityPts = [];
let equityChart = null;
let positions = {};   // symbol → {side, upnl, qty}
let regimes = {};     // symbol → regime
let fills = [];

// ── Helpers ───────────────────────────────────────
const fmt2 = n => parseFloat(n||0).toFixed(2);
const fmtPct = n => (parseFloat(n||0)*100).toFixed(1)+'%';
const fmtTime = iso => { try { return new Date(iso).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}); } catch(_){return '—';} };
const regimeShort = r => (r||'—').replace('trend','TR').replace('range','RNG').replace('_high_vol','↑').replace('_low_vol','↓').replace('chop','CHOP');
const regimeEmoji = r => ({'trend_high_vol':'🚀','trend_low_vol':'📈','range_high_vol':'↔️','range_low_vol':'😴','chop':'🌀'})[r]||'❓';
function pclass(v){return v>0?'pos':v<0?'neg':'';}

// ── Signal feed ───────────────────────────────────
function addSignal(d) {
  if (!d.symbol) return;
  signals.unshift(d);
  if (signals.length > 80) signals.pop();
  renderSignals();
}

function renderSignals() {
  const el = document.getElementById('feed-list');
  const cnt = document.getElementById('feed-count');
  cnt.textContent = '('+signals.length+')';
  if (!signals.length) { el.innerHTML='<div class="empty">Waiting for first signal…</div>'; return; }
  el.innerHTML = signals.slice(0,40).map(d => {
    const verdict = d.cold_start
      ? '<span class="verdict v-cold">COLD</span>'
      : d.accepted
        ? '<span class="verdict v-pass">PASS</span>'
        : '<span class="verdict v-skip">SKIP</span>';
    const sideCol = d.side==='buy' ? 'var(--green)' : 'var(--red)';
    return '<div class="feed-row">'+
      '<span class="feed-time">'+fmtTime(d.ts)+'</span>'+
      '<span class="feed-sym">'+d.symbol.split('/')[0]+'</span>'+
      '<span class="feed-side" style="color:'+sideCol+';">'+(d.side==='buy'?'▲':'▼')+'</span>'+
      '<span class="feed-conf">'+Math.round((d.confidence||d.ml_score||0)*100)+'%</span>'+
      '<span class="feed-regime">'+regimeShort(d.regime)+'</span>'+
      verdict+'</div>';
  }).join('');
}

// ── Positions ─────────────────────────────────────
function renderPositions() {
  const el = document.getElementById('pos-list');
  const posArr = Object.values(positions);
  if (!posArr.length) { el.innerHTML='<div class="empty">No open positions</div>'; return; }
  el.innerHTML = posArr.map(p => {
    const arrow = p.side==='buy'||p.side==='long' ? '▲' : '▼';
    const upnl = p.unrealized_pnl||0;
    const cls = pclass(upnl);
    return '<div class="pos-row">'+
      '<div><div class="pos-sym">'+arrow+' '+p.symbol.split('/')[0]+'</div>'+
      '<div class="pos-detail">qty='+p.quantity+' entry='+parseFloat(p.entry_price||0).toFixed(4)+'</div></div>'+
      '<div class="'+cls+'" style="font-weight:700;">'+(upnl>=0?'+':'')+'$'+fmt2(Math.abs(upnl))+'</div>'+
      '</div>';
  }).join('');
}

// ── Regime grid ───────────────────────────────────
function renderRegimes(regMap) {
  regimes = regMap || regimes;
  const el = document.getElementById('regime-grid');
  const byReg = {};
  Object.entries(regimes).forEach(([sym,reg])=>{
    if (!byReg[reg]) byReg[reg]=[];
    byReg[reg].push(sym.split('/')[0]);
  });
  const entries = Object.entries(byReg);
  if (!entries.length) { el.innerHTML='<div class="empty" style="grid-column:1/-1">Waiting for bar close…</div>'; return; }
  el.innerHTML = entries.map(([reg,syms])=>
    '<div class="regime-cell">'+
    '<div class="regime-name">'+regimeEmoji(reg)+' '+reg.replace(/_/g,' ')+'</div>'+
    '<div class="regime-syms">'+syms.join(', ')+'</div>'+
    '</div>'
  ).join('');
}

// ── Fills ─────────────────────────────────────────
function renderFills() {
  const el = document.getElementById('fills-list');
  if (!fills.length) { el.innerHTML='<div class="empty">No fills yet</div>'; return; }
  el.innerHTML = fills.slice(0,20).map(f => {
    const pnl = f.realized_pnl;
    const cls = pclass(pnl);
    return '<div class="fill-row">'+
      '<span>'+fmtTime(f.ts)+' '+f.symbol.split('/')[0]+'</span>'+
      '<span class="'+cls+'">'+(pnl!=null?(pnl>=0?'+':'')+'$'+fmt2(Math.abs(pnl)):'exec')+'</span>'+
      '</div>';
  }).join('');
}

// ── ML stats ─────────────────────────────────────
function updateML(m) {
  const status = document.getElementById('ml-status');
  if (m.cold_start) {
    status.textContent='Cold Start'; status.className='stat-val warn';
  } else {
    status.textContent='Trained'; status.className='stat-val pos';
  }
  document.getElementById('ml-ver').textContent = 'v'+(m.model_version||0);
  document.getElementById('ml-samples').textContent = (m.n_samples||0)+' / 50';
  document.getElementById('ml-wr').textContent = m.win_rate!=null ? fmtPct(m.win_rate) : '—';
  document.getElementById('ml-thr').textContent = Math.round((m.accept_threshold||0.6)*100)+'%';
  document.getElementById('ml-pending').textContent = m.pending_positions||0;
  document.getElementById('ml-badge').textContent = m.cold_start ? 'ML: cold' : 'ML: v'+(m.model_version||0);

  if (m.feature_importances) {
    const sorted = Object.entries(m.feature_importances).sort((a,b)=>b[1]-a[1]).slice(0,8);
    const max = sorted[0]?sorted[0][1]:1;
    document.getElementById('features-list').innerHTML = sorted.map(([k,v])=>
      '<div style="margin-bottom:4px">'+
      '<div style="display:flex;justify-content:space-between;font-size:10px;color:var(--hint);margin-bottom:2px">'+
      '<span>'+k+'</span><span>'+Math.round(v/max*100)+'%</span></div>'+
      '<div style="background:var(--bg3);border-radius:2px;height:4px">'+
      '<div style="width:'+Math.round(v/max*100)+'%;height:4px;border-radius:2px;background:var(--blue)"></div>'+
      '</div></div>'
    ).join('');
  }
}

// ── Trading stats ─────────────────────────────────
function updateStats(s) {
  const bal = s.paper_balance || 10000;
  const pnl = s.total_pnl || 0;
  const pct = ((bal-10000)/10000*100);
  const cls = pnl>0?'pos':pnl<0?'neg':'';
  document.getElementById('eq-balance').textContent = '$'+fmt2(bal);
  const pnlEl = document.getElementById('eq-pnl');
  pnlEl.textContent = (pnl>=0?'+':'')+fmt2(pnl);
  pnlEl.className = 'eq-val '+cls;
  const wr = s.win_rate;
  const wrEl = document.getElementById('eq-wr');
  wrEl.textContent = wr!=null ? fmtPct(wr) : '—';
  wrEl.className = 'eq-val '+(wr!=null?(wr>=0.5?'pos':'neg'):'');
  document.getElementById('eq-dd').textContent = '—';
  // Today
  document.getElementById('s-trades').textContent = s.closed_trades||0;
  document.getElementById('s-wr').textContent = wr!=null ? fmtPct(wr) : '—';
  document.getElementById('s-pnl').textContent = (pnl>=0?'+':'')+fmt2(pnl);
  document.getElementById('s-fees').textContent = '-$'+fmt2(s.total_fees||0);
  // Equity chart
  equityPts.push(bal);
  if (equityPts.length>200) equityPts.shift();
  drawChart();
}

// ── Equity chart ──────────────────────────────────
function drawChart() {
  const canvas = document.getElementById('equity-chart');
  const latest = equityPts[equityPts.length-1]||10000;
  const pct = ((latest-10000)/10000*100).toFixed(2);
  document.getElementById('chart-info').textContent = '$'+fmt2(latest)+(latest>=10000?' +':' ')+pct+'%';
  const color = latest>=10000 ? '#3ddc84' : '#f85149';
  if (!equityChart) {
    equityChart = new Chart(canvas, {
      type:'line',
      data:{labels:equityPts.map((_,i)=>i),
        datasets:[{data:equityPts,borderColor:color,borderWidth:1.5,
          fill:true,backgroundColor:color+'22',pointRadius:0,tension:.3}]},
      options:{animation:false,responsive:true,maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{x:{display:false},y:{display:false}}}
    });
  } else {
    equityChart.data.labels = equityPts.map((_,i)=>i);
    equityChart.data.datasets[0].data = equityPts;
    equityChart.data.datasets[0].borderColor = color;
    equityChart.data.datasets[0].backgroundColor = color+'22';
    equityChart.update('none');
  }
}

// ── WebSocket ────────────────────────────────────
function connectWS() {
  const proto = location.protocol==='https:'?'wss:':'ws:';
  const ws = new WebSocket(proto+'//'+location.host+'/ws/live');
  ws.onopen = () => document.getElementById('dot').classList.add('live');
  ws.onclose = () => { document.getElementById('dot').classList.remove('live'); setTimeout(connectWS,3000); };
  ws.onerror = () => ws.close();
  ws.onmessage = e => {
    try {
      const d = JSON.parse(e.data);
      if (d.type==='ping') return;
      if (d.type==='signal') addSignal(d);
      else if (d.type==='regime') renderRegimes(d.regimes);
      else if (d.type==='position') { positions[d.symbol]=d; renderPositions(); }
      else if (d.type==='position_close') { delete positions[d.symbol]; renderPositions(); }
      else if (d.type==='stats') updateStats(d);
    } catch(_) {}
  };
}

// ── SSE ───────────────────────────────────────────
function connectSSE() {
  const es = new EventSource('/ml/stream');
  es.onopen = () => document.getElementById('dot').classList.add('live');
  es.onmessage = e => { try { addSignal(JSON.parse(e.data)); } catch(_){} };
  es.onerror = () => { es.close(); setTimeout(connectSSE,4000); };
}

// ── Polling (fallback + initial load) ─────────────
async function poll() {
  try {
    const [sRes, mRes] = await Promise.all([fetch('/trading/stats'), fetch('/ml/stats')]);
    if (sRes.ok) updateStats(await sRes.json());
    if (mRes.ok) updateML(await mRes.json());
  } catch(_) {}
  try {
    const fRes = await fetch('/trading/fills');
    if (fRes.ok) { fills = await fRes.json(); renderFills(); }
  } catch(_) {}
}
async function loadHistory() {
  try {
    const r = await fetch('/ml/decisions');
    if (r.ok) { const rows = await r.json(); rows.slice().reverse().forEach(addSignal); }
  } catch(_) {}
}
async function loadPositions() {
  try {
    const r = await fetch('/api/positions');
    if (r.ok) {
      const data = await r.json();
      positions = {};
      data.forEach(p => { positions[p.symbol] = p; });
      renderPositions();
    }
  } catch(_) {}
}

// ── Boot ──────────────────────────────────────────
connectWS();
connectSSE();
loadHistory();
loadPositions();
poll();
setInterval(poll, 10000);
setInterval(loadPositions, 15000);
</script>
</body>
</html>"""


# ─────────────────────── FastAPI app ───────────────────────

def build_app(
    connectors: dict[str, "ExchangeConnector"],
    storage: "Storage",
    kill_switch: "KillSwitch",
    ml_filter: "MLSignalFilter | None" = None,
    broadcaster: "LiveBroadcaster | None" = None,
) -> FastAPI:
    app = FastAPI(title="Cryprobotik", version="0.1.0", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(response: Response) -> dict[str, object]:
        problems: list[str] = []

        # DB
        try:
            async with storage.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as e:
            problems.append(f"db: {e}")

        # WS feeds
        for name, conn in connectors.items():
            public = getattr(conn, "_ws_public", None)
            private = getattr(conn, "_ws_private", None)
            if public is not None and not getattr(public, "connected", False):
                problems.append(f"{name}.public ws disconnected")
            if private is not None and not getattr(private, "connected", False):
                problems.append(f"{name}.private ws disconnected")

        # Kill switch
        if kill_switch.is_halted:
            problems.append("kill_switch: halted")

        if problems:
            response.status_code = 503
            return {"ready": False, "problems": problems}
        return {"ready": True}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    # ── ML endpoints ──────────────────────────────────────────────

    @app.get("/ml/dashboard", response_class=HTMLResponse)
    async def ml_dashboard() -> HTMLResponse:
        return HTMLResponse(content=_DASHBOARD_HTML)

    @app.get("/ml/stats")
    async def ml_stats() -> dict:
        if ml_filter is None:
            return {"error": "ML filter not initialized"}
        return ml_filter.stats()

    @app.get("/ml/decisions")
    async def ml_decisions() -> list[dict]:
        try:
            async with storage.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT ts, symbol, exchange, side, confidence, ml_score,
                           accepted, cold_start, model_version, regime, net_vote
                    FROM ml_decisions
                    ORDER BY ts DESC
                    LIMIT 50
                    """
                )
            return [
                {
                    "ts": row["ts"].isoformat(),
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "confidence": round(row["confidence"], 3),
                    "ml_score": round(row["ml_score"], 3),
                    "accepted": row["accepted"],
                    "cold_start": row["cold_start"],
                    "model_version": row["model_version"],
                    "regime": row["regime"],
                    "net_vote": row["net_vote"],
                }
                for row in rows
            ]
        except Exception as e:
            log.warning("ml.decisions_query_failed", error=str(e))
            return []

    @app.get("/ml/stream")
    async def ml_stream() -> StreamingResponse:
        """Server-Sent Events — one event per new ML decision."""
        if ml_filter is None:
            async def empty():
                yield "data: {}\n\n"
            return StreamingResponse(empty(), media_type="text/event-stream")

        q = ml_filter.subscribe_sse()

        async def generate() -> AsyncIterator[str]:
            try:
                while True:
                    try:
                        payload = await asyncio.wait_for(q.get(), timeout=30.0)
                        yield f"data: {payload}\n\n"
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"   # keep-alive comment
            finally:
                ml_filter.unsubscribe_sse(q)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/trading/stats")
    async def trading_stats() -> dict:
        try:
            async with storage.pool.acquire() as conn:
                fills = await conn.fetch(
                    """
                    SELECT symbol, side, quantity, price, fee, realized_pnl, ts
                    FROM fills
                    ORDER BY ts
                    """
                )
                total_trades = len([f for f in fills if f["realized_pnl"] is not None])
                winning_trades = len([f for f in fills if (f["realized_pnl"] or 0) > 0])
                total_pnl = sum((f["realized_pnl"] or 0) for f in fills)
                total_fees = sum((f["fee"] or 0) for f in fills)

                # Recent signals
                decisions = await conn.fetch(
                    """
                    SELECT regime, COUNT(*) as cnt, AVG(ml_score) as avg_score
                    FROM ml_decisions
                    GROUP BY regime
                    ORDER BY cnt DESC
                    """
                )

            return {
                "paper_balance": 10000.0 + total_pnl - total_fees,
                "total_pnl": round(total_pnl, 2),
                "total_fees": round(total_fees, 2),
                "total_fills": len(fills),
                "closed_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": round(winning_trades / total_trades, 3) if total_trades > 0 else None,
                "regimes_seen": [
                    {"regime": r["regime"], "signals": r["cnt"], "avg_ml_score": round(r["avg_score"], 3)}
                    for r in decisions
                ],
            }
        except Exception as e:
            log.warning("trading.stats_failed", error=str(e))
            return {"error": str(e)}

    @app.get("/trading/fills")
    async def trading_fills() -> list[dict]:
        try:
            async with storage.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT ts, symbol, side, quantity, price, fee, realized_pnl
                    FROM fills
                    ORDER BY ts DESC
                    LIMIT 50
                    """
                )
            return [
                {
                    "ts": row["ts"].isoformat(),
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "quantity": float(row["quantity"]),
                    "price": float(row["price"]),
                    "fee": float(row["fee"]),
                    "realized_pnl": float(row["realized_pnl"]) if row["realized_pnl"] is not None else None,
                }
                for row in rows
            ]
        except Exception as e:
            log.warning("trading.fills_failed", error=str(e))
            return []

    # ── Mini App endpoints ─────────────────────────────────────────────

    @app.get("/app", response_class=HTMLResponse)
    async def miniapp() -> HTMLResponse:
        return HTMLResponse(content=_MINIAPP_HTML)

    @app.get("/api/positions")
    async def api_positions() -> list[dict]:
        """Return current open positions for initial Mini App load."""
        try:
            async with storage.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT ON (exchange, symbol)
                           symbol, exchange, side, quantity, entry_price,
                           mark_price, unrealized_pnl
                    FROM positions_snapshot
                    WHERE side != 'flat'
                    ORDER BY exchange, symbol, ts DESC
                    """
                )
            return [
                {
                    "symbol": r["symbol"],
                    "exchange": r["exchange"],
                    "side": r["side"],
                    "quantity": float(r["quantity"]),
                    "entry_price": float(r["entry_price"]),
                    "mark_price": float(r["mark_price"]) if r["mark_price"] else None,
                    "unrealized_pnl": float(r["unrealized_pnl"]) if r["unrealized_pnl"] else 0.0,
                }
                for r in rows
            ]
        except Exception as e:
            log.warning("api.positions_failed", error=str(e))
            return []

    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket) -> None:
        """Zero-delay push: position updates, regime changes, signal events."""
        await websocket.accept()
        if broadcaster is None:
            # No broadcaster — keep alive with pings only
            try:
                while True:
                    await asyncio.sleep(30)
                    await websocket.send_json({"type": "ping"})
            except WebSocketDisconnect:
                return
            return
        q = broadcaster.subscribe()
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=25.0)
                    await websocket.send_text(payload)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping"})
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            broadcaster.unsubscribe(q)

    return app


async def serve_health(
    app: FastAPI, host: str = "0.0.0.0", port: int = 8080
) -> None:
    """Run the FastAPI app with uvicorn — intended to be launched as a task."""
    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning", access_log=False,
    )
    server = uvicorn.Server(config)
    await server.serve()
