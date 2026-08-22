"""Watch the blockchain live — a local web dashboard (stdlib only).

Runs the integrated node (real beacon + DA + leader election) in a background
thread and streams each block to a browser page over Server-Sent Events. Open
the printed URL and watch blocks arrive: height, hash, the beacon-elected
leader, the beacon value, how many deltas the DA layer admitted, and the model
accuracy climbing in real time. Toggle a withholding miner and watch its
deltas get excluded by availability sampling, live.

  scripts/run live          # then open http://127.0.0.1:8737

No dependencies beyond what the node already needs (numpy, py_ecc).
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .integrated import DA_N, MODEL, new_chain

STATE = {"blocks": [], "n": 0, "t": 0, "withholders": [], "running": True}
_LOCK = threading.Lock()


def _chain_thread(n=5, t=3, interval=1.5):
    import numpy as np
    from .chain import dequantize
    chain = new_chain(n=n, t=t)
    with _LOCK:
        STATE["n"], STATE["t"] = n, t
    while STATE["running"]:
        blk = chain.round()
        acc = MODEL.accuracy(dequantize(chain.w_int),
                             MODEL.sample_batch(np.random.default_rng(999), 200))
        rec = dict(height=blk.height, hash=blk.hash()[:16], prev=blk.prev_hash[:12],
                   leader=blk.leader, beacon=blk.beacon_hex[:16],
                   included=len(blk.miner_ids), miners=blk.miner_ids,
                   da=len(blk.da_roots), total=n, acc=round(acc, 3),
                   ts=int(time.time()))
        with _LOCK:
            STATE["blocks"].append(rec)
            STATE["withholders"] = sorted(chain.withholders)
            STATE["_apply_withhold"] = chain.withholders   # live handle
        time.sleep(interval)


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Sestrian — live chain</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1218;--surface:#161b24;--line:#232a35;--ink:#e7e9ee;--soft:#99a0ac;
--gold:#dca85a;--teal:#62b9a6;--mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,monospace;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--mono);font-size:14px}
.wrap{max-width:1000px;margin:0 auto;padding:24px}
h1{font-size:15px;letter-spacing:.18em;text-transform:uppercase;color:var(--gold);margin:0 0 2px}
.sub{color:var(--soft);font-size:12px;margin-bottom:20px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:8px}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.stat .k{color:var(--soft);font-size:11px;letter-spacing:.1em;text-transform:uppercase}
.stat .v{font-size:26px;margin-top:6px;font-variant-numeric:tabular-nums}
.stat .v.gold{color:var(--gold)} .stat .v.teal{color:var(--teal)}
#spark{height:44px;width:100%;display:block;margin:14px 0}
.ctl{margin:10px 0 18px;color:var(--soft);font-size:12px}
.ctl button{font-family:var(--mono);font-size:11px;background:transparent;color:var(--ink);
border:1px solid var(--line);border-radius:100px;padding:5px 12px;cursor:pointer;margin-right:6px}
.ctl button:hover{border-color:var(--gold)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{text-align:left;color:var(--soft);font-size:11px;letter-spacing:.08em;text-transform:uppercase;
padding:8px 10px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg)}
td{padding:9px 10px;border-bottom:1px solid var(--line)}
tr:first-child td{animation:flash 1s ease}
@keyframes flash{from{background:rgba(220,168,90,.14)}to{background:transparent}}
.h{color:var(--gold)} .rt{color:var(--soft)} .ok{color:var(--teal)}
.bar{display:inline-block;height:7px;background:var(--teal);border-radius:4px;vertical-align:middle}
</style></head><body><div class=wrap>
<h1>Sestrian — live chain</h1>
<div class=sub id=meta>connecting…</div>
<div class=stats>
 <div class=stat><div class=k>height</div><div class="v gold" id=height>—</div></div>
 <div class=stat><div class=k>model accuracy</div><div class="v teal" id=acc>—</div></div>
 <div class=stat><div class=k>current leader</div><div class=v id=leader>—</div></div>
 <div class=stat><div class=k>beacon (round)</div><div class=v id=beacon style=font-size:15px>—</div></div>
</div>
<canvas id=spark></canvas>
<div class=ctl>data-availability: <span id=wh>no withholders</span>
 &nbsp; <button onclick="toggle(2)">toggle miner 2 withholding</button>
 <span class=rt>— its deltas get excluded by DA sampling, live</span></div>
<table><thead><tr><th>blk</th><th>hash</th><th>leader</th><th>beacon</th>
<th>DA-admitted</th><th>miners</th><th>acc</th></tr></thead><tbody id=rows></tbody></table>
</div><script>
let accs=[];
function toggle(m){fetch('/withhold?miner='+m).then(r=>r.json()).then(s=>{});}
function spark(){const c=document.getElementById('spark'),x=c.getContext('2d');
const w=c.width=c.clientWidth*2,h=c.height=88;x.clearRect(0,0,w,h);
if(accs.length<2)return;const n=accs.length,dx=w/(n-1);
x.strokeStyle='#62b9a6';x.lineWidth=3;x.beginPath();
accs.forEach((a,i)=>{const y=h-6-a*(h-12);i?x.lineTo(i*dx,y):x.moveTo(0,y)});x.stroke();}
function row(b){const tr=document.createElement('tr');
tr.innerHTML=`<td class=h>#${b.height}</td><td class=rt>${b.hash}…</td>
<td>node ${b.leader}</td><td class=rt>${b.beacon}…</td>
<td class=ok>${b.da}/${b.total} <span class=bar style="width:${b.da/b.total*40}px"></span></td>
<td class=rt>${b.miners.join(',')}</td><td>${b.acc.toFixed(3)}</td>`;return tr;}
const es=new EventSource('/events');
es.onmessage=e=>{const b=JSON.parse(e.data);
document.getElementById('height').textContent=b.height;
document.getElementById('acc').textContent=b.acc.toFixed(3);
document.getElementById('leader').textContent='node '+b.leader;
document.getElementById('beacon').textContent=b.beacon+'…';
document.getElementById('meta').textContent=`${b.total}-node network · 3-of-${b.total} threshold beacon · erasure-coded DA · no coordinator`;
const rows=document.getElementById('rows');rows.insertBefore(row(b),rows.firstChild);
while(rows.children.length>40)rows.removeChild(rows.lastChild);
accs.push(b.acc);if(accs.length>120)accs.shift();spark();};
es.addEventListener('withholders',e=>{const w=JSON.parse(e.data);
document.getElementById('wh').textContent=w.length?('withholding: miners '+w.join(', ')):'no withholders';});
window.addEventListener('resize',spark);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/withhold"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            m = int(q.get("miner", ["2"])[0])
            with _LOCK:
                wh = STATE.get("_apply_withhold")
                if wh is not None:
                    wh.discard(m) if m in wh else wh.add(m)
            self._json({"ok": True})
        elif self.path == "/events":
            self._events()
        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        sent = 0
        last_wh = None
        try:
            while STATE["running"]:
                with _LOCK:
                    blocks = STATE["blocks"][sent:]
                    sent = len(STATE["blocks"])
                    wh = list(STATE["withholders"])
                for b in blocks:
                    self.wfile.write(f"data: {json.dumps(b)}\n\n".encode())
                if wh != last_wh:
                    self.wfile.write(
                        f"event: withholders\ndata: {json.dumps(wh)}\n\n".encode())
                    last_wh = wh
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main(host="127.0.0.1", port=8737):
    threading.Thread(target=_chain_thread, daemon=True).start()
    srv = ThreadingHTTPServer((host, port), Handler)
    print("=" * 60)
    print("  Sestrian live chain viewer")
    print(f"  open  ->  http://{host}:{port}")
    print("  (Ctrl-C to stop)")
    print("=" * 60, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        STATE["running"] = False
        print("\nstopped.")


if __name__ == "__main__":
    main()
