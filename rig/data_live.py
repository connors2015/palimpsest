"""Watch your data earn — a live dashboard for the data economy.

Runs the data-economy simulator (Stage 1 pricing + Stage 2 royalties) and
streams it to a browser: each contributor's signing bonus, their royalties
ticking up as the model answers questions their data shaped, and a live feed of
"your data was used in an answer → +x" events. This is the "get paid for your
data" experience made tangible.

  scripts/run data_live     # then open http://127.0.0.1:8738

Stdlib only (numpy + pynacl for the sim).
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .data_flywheel import CONTRIBUTORS, POPULARITY, Simulator

STATE = {"contribs": {}, "feed": [], "served": 0, "running": True}
_LOCK = threading.Lock()


def _sim_thread():
    sim = Simulator(seed=0)
    with _LOCK:
        STATE["contribs"] = sim.snapshot()
    while STATE["running"]:
        ev = sim.serve_query()
        with _LOCK:
            STATE["contribs"] = sim.snapshot()
            STATE["served"] = sim.served
            top_pay = max(ev["paid"].items(), key=lambda kv: kv[1]) if ev["paid"] else None
            if top_pay:
                STATE["feed"].insert(0, dict(
                    q=int(ev["query_domain"]), who=top_pay[0],
                    amt=round(top_pay[1], 3)))
                STATE["feed"] = STATE["feed"][:30]
        time.sleep(0.25)


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Palimpsest — your data, earning</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1218;--surface:#161b24;--line:#232a35;--ink:#e7e9ee;--soft:#99a0ac;
--gold:#dca85a;--teal:#62b9a6;--mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,monospace;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--mono);font-size:14px}
.wrap{max-width:960px;margin:0 auto;padding:24px}
h1{font-size:15px;letter-spacing:.18em;text-transform:uppercase;color:var(--gold);margin:0 0 2px}
.sub{color:var(--soft);font-size:12px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:1.4fr 1fr;gap:18px}
@media(max-width:720px){.grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:12px}
.who{display:flex;align-items:baseline;justify-content:space-between}
.name{font-size:16px;text-transform:capitalize} .chan{color:var(--soft);font-size:11px;letter-spacing:.05em}
.pop{color:var(--soft);font-size:11px}
.earn{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:12px}
.e .k{color:var(--soft);font-size:10px;letter-spacing:.06em;text-transform:uppercase}
.e .v{font-size:19px;font-variant-numeric:tabular-nums;margin-top:3px}
.e .v.roy{color:var(--teal)} .e .v.tot{color:var(--gold)}
.rbar{height:6px;background:var(--line);border-radius:4px;margin-top:10px;overflow:hidden}
.rbar>i{display:block;height:100%;background:var(--teal);transition:width .3s}
.feedh{color:var(--soft);font-size:11px;letter-spacing:.08em;text-transform:uppercase;margin:0 0 8px}
.ev{padding:8px 10px;border-bottom:1px solid var(--line);font-size:12.5px;animation:flash 1s ease}
.ev:first-child{}.ev b{color:var(--teal);text-transform:capitalize}.ev .q{color:var(--soft)}
@keyframes flash{from{background:rgba(98,185,166,.12)}to{background:transparent}}
.hstat{font-size:12px;color:var(--soft);margin-bottom:14px}
.hstat b{color:var(--ink)}
</style></head><body><div class=wrap>
<h1>Your data, earning</h1>
<div class=sub id=meta>the model answers questions; the data that shaped each answer gets paid</div>
<div class=hstat>queries answered: <b id=served>0</b> &nbsp;·&nbsp; royalty share per query: <b>30%</b></div>
<div class=grid>
 <div id=cards></div>
 <div class=card><div class=feedh>live — data used in answers</div><div id=feed></div></div>
</div>
</div><script>
const POP=[0.40,0.30,0.20,0.10];
let maxRoy=1;
function card(name,c){
 const tot=(c.bonus+c.royalties);
 return `<div class=card><div class=who><span class=name>${name}</span>
 <span class=chan>${c.channel} · domain ${c.domain}</span></div>
 <div class=pop>queried ${(POP[c.domain]*100).toFixed(0)}% of the time</div>
 <div class=earn>
  <div class=e><div class=k>signing bonus</div><div class=v>${c.bonus.toFixed(0)}</div></div>
  <div class=e><div class=k>royalties</div><div class="v roy">${c.royalties.toFixed(1)}</div></div>
  <div class=e><div class=k>total</div><div class="v tot">${tot.toFixed(1)}</div></div>
 </div><div class=rbar><i style="width:${Math.min(100,c.royalties/maxRoy*100)}%"></i></div></div>`;
}
const es=new EventSource('/events');
es.onmessage=e=>{const s=JSON.parse(e.data);
 document.getElementById('served').textContent=s.served;
 maxRoy=Math.max(1,...Object.values(s.contribs).map(c=>c.royalties));
 document.getElementById('cards').innerHTML=Object.entries(s.contribs)
   .sort((a,b)=>(b[1].bonus+b[1].royalties)-(a[1].bonus+a[1].royalties))
   .map(([n,c])=>card(n,c)).join('');
 document.getElementById('feed').innerHTML=s.feed.map(f=>
   `<div class=ev><span class=q>Q about domain ${f.q} →</span> <b>${f.who}</b>'s data +${f.amt.toFixed(3)}</div>`).join('');
};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/events":
            self._events()
        else:
            self.send_response(404); self.end_headers()

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while STATE["running"]:
                with _LOCK:
                    payload = json.dumps(dict(contribs=STATE["contribs"],
                                              feed=STATE["feed"], served=STATE["served"]))
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.4)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main(host="127.0.0.1", port=8738):
    threading.Thread(target=_sim_thread, daemon=True).start()
    print("=" * 60)
    print("  Palimpsest — 'watch your data earn' dashboard")
    print(f"  open  ->  http://{host}:{port}")
    print("  (Ctrl-C to stop)")
    print("=" * 60, flush=True)
    srv = ThreadingHTTPServer((host, port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        STATE["running"] = False
        print("\nstopped.")


if __name__ == "__main__":
    main()
