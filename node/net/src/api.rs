//! HTTP API — the same JSON routes the Python watcher exposes, so the wallet
//! CLI and any UI work against a Rust node unchanged:
//!
//!   GET  /status           chain summary
//!   GET  /balance?addr=    grains, nonce, supply, height
//!   GET  /data/registry    data registry + open challenges
//!   POST /transfer         signed transfer -> mempool + gossip
//!   POST /data/submit | /data/challenge | /data/vote
//!
//! Handlers talk to the node loop over an mpsc command channel with oneshot
//! replies — the node's state is single-owner, no locks.

use axum::{
    extract::{Query, State},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use tokio::sync::{mpsc, oneshot};
use tracing::info;

#[derive(Debug)]
pub enum ApiCmd {
    Status(oneshot::Sender<Value>),
    Balance(String, oneshot::Sender<Value>),
    Registry(oneshot::Sender<Value>),
    Chain(oneshot::Sender<Value>),
    SubmitAccountTx(Value, oneshot::Sender<Value>),
}

#[derive(Clone)]
struct Api {
    tx: mpsc::Sender<ApiCmd>,
}

async fn ask(tx: &mpsc::Sender<ApiCmd>, make: impl FnOnce(oneshot::Sender<Value>) -> ApiCmd)
    -> Json<Value>
{
    let (otx, orx) = oneshot::channel();
    if tx.send(make(otx)).await.is_err() {
        return Json(json!({"ok": false, "error": "node shutting down"}));
    }
    Json(orx.await.unwrap_or_else(|_| json!({"ok": false, "error": "node dropped request"})))
}

async fn status(State(api): State<Api>) -> Json<Value> {
    ask(&api.tx, ApiCmd::Status).await
}

async fn balance(State(api): State<Api>, Query(q): Query<HashMap<String, String>>)
    -> Json<Value>
{
    let addr = q.get("addr").cloned().unwrap_or_default();
    ask(&api.tx, |o| ApiCmd::Balance(addr, o)).await
}

async fn registry(State(api): State<Api>) -> Json<Value> {
    ask(&api.tx, ApiCmd::Registry).await
}

async fn chain(State(api): State<Api>) -> Json<Value> {
    ask(&api.tx, ApiCmd::Chain).await
}

async fn dashboard() -> axum::response::Html<&'static str> {
    axum::response::Html(PAGE)
}

fn tag(kind: &str, mut body: Value) -> Value {
    body["kind"] = json!(kind);
    body
}

async fn transfer(State(api): State<Api>, Json(b): Json<Value>) -> Json<Value> {
    ask(&api.tx, |o| ApiCmd::SubmitAccountTx(tag("transfer", b), o)).await
}

async fn data_submit(State(api): State<Api>, Json(b): Json<Value>) -> Json<Value> {
    ask(&api.tx, |o| ApiCmd::SubmitAccountTx(tag("data_submit", b), o)).await
}

async fn data_challenge(State(api): State<Api>, Json(b): Json<Value>) -> Json<Value> {
    ask(&api.tx, |o| ApiCmd::SubmitAccountTx(tag("data_challenge", b), o)).await
}

async fn data_vote(State(api): State<Api>, Json(b): Json<Value>) -> Json<Value> {
    ask(&api.tx, |o| ApiCmd::SubmitAccountTx(tag("data_vote", b), o)).await
}

pub async fn run(port: u16, tx: mpsc::Sender<ApiCmd>) {
    let app = Router::new()
        .route("/", get(dashboard))
        .route("/status", get(status))
        .route("/balance", get(balance))
        .route("/chain", get(chain))
        .route("/data/registry", get(registry))
        .route("/transfer", post(transfer))
        .route("/data/submit", post(data_submit))
        .route("/data/challenge", post(data_challenge))
        .route("/data/vote", post(data_vote))
        .with_state(Api { tx });
    // retry the bind: fast restarts leave the old socket lingering briefly, and
    // a silently-dead API made live nodes look wedged during the rehearsal
    let listener = loop {
        match tokio::net::TcpListener::bind(("0.0.0.0", port)).await {
            Ok(l) => break l,
            Err(e) => {
                tracing::warn!("api port {port} busy ({e}); retrying in 3s");
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }
        }
    };
    info!("http api on 0.0.0.0:{port}");
    let _ = axum::serve(listener, app).await;
}

/// The always-on chain dashboard, served by the node itself at `/`.
const PAGE: &str = r#"<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>palimpsest · chain</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>&#129504;</text></svg>">
<style>
:root{--bg:#0a0d12;--s:#111721;--s2:#0d1219;--ink:#dbe4ee;--mut:#6d7f92;--line:#1d2836;
--a:#3fe6cd;--a2:#8f80ff;--mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:var(--mono);font-size:14px;line-height:1.5}
.wrap{max-width:1100px;margin:0 auto;padding:18px}
header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;
border-bottom:1px solid var(--line);padding-bottom:12px}
h1{font-size:16px;margin:0;letter-spacing:.06em;display:flex;align-items:center;gap:9px}
h1 b{color:var(--a)}
#dot{width:9px;height:9px;border-radius:50%;background:var(--a);
box-shadow:0 0 8px var(--a);animation:pulse 1.6s ease-in-out infinite}
#dot.dead{background:#c0392b;box-shadow:none;animation:none}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
@media (prefers-reduced-motion:reduce){#dot{animation:none}}
#sub{color:var(--mut);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.stat{background:var(--s);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.1em}
.stat .v{font-size:22px;margin-top:2px;color:var(--a);font-variant-numeric:tabular-nums;
overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.stat .v.small{font-size:14px}
.panel{background:var(--s);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:14px}
.panel h2{font-size:12px;margin:0 0 10px;color:var(--mut);text-transform:uppercase;letter-spacing:.12em}
#blocks{display:flex;gap:6px;overflow-x:auto;padding-bottom:6px}
.blk{flex:0 0 auto;background:var(--s2);border:1px solid var(--line);border-radius:8px;
padding:8px 10px;min-width:96px;text-align:center}
.blk.new{border-color:var(--a);box-shadow:0 0 12px rgba(63,230,205,.25)}
.blk .h{color:var(--a);font-size:15px}.blk .r,.blk .t{color:var(--mut);font-size:10.5px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
td,th{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);color:var(--mut)}
th{text-transform:uppercase;font-size:10.5px;letter-spacing:.08em}
td.hi{color:var(--ink)}
.note{color:var(--mut);font-size:11.5px;margin-top:8px}
</style></head><body><div class="wrap">
<header><h1><span id="dot"></span><b>palimpsest</b> chain</h1>
<div id="sub">connecting&hellip;</div></header>
<div class="grid">
 <div class="stat"><div class="k">height</div><div class="v" id="height">&ndash;</div></div>
 <div class="stat"><div class="k">head</div><div class="v small" id="head">&ndash;</div></div>
 <div class="stat"><div class="k">total supply</div><div class="v" id="supply">&ndash;</div></div>
 <div class="stat"><div class="k">founder balance</div><div class="v" id="founder">&ndash;</div></div>
 <div class="stat"><div class="k">delta mempool</div><div class="v" id="dpool">&ndash;</div></div>
 <div class="stat"><div class="k">account mempool</div><div class="v" id="apool">&ndash;</div></div>
</div>
<div class="panel"><h2>chain &mdash; newest blocks land on the right</h2><div id="blocks"></div></div>
<div class="panel"><h2>data registry &mdash; who feeds the model, and their stake</h2>
<table id="reg"><thead><tr><th>entry</th><th>owner</th><th>type</th><th>size</th>
<th>stake</th><th>weight</th><th>status</th></tr></thead><tbody></tbody></table>
<div class="note">every block: 70% of emission to that block's miners &middot; 10% to the
proposer &middot; 20% split across these entries by weight. entries are challengeable
(stake vs stake) &mdash; this table is the data economy, live.</div></div>
<div class="note" style="margin-top:14px">this dashboard is served by the seed node
itself &mdash; the same process that relays the network. chat-with-the-model arrives
when the serving lane lands on the rust node.</div>
</div><script>
var FOUNDER='3432d48fd6878b4f2e7a1e40cc15e112c512fae7';
var lastH=-1;
function g(u){return fetch(u).then(function(r){return r.json()})}
function poll(){
 Promise.all([g('/status'),g('/chain'),g('/balance?addr='+FOUNDER),g('/data/registry')])
 .then(function(rs){
  var s=rs[0],c=rs[1],b=rs[2],reg=rs[3];
  document.getElementById('dot').className='';
  document.getElementById('sub').textContent=(s.producer?'producer':'seed / relay')+
    ' node · live · miner '+String(s.miner).slice(0,10)+'…';
  document.getElementById('height').textContent=s.height;
  document.getElementById('head').textContent=String(s.head).slice(0,14);
  document.getElementById('supply').textContent=(s.supply/1e9).toLocaleString();
  document.getElementById('founder').textContent=(b.grains/1e9).toLocaleString();
  document.getElementById('dpool').textContent=s.delta_pool;
  document.getElementById('apool').textContent=s.account_pool;
  var bl=document.getElementById('blocks');bl.innerHTML='';
  (c.blocks||[]).forEach(function(x){
    var d=document.createElement('div');
    d.className='blk'+(x.height===s.height&&s.height!==lastH?' new':'');
    d.innerHTML='<div class="h">#'+x.height+'</div><div class="r">'+
      String(x.hash).slice(0,10)+'</div><div class="t">'+x.n_txs+
      ' Δ · '+String(x.proposer).slice(0,8)+'</div>';
    bl.appendChild(d)});
  bl.scrollLeft=bl.scrollWidth;lastH=s.height;
  var tb=document.querySelector('#reg tbody');tb.innerHTML='';
  var R=reg.registry||{};
  Object.keys(R).forEach(function(id){var e=R[id];
    var tr=document.createElement('tr');
    tr.innerHTML='<td class="hi">'+id.slice(0,12)+'</td><td>'+
      String(e.owner).slice(0,12)+'&hellip;</td><td>'+e.media_type+'</td><td>'+
      (e.size||0).toLocaleString()+'</td><td>'+((e.stake||0)/1e9).toLocaleString()+
      '</td><td>'+((e.weight||0)/1e6).toLocaleString()+'M</td><td class="hi">'+
      e.status+'</td>';
    tb.appendChild(tr)});
 }).catch(function(){
  document.getElementById('dot').className='dead';
  document.getElementById('sub').textContent='disconnected… retrying';
 })}
poll();setInterval(poll,5000);
</script></body></html>"#;
