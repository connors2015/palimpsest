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
        .route("/status", get(status))
        .route("/balance", get(balance))
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
