//! The trainer bridge — the consensus boundary as a socket.
//!
//! The node owns consensus and networking; training is an UNCONSTRAINED local
//! compute plugin (§6.3): a PyTorch process (client/miner_bridge.py) connects
//! on localhost, receives the head state once, then per round trains and
//! returns a COMPRESSED quantized delta. When the head advances, the node
//! sends the sparse state difference so the bridge stays synced without ever
//! re-shipping the full model.
//!
//! Frames: [u32 BE length][bytes]. Control messages are JSON; the one big blob
//! (the initial state) follows a {"bin_next": true} message as a raw frame of
//! i64-LE. Everything else (payloads, sparse advances) is small enough to ride
//! base64 inside JSON.

use crate::proto::{Payload, SparseI64};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tracing::{info, warn};

#[derive(Debug)]
pub enum ToBridge {
    State { height: u64, state: Vec<i64> },
    Train { height: u64, seed: u64 },
    Advance { height: u64, sparse: SparseI64 },
}

#[derive(Debug)]
pub enum FromBridge {
    Connected,
    Delta { height: u64, loss: f64, payload: Payload },
    NeedState,
}

async fn write_frame(s: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    s.write_all(&(bytes.len() as u32).to_be_bytes()).await?;
    s.write_all(bytes).await
}

async fn read_frame(s: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut len = [0u8; 4];
    s.read_exact(&mut len).await?;
    let n = u32::from_be_bytes(len) as usize;
    let mut buf = vec![0u8; n];
    s.read_exact(&mut buf).await?;
    Ok(buf)
}

async fn serve_one(
    mut sock: TcpStream,
    cmds: &mut mpsc::Receiver<ToBridge>,
    events: &mpsc::Sender<FromBridge>,
) -> std::io::Result<()> {
    // handshake: the bridge speaks first
    let hello: Value = serde_json::from_slice(&read_frame(&mut sock).await?)?;
    if hello["t"] != "hello" {
        return Ok(());
    }
    info!("trainer bridge connected");
    let _ = events.send(FromBridge::Connected).await;
    loop {
        tokio::select! {
            cmd = cmds.recv() => {
                let Some(cmd) = cmd else { return Ok(()) };
                match cmd {
                    ToBridge::State { height, state } => {
                        let head = json!({"t": "state", "height": height,
                                          "n": state.len(), "bin_next": true});
                        write_frame(&mut sock, head.to_string().as_bytes()).await?;
                        write_frame(&mut sock,
                                    &palimpsest_core::int64_bytes(&state)).await?;
                    }
                    ToBridge::Train { height, seed } => {
                        let m = json!({"t": "train", "height": height, "seed": seed});
                        write_frame(&mut sock, m.to_string().as_bytes()).await?;
                    }
                    ToBridge::Advance { height, sparse } => {
                        let m = json!({"t": "advance", "height": height,
                                       "sparse": sparse});
                        write_frame(&mut sock, m.to_string().as_bytes()).await?;
                    }
                }
            }
            frame = read_frame(&mut sock) => {
                let v: Value = serde_json::from_slice(&frame?)?;
                match v["t"].as_str() {
                    Some("delta") => {
                        let payload: Payload =
                            serde_json::from_value(v["payload"].clone())
                            .map_err(std::io::Error::other)?;
                        let _ = events.send(FromBridge::Delta {
                            height: v["height"].as_u64().unwrap_or(0),
                            loss: v["loss"].as_f64().unwrap_or(0.0),
                            payload,
                        }).await;
                    }
                    Some("resync") => { let _ = events.send(FromBridge::NeedState).await; }
                    _ => {}
                }
            }
        }
    }
}

/// Run the bridge listener forever; one trainer at a time, reconnects welcome.
pub async fn run(
    port: u16,
    mut cmds: mpsc::Receiver<ToBridge>,
    events: mpsc::Sender<FromBridge>,
) {
    let listener = match TcpListener::bind(("127.0.0.1", port)).await {
        Ok(l) => l,
        Err(e) => {
            warn!("bridge listener failed: {e}");
            return;
        }
    };
    info!("trainer bridge listening on 127.0.0.1:{port}");
    loop {
        match listener.accept().await {
            Ok((sock, _)) => {
                if let Err(e) = serve_one(sock, &mut cmds, &events).await {
                    warn!("bridge connection ended: {e}");
                }
            }
            Err(e) => warn!("bridge accept error: {e}"),
        }
    }
}
