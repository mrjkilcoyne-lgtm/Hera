// connectors.rs — Device and data stream connectors

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataEvent {
    pub source:    String,
    pub device_id: String,
    pub raw:       String,
    pub label:     String,
    pub timestamp: u64,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default().as_millis() as u64
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerialConfig { pub port: String, pub baud: u32, pub label: String }

pub fn list_serial_ports() -> Vec<String> {
    serialport::available_ports().unwrap_or_default()
        .iter().map(|p| p.port_name.clone()).collect()
}

pub fn connect_serial(cfg: SerialConfig, tx: mpsc::Sender<DataEvent>) -> Result<()> {
    let port = serialport::new(&cfg.port, cfg.baud)
        .timeout(std::time::Duration::from_millis(100))
        .open().map_err(|e| anyhow::anyhow!("Serial error: {}", e))?;
    let label = cfg.label.clone();
    let device_id = cfg.port.clone();
    tokio::task::spawn_blocking(move || {
        let mut reader = std::io::BufReader::new(port);
        let mut line = String::new();
        loop {
            line.clear();
            match std::io::BufRead::read_line(&mut reader, &mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let trimmed = line.trim().to_string();
                    if !trimmed.is_empty() {
                        let _ = tx.blocking_send(DataEvent {
                            source: "serial".to_string(), device_id: device_id.clone(),
                            raw: trimmed, label: label.clone(), timestamp: now_ms(),
                        });
                    }
                }
                Err(_) => break,
            }
        }
    });
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpPollConfig { pub url: String, pub interval_secs: u64, pub label: String, pub headers: Vec<(String, String)> }

pub async fn connect_http_poll(cfg: HttpPollConfig, tx: mpsc::Sender<DataEvent>) {
    let client = reqwest::Client::new();
    loop {
        let mut req = client.get(&cfg.url);
        for (k, v) in &cfg.headers { req = req.header(k.as_str(), v.as_str()); }
        if let Ok(resp) = req.send().await {
            if let Ok(text) = resp.text().await {
                let _ = tx.send(DataEvent {
                    source: "http-poll".to_string(), device_id: cfg.url.clone(),
                    raw: text, label: cfg.label.clone(), timestamp: now_ms(),
                }).await;
            }
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(cfg.interval_secs)).await;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsConfig { pub url: String, pub label: String }

pub async fn connect_websocket(cfg: WsConfig, tx: mpsc::Sender<DataEvent>) {
    use tokio_tungstenite::connect_async;
    use futures_util::StreamExt;

    if let Ok((stream, _)) = connect_async(&cfg.url).await {
        let (_, mut read) = stream.split();
        while let Some(Ok(msg)) = read.next().await {
            if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                let _ = tx.send(DataEvent {
                    source: "websocket".to_string(), device_id: cfg.url.clone(),
                    raw: text.to_string(), label: cfg.label.clone(), timestamp: now_ms(),
                }).await;
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqttConfig { pub host: String, pub port: u16, pub client_id: String, pub topic: String, pub label: String }

pub async fn connect_mqtt(cfg: MqttConfig, tx: mpsc::Sender<DataEvent>) {
    use rumqttc::{AsyncClient, MqttOptions, QoS, Event, Packet};
    let mut opts = MqttOptions::new(&cfg.client_id, &cfg.host, cfg.port);
    opts.set_keep_alive(std::time::Duration::from_secs(30));
    let (client, mut eventloop) = AsyncClient::new(opts, 64);
    if client.subscribe(&cfg.topic, QoS::AtMostOnce).await.is_err() { return; }
    loop {
        match eventloop.poll().await {
            Ok(Event::Incoming(Packet::Publish(p))) => {
                let _ = tx.send(DataEvent {
                    source: "mqtt".to_string(),
                    device_id: format!("{}:{}/{}", cfg.host, cfg.port, cfg.topic),
                    raw: String::from_utf8_lossy(&p.payload).to_string(),
                    label: cfg.label.clone(), timestamp: now_ms(),
                }).await;
            }
            Err(_) => { tokio::time::sleep(tokio::time::Duration::from_secs(5)).await; }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchConfig { pub path: String, pub label: String, pub recursive: bool }

pub fn connect_file_watch(cfg: WatchConfig, tx: mpsc::Sender<DataEvent>) -> Result<()> {
    use notify::{Watcher, RecursiveMode, recommended_watcher, Event as NE, EventKind};
    let label = cfg.label.clone();
    let mut watcher = recommended_watcher(move |res: notify::Result<NE>| {
        if let Ok(event) = res {
            if matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_)) {
                for path in event.paths {
                    let path_str = path.to_string_lossy().to_string();
                    let _ = tx.blocking_send(DataEvent {
                        source: "file-watch".to_string(), device_id: path_str.clone(),
                        raw: path_str, label: label.clone(), timestamp: now_ms(),
                    });
                }
            }
        }
    })?;
    let mode = if cfg.recursive { RecursiveMode::Recursive } else { RecursiveMode::NonRecursive };
    watcher.watch(std::path::Path::new(&cfg.path), mode)?;
    std::mem::forget(watcher);
    Ok(())
}
