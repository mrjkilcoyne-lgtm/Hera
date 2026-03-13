// connectors.rs — Universal device and data stream connectors
//
// Supported: Serial/USB, HTTP polling, WebSocket, MQTT, file watch
// Each connector produces a stream of text events → ingested into HDC

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataEvent {
    pub source:    String,   // connector ID
    pub device_id: String,   // device/endpoint identifier
    pub raw:       String,   // raw text/JSON payload
    pub label:     String,   // auto-label for HDC
    pub timestamp: u64,      // unix millis
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ─── Serial / USB ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerialConfig {
    pub port:     String,   // e.g. "COM3" on Windows, "/dev/ttyUSB0" on Linux
    pub baud:     u32,
    pub label:    String,
}

/// List available serial ports.
pub fn list_serial_ports() -> Vec<String> {
    serialport::available_ports()
        .unwrap_or_default()
        .iter()
        .map(|p| p.port_name.clone())
        .collect()
}

/// Start reading from a serial port. Sends lines to tx channel.
/// Runs in its own thread; drop the returned handle to stop.
pub fn connect_serial(cfg: SerialConfig, tx: mpsc::Sender<DataEvent>) -> Result<()> {
    let port = serialport::new(&cfg.port, cfg.baud)
        .timeout(std::time::Duration::from_millis(100))
        .open()
        .map_err(|e| anyhow::anyhow!("Serial open error: {}", e))?;

    let label = cfg.label.clone();
    let device_id = cfg.port.clone();

    tokio::task::spawn_blocking(move || {
        let mut reader = std::io::BufReader::new(port);
        let mut line = String::new();
        loop {
            line.clear();
            match std::io::BufRead::read_line(&mut reader, &mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let trimmed = line.trim().to_string();
                    if !trimmed.is_empty() {
                        let _ = tx.blocking_send(DataEvent {
                            source:    "serial".to_string(),
                            device_id: device_id.clone(),
                            raw:       trimmed,
                            label:     label.clone(),
                            timestamp: now_ms(),
                        });
                    }
                }
                Err(_) => break,
            }
        }
    });
    Ok(())
}

// ─── HTTP polling ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpPollConfig {
    pub url:           String,
    pub interval_secs: u64,
    pub label:         String,
    pub headers:       Vec<(String, String)>,
}

/// Poll an HTTP endpoint every interval_secs, send text to tx.
pub async fn connect_http_poll(cfg: HttpPollConfig, tx: mpsc::Sender<DataEvent>) {
    let client = reqwest::Client::new();
    loop {
        let mut req = client.get(&cfg.url);
        for (k, v) in &cfg.headers { req = req.header(k, v); }

        match req.send().await {
            Ok(resp) => {
                if let Ok(text) = resp.text().await {
                    let _ = tx.send(DataEvent {
                        source:    "http-poll".to_string(),
                        device_id: cfg.url.clone(),
                        raw:       text,
                        label:     cfg.label.clone(),
                        timestamp: now_ms(),
                    }).await;
                }
            }
            Err(e) => { eprintln!("[http-poll] error: {}", e); }
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(cfg.interval_secs)).await;
    }
}

// ─── WebSocket stream ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsConfig {
    pub url:   String,
    pub label: String,
}

/// Connect to a WebSocket, forward all text messages to tx.
pub async fn connect_websocket(cfg: WsConfig, tx: mpsc::Sender<DataEvent>) {
    use tokio_tungstenite::connect_async;
    use futures_util::StreamExt;

    match connect_async(&cfg.url).await {
        Ok((stream, _)) => {
            let (_, mut read) = stream.split();
            while let Some(msg) = read.next().await {
                if let Ok(tokio_tungstenite::tungstenite::Message::Text(text)) = msg {
                    let _ = tx.send(DataEvent {
                        source:    "websocket".to_string(),
                        device_id: cfg.url.clone(),
                        raw:       text,
                        label:     cfg.label.clone(),
                        timestamp: now_ms(),
                    }).await;
                }
            }
        }
        Err(e) => { eprintln!("[websocket] connect error: {}", e); }
    }
}

// ─── MQTT ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqttConfig {
    pub host:      String,
    pub port:      u16,
    pub client_id: String,
    pub topic:     String,
    pub label:     String,
}

/// Subscribe to an MQTT topic, forward messages to tx.
pub async fn connect_mqtt(cfg: MqttConfig, tx: mpsc::Sender<DataEvent>) {
    use rumqttc::{AsyncClient, MqttOptions, QoS, Event, Packet};

    let mut opts = MqttOptions::new(&cfg.client_id, &cfg.host, cfg.port);
    opts.set_keep_alive(std::time::Duration::from_secs(30));
    let (client, mut eventloop) = AsyncClient::new(opts, 64);

    if let Err(e) = client.subscribe(&cfg.topic, QoS::AtMostOnce).await {
        eprintln!("[mqtt] subscribe error: {}", e); return;
    }

    loop {
        match eventloop.poll().await {
            Ok(Event::Incoming(Packet::Publish(p))) => {
                let text = String::from_utf8_lossy(&p.payload).to_string();
                let _ = tx.send(DataEvent {
                    source:    "mqtt".to_string(),
                    device_id: format!("{}:{}/{}", cfg.host, cfg.port, cfg.topic),
                    raw:       text,
                    label:     cfg.label.clone(),
                    timestamp: now_ms(),
                }).await;
            }
            Err(e) => {
                eprintln!("[mqtt] error: {}", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
            _ => {}
        }
    }
}

// ─── File watcher ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchConfig {
    pub path:      String,
    pub label:     String,
    pub recursive: bool,
}

/// Watch a directory for new/modified files. Send file paths to tx.
/// Caller handles ingestion from the path string in DataEvent.raw.
pub fn connect_file_watch(cfg: WatchConfig, tx: mpsc::Sender<DataEvent>) -> Result<()> {
    use notify::{Watcher, RecursiveMode, recommended_watcher, Event as NotifyEvent, EventKind};
    use std::sync::Arc;

    let tx_clone = tx.clone();
    let label = cfg.label.clone();

    let mut watcher = recommended_watcher(move |res: notify::Result<NotifyEvent>| {
        if let Ok(event) = res {
            if matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_)) {
                for path in event.paths {
                    let path_str = path.to_string_lossy().to_string();
                    let _ = tx_clone.blocking_send(DataEvent {
                        source:    "file-watch".to_string(),
                        device_id: path_str.clone(),
                        raw:       path_str,  // caller ingests from this path
                        label:     label.clone(),
                        timestamp: now_ms(),
                    });
                }
            }
        }
    })?;

    let mode = if cfg.recursive { RecursiveMode::Recursive } else { RecursiveMode::NonRecursive };
    watcher.watch(std::path::Path::new(&cfg.path), mode)?;

    // Keep watcher alive — leak intentionally (it runs forever)
    std::mem::forget(watcher);
    Ok(())
}

// ─── Connector registry ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ConnectorConfig {
    Serial(SerialConfig),
    HttpPoll(HttpPollConfig),
    WebSocket(WsConfig),
    Mqtt(MqttConfig),
    FileWatch(WatchConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorStatus {
    pub id:      String,
    pub kind:    String,
    pub target:  String,
    pub active:  bool,
    pub events:  usize,
}
