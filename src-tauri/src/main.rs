// main.rs — HERA Tauri application entry point
// All Tauri commands exposed to the frontend live here.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod hdc;
mod ingest;
mod connectors;

use std::sync::Mutex;
use tauri::State;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ─── App state ────────────────────────────────────────────────────────────────

struct AppState(Mutex<hdc::HDCEngine>);

// ─── Document commands ────────────────────────────────────────────────────────

#[tauri::command]
fn add_text(state: State<AppState>, text: String, label: String, source: String) -> String {
    let mut engine = state.0.lock().unwrap();
    let id = Uuid::new_v4().to_string();
    engine.add_document(id.clone(), &text, &label, &source, "text/plain");
    format!(r#"{{"id":"{}","chunks":1}}"#, id)
}

#[tauri::command]
fn add_document_bytes(
    state:  State<AppState>,
    bytes:  Vec<u8>,
    source: String,
    label:  String,
) -> Result<String, String> {
    let ingested = ingest::ingest_bytes(&bytes, &source)
        .map_err(|e| e.to_string())?;

    let mut engine = state.0.lock().unwrap();
    let mut ids = Vec::new();
    for (i, chunk) in ingested.chunks.iter().enumerate() {
        let id = format!("{}-{}", Uuid::new_v4(), i);
        engine.add_document(id.clone(), chunk, &label, &source, &ingested.mime);
        ids.push(id);
    }

    Ok(serde_json::json!({
        "source": source,
        "mime":   ingested.mime,
        "chunks": ids.len(),
        "ids":    ids,
    }).to_string())
}

#[tauri::command]
fn add_document_path(
    state: State<AppState>,
    path:  String,
    label: String,
) -> Result<String, String> {
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    add_document_bytes(state, bytes, path, label)
}

#[tauri::command]
fn add_url(
    state: State<AppState>,
    url:   String,
    label: String,
) -> Result<String, String> {
    // Synchronous HTTP fetch — acceptable for on-demand URL ingest
    let response = reqwest::blocking::get(&url).map_err(|e| e.to_string())?;
    let bytes = response.bytes().map_err(|e| e.to_string())?.to_vec();

    // Determine source extension from URL or Content-Type
    let source = if url.contains('.') { url.clone() } else { format!("{}.html", url) };
    add_document_bytes(state, bytes, source, label)
}

// ─── Query commands ───────────────────────────────────────────────────────────

/// Fast search — no Monte Carlo, instant results.
#[tauri::command]
fn search(state: State<AppState>, query: String, top_k: usize) -> String {
    let mut engine = state.0.lock().unwrap();
    let results = engine.search_raw(&query, top_k);
    let items: Vec<serde_json::Value> = results.iter().map(|(score, doc)| {
        serde_json::json!({
            "id":     doc.id,
            "text":   doc.text,
            "label":  doc.label,
            "source": doc.source,
            "score":  score,
        })
    }).collect();
    serde_json::json!({"results": items, "mode": "fast"}).to_string()
}

/// Monte Carlo validated search — 1000 iterations, only returns "always true" responses.
#[tauri::command]
fn search_validated(state: State<AppState>, query: String, top_k: usize) -> String {
    let mut engine = state.0.lock().unwrap();
    let results = engine.search_mc(&query, top_k);
    serde_json::json!({
        "results":    results,
        "mode":       "validated",
        "iterations": 1000,
        "threshold":  0.70,
    }).to_string()
}

#[tauri::command]
fn classify(state: State<AppState>, text: String) -> String {
    let mut engine = state.0.lock().unwrap();
    engine.classify_mc(&text).to_string()
}

#[tauri::command]
fn similarity(state: State<AppState>, a: String, b: String) -> String {
    let mut engine = state.0.lock().unwrap();
    engine.similarity_mc(&a, &b).to_string()
}

#[tauri::command]
fn analogy(state: State<AppState>, a: String, b: String, c: String, top_k: usize) -> String {
    let mut engine = state.0.lock().unwrap();
    let result = hdc::bind(
        &hdc::bind(&hdc::encode(&b, &mut engine.mem), &hdc::encode(&a, &mut engine.mem)),
        &hdc::encode(&c, &mut engine.mem),
    );
    let mut scored: Vec<serde_json::Value> = engine.docs.iter()
        .map(|d| {
            let s = hdc::similarity(&result, &d.hv);
            serde_json::json!({"text": d.text, "label": d.label, "score": s})
        })
        .collect();
    scored.sort_by(|a, b| b["score"].as_f64().unwrap_or(0.0)
        .partial_cmp(&a["score"].as_f64().unwrap_or(0.0))
        .unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    serde_json::json!({"results": scored, "analogy": format!("{} : {} :: {} : ?", a, b, c)}).to_string()
}

// ─── Corpus management ────────────────────────────────────────────────────────

#[tauri::command]
fn get_stats(state: State<AppState>) -> String {
    let engine = state.0.lock().unwrap();
    serde_json::json!({
        "doc_count":  engine.doc_count(),
        "vocab_size": engine.vocab_size(),
    }).to_string()
}

#[tauri::command]
fn clear_corpus(state: State<AppState>) {
    state.0.lock().unwrap().clear();
}

// ─── Device commands ──────────────────────────────────────────────────────────

#[tauri::command]
fn list_serial_ports() -> Vec<String> {
    connectors::list_serial_ports()
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    tracing_subscriber::fmt::init();

    tauri::Builder::default()
        .manage(AppState(Mutex::new(hdc::HDCEngine::new())))
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_notification::init())
        .invoke_handler(tauri::generate_handler![
            add_text,
            add_document_bytes,
            add_document_path,
            add_url,
            search,
            search_validated,
            classify,
            similarity,
            analogy,
            get_stats,
            clear_corpus,
            list_serial_ports,
        ])
        .run(tauri::generate_context!())
        .expect("HERA startup failed");
}
