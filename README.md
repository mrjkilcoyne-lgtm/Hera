# HERA
### Hyperdimensional Engine for Reasoning and Analysis

Offline semantic reasoning for any data. No cloud. No GPU. No API key.

---

## What it does

- **Ingest anything** — PDF, DOCX, XLSX, CSV, JSON, XML, HTML, TXT, MD, images, audio, URLs, paste
- **Live device data** — Serial/USB, HTTP polling, WebSocket, MQTT, file system watch
- **Monte Carlo truth validation** — 1000 iterations, 2% noise per query. Only responses appearing in ≥70% of iterations are marked validated. Structurally robust answers, not coincidental ones.
- **Four reasoning modes** — Semantic search · Classification · Similarity (with confidence intervals) · Analogy

## Install (Windows)

### Option 1: Download pre-built MSI
1. Go to [Releases](../../releases)
2. Download `HERA_x.x.x_x64_en-US.msi`
3. Run installer

### Option 2: Build from source

**Prerequisites:** [Rust](https://rustup.rs) + [Node 20+](https://nodejs.org)

```powershell
git clone https://github.com/yourrepo/hera
cd hera
npm install
npm run tauri:build
# Installer: src-tauri/target/release/bundle/msi/HERA_*.msi
```

### Option 3: Batch install (enterprise)
```powershell
# Silent install, per-machine
msiexec /i HERA_0.1.0_x64_en-US.msi /qn ALLUSERS=1
```

### Dev mode (browser, no install)
```bash
npm run dev
# Open http://localhost:5173
# Uses JS HDC engine — full WASM/Rust features require Tauri build
```

---

## Architecture

```
hera/
├── src-tauri/
│   ├── src/
│   │   ├── main.rs          Tauri entry, all commands
│   │   ├── hdc.rs           HDC engine + Monte Carlo validator
│   │   ├── ingest.rs        Universal file ingestion pipeline
│   │   └── connectors.rs    Device/stream connectors
│   ├── Cargo.toml
│   └── tauri.conf.json
├── frontend/
│   └── index.html           UI (single file, no framework)
├── .github/workflows/
│   └── build.yml            Auto-build MSI on git tag push
└── package.json
```

---

## How Monte Carlo validation works

```
query → encode → base hypervector Q
for i in 1..1000:
    seed = hash(i + query)
    Q_perturbed = flip ~2% of Q's bits (deterministic noise)
    top_k = similarity_search(Q_perturbed, corpus)
    record which documents appear

final result:
    doc.consensus = appearances / 1000
    doc.validated = consensus ≥ 0.70  ← "always true" threshold
```

A document scoring 52% similarity under one exact query framing might only appear in 30% of iterations under noise → **not validated**.

A document that genuinely answers the question appears in 85% of iterations → **validated ✓**.

This is semantic robustness testing, not confidence scoring. It eliminates responses that depend on lucky word overlap.

---

## Connecting devices

| Type | Config | Example |
|------|--------|---------|
| Serial/USB | Port + baud | `COM3`, `9600` |
| HTTP poll | URL + interval | `https://api.example.com/data`, `30s` |
| WebSocket | URL | `ws://localhost:8080/stream` |
| MQTT | Broker + topic | `broker.hivemq.com:1883`, `sensors/#` |
| File watch | Directory path | `C:\logs\incoming` |

Each connector labels incoming data automatically and ingests it into the HDC corpus in real-time.

---

## The HDC engine

**4096-bit hypervectors** (64 × `u64`)

| Operation | Implementation | Property |
|-----------|----------------|----------|
| Bind | XOR | Invertible: `bind(A, bind(A,B)) = B` |
| Bundle | Majority vote | Similarity-preserving superposition |
| Permute | Bit rotation | Position-sensitive encoding |
| Similarity | Hamming distance | ∈ [0,1] · ~0.5 = orthogonal |

Documents encoded as bundle of position-bound unigrams + double-weighted bigrams.

**Data requirement:** Works from 10 documents.  
**Memory:** ~512 bytes per document.  
**Speed:** <1ms per query on modern CPU.  
**Training:** None. The structure is algebraic, not statistical.

---

## Adding file type support

In `src-tauri/src/ingest.rs`, add a new arm to `ingest_bytes()`:

```rust
"myext" => ingest_myformat(bytes, source),
```

Then implement `fn ingest_myformat(bytes: &[u8], source: &str) -> Result<IngestedDoc>`.

---

## Roadmap

- [ ] Whisper.cpp integration for audio transcription
- [ ] Tesseract OCR for image text extraction  
- [ ] BLE (Bluetooth Low Energy) connector
- [ ] Export corpus to JSON / reimport
- [ ] Persistent corpus (SQLite store)
- [ ] Multi-corpus / namespace support
- [ ] Streaming ingest display

---

*Built on hyperdimensional computing. Zero gradient descent. Zero cloud dependency. Zero training epochs.*
