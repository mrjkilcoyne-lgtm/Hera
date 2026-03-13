// ingest.rs — Universal document ingestion pipeline
//
// Handles: PDF, DOCX, XLSX, CSV, TXT, MD, JSON, XML, HTML,
//          images (OCR stub), audio (transcript stub), URLs, raw bytes
// Extensible: add a new arm to `ingest_bytes` for any new type.

use anyhow::{anyhow, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct IngestedDoc {
    pub text:   String,
    pub source: String,
    pub mime:   String,
    pub chunks: Vec<String>,  // pre-split into ingestible pieces
}

impl IngestedDoc {
    fn new(text: String, source: String, mime: String) -> Self {
        let chunks = chunk_text(&text, 512);
        Self { text, source, mime, chunks }
    }
}

/// Split text into overlapping chunks of ~max_words words.
/// Overlap = 20% for context continuity at boundaries.
pub fn chunk_text(text: &str, max_words: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() { return vec![]; }
    if words.len() <= max_words { return vec![text.trim().to_string()]; }

    let overlap = max_words / 5;
    let step = max_words - overlap;
    let mut chunks = Vec::new();
    let mut i = 0;

    while i < words.len() {
        let end = (i + max_words).min(words.len());
        chunks.push(words[i..end].join(" "));
        if end == words.len() { break; }
        i += step;
    }
    chunks
}

/// Main entry point: ingest anything.
/// Pass raw bytes + a filename/URL hint for MIME detection.
pub fn ingest_bytes(bytes: &[u8], source: &str) -> Result<IngestedDoc> {
    let ext = Path::new(source)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "pdf"              => ingest_pdf(bytes, source),
        "docx"             => ingest_docx(bytes, source),
        "xlsx" | "xls"     => ingest_xlsx(bytes, source),
        "csv"              => ingest_csv(bytes, source),
        "json"             => ingest_json(bytes, source),
        "xml"              => ingest_xml(bytes, source),
        "html" | "htm"     => ingest_html(bytes, source),
        "md" | "markdown"  => ingest_text(bytes, source, "text/markdown"),
        "txt" | "log"      => ingest_text(bytes, source, "text/plain"),
        "rs" | "py" | "js" | "ts" | "go" | "c" | "cpp" | "java" => {
            ingest_text(bytes, source, "text/code")
        }
        "png" | "jpg" | "jpeg" | "bmp" | "tiff" | "webp" => {
            ingest_image(bytes, source)
        }
        "mp3" | "wav" | "ogg" | "m4a" | "flac" => {
            ingest_audio(bytes, source)
        }
        // Unknown: try UTF-8 text, fall back to hex summary
        _ => {
            if let Ok(text) = std::str::from_utf8(bytes) {
                ingest_text(bytes, source, "text/plain")
            } else {
                Ok(IngestedDoc::new(
                    format!("[Binary data: {} bytes from {}]", bytes.len(), source),
                    source.to_string(),
                    "application/octet-stream".to_string(),
                ))
            }
        }
    }
}

/// Ingest a file path directly.
pub fn ingest_path(path: &str) -> Result<IngestedDoc> {
    let bytes = std::fs::read(path).map_err(|e| anyhow!("Read error: {}", e))?;
    ingest_bytes(&bytes, path)
}

/// Ingest from a URL (synchronous wrapper — call from async context via spawn_blocking).
pub fn ingest_url_sync(url: &str) -> Result<IngestedDoc> {
    // Minimal sync HTTP using std — full reqwest available in async commands
    Err(anyhow!("Use ingest_url_async in Tauri command context"))
}

// ─── Format handlers ──────────────────────────────────────────────────────────

fn ingest_pdf(bytes: &[u8], source: &str) -> Result<IngestedDoc> {
    // pdf-extract: pure Rust, no system deps
    let text = pdf_extract::extract_text_from_mem(bytes)
        .map_err(|e| anyhow!("PDF error: {}", e))?;
    Ok(IngestedDoc::new(clean_text(&text), source.to_string(), "application/pdf".to_string()))
}

fn ingest_docx(bytes: &[u8], source: &str) -> Result<IngestedDoc> {
    use docx_rust::document::BodyContent;
    let cursor = std::io::Cursor::new(bytes);
    let docx = docx_rust::DocxFile::from_reader(cursor)
        .map_err(|e| anyhow!("DOCX error: {:?}", e))?
        .parse()
        .map_err(|e| anyhow!("DOCX parse: {:?}", e))?;

    let mut text = String::new();
    for content in &docx.document.body.content {
        if let BodyContent::Paragraph(para) = content {
            for run_content in &para.content {
                if let docx_rust::document::ParagraphContent::Run(run) = run_content {
                    for rc in &run.content {
                        if let docx_rust::document::RunContent::Text(t) = rc {
                            text.push_str(&t.text);
                        }
                    }
                }
            }
            text.push('\n');
        }
    }
    Ok(IngestedDoc::new(clean_text(&text), source.to_string(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string()))
}

fn ingest_xlsx(bytes: &[u8], source: &str) -> Result<IngestedDoc> {
    use calamine::{Reader, open_workbook_from_rs, Xlsx};
    let cursor = std::io::Cursor::new(bytes);
    let mut workbook: Xlsx<_> = open_workbook_from_rs(cursor)
        .map_err(|e| anyhow!("XLSX error: {}", e))?;

    let mut text = String::new();
    for sheet_name in workbook.sheet_names().to_vec() {
        text.push_str(&format!("# Sheet: {}\n", sheet_name));
        if let Some(Ok(range)) = workbook.worksheet_range(&sheet_name) {
            for row in range.rows() {
                let cells: Vec<String> = row.iter().map(|c| c.to_string()).collect();
                text.push_str(&cells.join("\t"));
                text.push('\n');
            }
        }
    }
    Ok(IngestedDoc::new(text, source.to_string(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet".to_string()))
}

fn ingest_csv(bytes: &[u8], source: &str) -> Result<IngestedDoc> {
    let mut rdr = csv::Reader::from_reader(bytes);
    let mut text = String::new();
    if let Ok(headers) = rdr.headers() {
        text.push_str(&headers.iter().collect::<Vec<_>>().join("\t"));
        text.push('\n');
    }
    for result in rdr.records() {
        if let Ok(record) = result {
            text.push_str(&record.iter().collect::<Vec<_>>().join("\t"));
            text.push('\n');
        }
    }
    Ok(IngestedDoc::new(text, source.to_string(), "text/csv".to_string()))
}

fn ingest_json(bytes: &[u8], source: &str) -> Result<IngestedDoc> {
    let v: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| anyhow!("JSON error: {}", e))?;
    // Flatten JSON to human-readable key: value pairs
    let mut lines = Vec::new();
    flatten_json(&v, "", &mut lines);
    Ok(IngestedDoc::new(lines.join("\n"), source.to_string(), "application/json".to_string()))
}

fn flatten_json(v: &serde_json::Value, prefix: &str, out: &mut Vec<String>) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, val) in m {
                let key = if prefix.is_empty() { k.clone() } else { format!("{}.{}", prefix, k) };
                flatten_json(val, &key, out);
            }
        }
        serde_json::Value::Array(arr) => {
            for (i, val) in arr.iter().enumerate() {
                flatten_json(val, &format!("{}[{}]", prefix, i), out);
            }
        }
        _ => { out.push(format!("{}: {}", prefix, v)); }
    }
}

fn ingest_xml(bytes: &[u8], source: &str) -> Result<IngestedDoc> {
    // Strip XML tags, extract text content
    let raw = std::str::from_utf8(bytes).map_err(|e| anyhow!("XML UTF-8: {}", e))?;
    let text = strip_tags(raw);
    Ok(IngestedDoc::new(clean_text(&text), source.to_string(), "application/xml".to_string()))
}

fn ingest_html(bytes: &[u8], source: &str) -> Result<IngestedDoc> {
    let raw = std::str::from_utf8(bytes).map_err(|e| anyhow!("HTML UTF-8: {}", e))?;
    // Remove <script> and <style> blocks first
    let no_scripts = regex_remove(raw, r"(?s)<(script|style)[^>]*>.*?</(script|style)>");
    let text = strip_tags(&no_scripts);
    Ok(IngestedDoc::new(clean_text(&text), source.to_string(), "text/html".to_string()))
}

fn ingest_text(bytes: &[u8], source: &str, mime: &str) -> Result<IngestedDoc> {
    let text = String::from_utf8_lossy(bytes).to_string();
    Ok(IngestedDoc::new(clean_text(&text), source.to_string(), mime.to_string()))
}

fn ingest_image(bytes: &[u8], source: &str) -> Result<IngestedDoc> {
    // Stub: in production, call tesseract or a cloud OCR endpoint
    // For now return metadata + size to acknowledge the file
    Ok(IngestedDoc::new(
        format!("[Image: {} bytes from {}. Connect Tesseract OCR or an OCR endpoint to extract text.]",
            bytes.len(), source),
        source.to_string(),
        "image/*".to_string(),
    ))
}

fn ingest_audio(bytes: &[u8], source: &str) -> Result<IngestedDoc> {
    // Stub: in production, pipe through whisper.cpp bindings
    Ok(IngestedDoc::new(
        format!("[Audio: {} bytes from {}. Connect whisper.cpp to transcribe.]",
            bytes.len(), source),
        source.to_string(),
        "audio/*".to_string(),
    ))
}

// ─── Text utilities ──────────────────────────────────────────────────────────

fn clean_text(s: &str) -> String {
    // Collapse whitespace, remove control chars
    let mut out = String::with_capacity(s.len());
    let mut last_space = true;
    for c in s.chars() {
        if c.is_control() && c != '\n' { continue; }
        if c.is_whitespace() {
            if !last_space { out.push(' '); }
            last_space = true;
        } else {
            out.push(c);
            last_space = false;
        }
    }
    out.trim().to_string()
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => { in_tag = true; out.push(' '); }
            '>' => { in_tag = false; out.push(' '); }
            _   => { if !in_tag { out.push(c); } }
        }
    }
    out
}

fn regex_remove(s: &str, pattern: &str) -> String {
    if let Ok(re) = regex::Regex::new(pattern) {
        re.replace_all(s, " ").to_string()
    } else {
        s.to_string()
    }
}
