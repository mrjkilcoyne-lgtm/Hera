// hdc.rs — Hyperdimensional Computing engine + Monte Carlo truth validator
//
// Monte Carlo truth validation:
//   Run 1000 iterations with small random perturbations of the query vector.
//   A response is "validated true" if it appears in top-k in ≥70% of iterations.
//   This eliminates spurious matches that only appear under one specific query framing.
//   Robust semantic matches survive noise; coincidental ones don't.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ─── Constants ────────────────────────────────────────────────────────────────
const WORDS: usize = 64;           // 64 × 64 = 4096 bits
const BITS: f32 = (WORDS * 64) as f32;
const MC_ITERATIONS: usize = 1000;
const MC_NOISE_RATE: f64 = 0.02;   // flip 2% of bits per iteration
const MC_CONSENSUS: f64 = 0.70;    // must appear in 70% of iterations
const MC_TOP_K: usize = 10;        // top-k window per iteration

type HVec = [u64; WORDS];

// ─── Primitives ───────────────────────────────────────────────────────────────

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() { h ^= b as u64; h = h.wrapping_mul(1099511628211); }
    h
}

fn xorshift(mut s: u64) -> impl FnMut() -> u64 {
    move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s }
}

fn random_hv(seed: u64) -> HVec {
    let mut rng = xorshift(seed.max(1));
    let mut hv = [0u64; WORDS];
    for w in hv.iter_mut() { *w = rng(); }
    hv
}

pub fn bind(a: &HVec, b: &HVec) -> HVec {
    let mut r = [0u64; WORDS];
    for i in 0..WORDS { r[i] = a[i] ^ b[i]; }
    r
}

pub fn bundle(vecs: &[HVec]) -> HVec {
    if vecs.is_empty() { return [0u64; WORDS]; }
    let n = vecs.len() as i32;
    let threshold = (n + 1) / 2;
    let mut counts = vec![0i32; WORDS * 64];
    for hv in vecs {
        for (wi, w) in hv.iter().enumerate() {
            for bi in 0..64u32 {
                if (w >> bi) & 1 == 1 { counts[wi * 64 + bi as usize] += 1; }
            }
        }
    }
    let mut r = [0u64; WORDS];
    for (idx, &c) in counts.iter().enumerate() {
        if c >= threshold { r[idx / 64] |= 1u64 << (idx % 64); }
    }
    r
}

pub fn permute(hv: &HVec, k: u32) -> HVec {
    let k = k % 64;
    let mut r = [0u64; WORDS];
    for i in 0..WORDS { r[i] = hv[i].rotate_left(k); }
    r
}

pub fn similarity(a: &HVec, b: &HVec) -> f32 {
    let m: u32 = a.iter().zip(b.iter())
        .map(|(x, y)| (!(x ^ y)).count_ones()).sum();
    m as f32 / BITS
}

/// Perturb a hypervector by flipping ~noise_rate fraction of bits.
/// Used by Monte Carlo validator to simulate query uncertainty.
fn perturb(hv: &HVec, noise_rate: f64, seed: u64) -> HVec {
    let mut rng = xorshift(seed.max(1));
    let mut r = *hv;
    let flip_prob = (noise_rate * u64::MAX as f64) as u64;
    for w in r.iter_mut() {
        let mut mask = 0u64;
        for b in 0..64 {
            if rng() < flip_prob { mask |= 1u64 << b; }
        }
        *w ^= mask;
    }
    r
}

// ─── Item memory ─────────────────────────────────────────────────────────────

pub struct ItemMemory(HashMap<String, HVec>);

impl ItemMemory {
    pub fn new() -> Self { Self(HashMap::new()) }
    pub fn get(&mut self, token: &str) -> HVec {
        *self.0.entry(token.to_string())
            .or_insert_with(|| random_hv(fnv1a(token)))
    }
    pub fn size(&self) -> usize { self.0.len() }
}

// ─── Tokenizer ────────────────────────────────────────────────────────────────

pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphabetic())
        .filter(|t| t.len() > 1)
        .map(str::to_string)
        .collect()
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

pub fn encode(text: &str, mem: &mut ItemMemory) -> HVec {
    let tokens = tokenize(text);
    if tokens.is_empty() { return [0u64; WORDS]; }
    let mut parts: Vec<HVec> = Vec::new();
    for (i, tok) in tokens.iter().enumerate() {
        let tv = mem.get(tok);
        let pv = mem.get(&format!("__p{}", i % 8));
        parts.push(bind(&tv, &pv));
    }
    for i in 0..tokens.len().saturating_sub(1) {
        let h1 = mem.get(&tokens[i]);
        let h2 = mem.get(&tokens[i + 1]);
        let bg = bind(&permute(&h1, 1), &h2);
        parts.push(bg); parts.push(bg);
    }
    bundle(&parts)
}

// ─── Document ─────────────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct Document {
    pub id:     String,
    pub text:   String,
    pub label:  String,
    pub source: String,   // filename, URL, device ID, etc.
    pub mime:   String,
    #[serde(skip)]
    pub hv:     HVec,
}

// ─── Monte Carlo result ───────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
pub struct MCResult {
    pub id:              String,
    pub text:            String,
    pub label:           String,
    pub source:          String,
    pub score:           f32,     // mean similarity across iterations where it appeared
    pub consensus:       f32,     // fraction of iterations in top-k  ∈ [0,1]
    pub validated:       bool,    // consensus ≥ MC_CONSENSUS
    pub iterations_seen: usize,
}

// ─── HDC Engine ──────────────────────────────────────────────────────────────

pub struct HDCEngine {
    pub mem:  ItemMemory,
    pub docs: Vec<Document>,
}

impl HDCEngine {
    pub fn new() -> Self {
        Self { mem: ItemMemory::new(), docs: Vec::new() }
    }

    pub fn add_document(&mut self, id: String, text: &str, label: &str, source: &str, mime: &str) {
        let hv = encode(text, &mut self.mem);
        self.docs.push(Document {
            id, text: text.to_string(),
            label: label.to_string(), source: source.to_string(),
            mime: mime.to_string(), hv,
        });
    }

    pub fn doc_count(&self) -> usize { self.docs.len() }
    pub fn vocab_size(&self) -> usize { self.mem.size() }
    pub fn clear(&mut self) { self.docs.clear(); }

    // ── Raw search (no MC) ───────────────────────────────────────────────────
    pub fn search_raw(&mut self, query: &str, top_k: usize) -> Vec<(f32, &Document)> {
        let qhv = encode(query, &mut self.mem);
        let mut scored: Vec<(f32, &Document)> = self.docs.iter()
            .map(|d| (similarity(&qhv, &d.hv), d)).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    // ── Monte Carlo validated search ─────────────────────────────────────────
    // For each of MC_ITERATIONS iterations:
    //   1. Perturb query vector with small random noise
    //   2. Find top-k documents by similarity
    //   3. Record each document's appearance and score
    // Final result: documents that appear in ≥70% of iterations, ranked by
    // (consensus × mean_score). These are the "always true" responses.
    pub fn search_mc(&mut self, query: &str, top_k: usize) -> Vec<MCResult> {
        let base_qhv = encode(query, &mut self.mem);
        let n_docs = self.docs.len();
        if n_docs == 0 { return vec![]; }

        // Accumulators: [doc_index] → (count_seen, sum_score)
        let mut seen_count = vec![0usize; n_docs];
        let mut sum_score  = vec![0f32;  n_docs];

        for iter in 0..MC_ITERATIONS {
            // Seed from iteration + base query hash (deterministic, reproducible)
            let seed = fnv1a(&format!("mc_{}_{}", iter, query));
            let qhv  = perturb(&base_qhv, MC_NOISE_RATE, seed);

            // Score all docs against perturbed query
            let mut iter_scores: Vec<(f32, usize)> = self.docs.iter().enumerate()
                .map(|(i, d)| (similarity(&qhv, &d.hv), i))
                .collect();
            iter_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Record top-k appearances
            for &(score, idx) in iter_scores.iter().take(MC_TOP_K) {
                seen_count[idx] += 1;
                sum_score[idx]  += score;
            }
        }

        // Build results
        let mut results: Vec<MCResult> = self.docs.iter().enumerate()
            .map(|(i, doc)| {
                let consensus = seen_count[i] as f32 / MC_ITERATIONS as f32;
                let mean_score = if seen_count[i] > 0 {
                    sum_score[i] / seen_count[i] as f32
                } else { 0.0 };
                MCResult {
                    id:              doc.id.clone(),
                    text:            doc.text.clone(),
                    label:           doc.label.clone(),
                    source:          doc.source.clone(),
                    score:           mean_score,
                    consensus,
                    validated:       consensus as f64 >= MC_CONSENSUS,
                    iterations_seen: seen_count[i],
                }
            })
            .filter(|r| r.iterations_seen > 0)
            .collect();

        // Sort: validated first, then by consensus × score
        results.sort_by(|a, b| {
            let wa = a.consensus * a.score;
            let wb = b.consensus * b.score;
            b.validated.cmp(&a.validated)
                .then(wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal))
        });

        results.truncate(top_k);
        results
    }

    // ── Classify with MC ─────────────────────────────────────────────────────
    pub fn classify_mc(&mut self, text: &str) -> serde_json::Value {
        let base_hv = encode(text, &mut self.mem);
        let mut label_consensus: HashMap<String, (usize, f32)> = HashMap::new();

        // Build label prototype once
        let mut by_label: HashMap<String, Vec<HVec>> = HashMap::new();
        for doc in &self.docs {
            by_label.entry(doc.label.clone()).or_default().push(doc.hv);
        }
        let prototypes: HashMap<String, HVec> = by_label.iter()
            .map(|(l, vecs)| (l.clone(), bundle(vecs))).collect();

        for iter in 0..MC_ITERATIONS {
            let seed  = fnv1a(&format!("cls_{}_{}", iter, text));
            let qhv   = perturb(&base_hv, MC_NOISE_RATE, seed);
            let best  = prototypes.iter()
                .map(|(l, p)| (similarity(&qhv, p), l))
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            if let Some((score, label)) = best {
                let e = label_consensus.entry(label.clone()).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += score;
            }
        }

        let mut scores: Vec<serde_json::Value> = label_consensus.iter().map(|(l, (count, sum))| {
            serde_json::json!({
                "label":     l,
                "consensus": *count as f32 / MC_ITERATIONS as f32,
                "score":     sum / *count as f32,
                "validated": (*count as f64 / MC_ITERATIONS as f64) >= MC_CONSENSUS,
            })
        }).collect();

        scores.sort_by(|a, b| {
            let ca = a["consensus"].as_f64().unwrap_or(0.0);
            let cb = b["consensus"].as_f64().unwrap_or(0.0);
            cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = scores.first().cloned().unwrap_or(serde_json::json!({"label":"","consensus":0}));
        serde_json::json!({
            "label":     best["label"],
            "consensus": best["consensus"],
            "validated": best["validated"],
            "scores":    scores,
            "iterations": MC_ITERATIONS,
        })
    }

    // ── Similarity with MC error bars ────────────────────────────────────────
    pub fn similarity_mc(&mut self, a: &str, b: &str) -> serde_json::Value {
        let ha = encode(a, &mut self.mem);
        let hb = encode(b, &mut self.mem);
        let base = similarity(&ha, &hb);

        let mut samples = Vec::with_capacity(MC_ITERATIONS);
        for iter in 0..MC_ITERATIONS {
            let sa  = perturb(&ha, MC_NOISE_RATE, fnv1a(&format!("sa_{}_{}", iter, a)));
            let sb  = perturb(&hb, MC_NOISE_RATE, fnv1a(&format!("sb_{}_{}", iter, b)));
            samples.push(similarity(&sa, &sb));
        }
        let mean: f32 = samples.iter().sum::<f32>() / MC_ITERATIONS as f32;
        let variance: f32 = samples.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / MC_ITERATIONS as f32;
        let std_dev = variance.sqrt();

        let interp = match base {
            s if s > 0.75 => "strongly similar",
            s if s > 0.60 => "related",
            s if s > 0.50 => "weakly related",
            _ => "dissimilar"
        };

        serde_json::json!({
            "score":    base,
            "mean_mc":  mean,
            "std_dev":  std_dev,
            "ci_low":   (mean - 2.0 * std_dev).max(0.0),
            "ci_high":  (mean + 2.0 * std_dev).min(1.0),
            "interpretation": interp,
            "iterations": MC_ITERATIONS,
        })
    }
}
