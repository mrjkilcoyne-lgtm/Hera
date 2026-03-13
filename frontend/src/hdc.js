/**
 * hdc.js — Hyperdimensional Computing engine (pure JS).
 * Identical API to Rust WASM build. loadEngine() tries WASM first, falls back here.
 *
 * 2048-bit hypervectors · FNV-1a hash · XOR bind · majority-vote bundle · Hamming similarity
 */

const WORDS = 64
const BITS  = WORDS * 32

function fnv1a(str) {
  let h = 2166136261
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, 16777619) >>> 0
  }
  return h
}

function makeRng(seed) {
  let s = seed || 1
  return () => {
    s ^= s << 13; s >>>= 0
    s ^= s >> 17
    s ^= s << 5;  s >>>= 0
    return s
  }
}

function randomHV(seed) {
  const rng = makeRng(seed || 1)
  const hv = new Uint32Array(WORDS)
  for (let i = 0; i < WORDS; i++) hv[i] = rng()
  return hv
}

function bind(a, b) {
  const r = new Uint32Array(WORDS)
  for (let i = 0; i < WORDS; i++) r[i] = a[i] ^ b[i]
  return r
}

function bundle(vecs) {
  if (!vecs.length) return new Uint32Array(WORDS)
  const threshold = (vecs.length + 1) >> 1
  const counts = new Int32Array(BITS)
  for (const hv of vecs)
    for (let w = 0; w < WORDS; w++)
      for (let b = 0; b < 32; b++)
        if ((hv[w] >>> b) & 1) counts[w * 32 + b]++
  const r = new Uint32Array(WORDS)
  for (let w = 0; w < WORDS; w++)
    for (let b = 0; b < 32; b++)
      if (counts[w * 32 + b] >= threshold) r[w] |= (1 << b)
  return r
}

function permute(hv, k) {
  k = ((k % 32) + 32) % 32
  if (!k) return new Uint32Array(hv)
  const r = new Uint32Array(WORDS)
  for (let i = 0; i < WORDS; i++)
    r[i] = ((hv[i] << k) | (hv[i] >>> (32 - k))) >>> 0
  return r
}

function similarity(a, b) {
  let m = 0
  for (let i = 0; i < WORDS; i++) {
    let x = (~(a[i] ^ b[i])) >>> 0
    x -= (x >>> 1) & 0x55555555
    x  = (x & 0x33333333) + ((x >>> 2) & 0x33333333)
    x  = (x + (x >>> 4)) & 0x0f0f0f0f
    m += Math.imul(x, 0x01010101) >>> 24
  }
  return m / BITS
}

class ItemMemory {
  constructor() { this._m = new Map() }
  get(t) {
    if (this._m.has(t)) return this._m.get(t)
    const hv = randomHV(fnv1a(t))
    this._m.set(t, hv)
    return hv
  }
  get size() { return this._m.size }
}

function tokenize(text) {
  return text.toLowerCase().split(/[^a-z]+/).filter(t => t.length > 1)
}

function encode(text, mem) {
  const tokens = tokenize(text)
  if (!tokens.length) return new Uint32Array(WORDS)
  const parts = []
  for (let i = 0; i < tokens.length; i++)
    parts.push(bind(mem.get(tokens[i]), mem.get(`__p${i % 8}`)))
  for (let i = 0; i < tokens.length - 1; i++) {
    const bg = bind(permute(mem.get(tokens[i]), 1), mem.get(tokens[i + 1]))
    parts.push(bg, bg)
  }
  return bundle(parts)
}

export class HDCEngine {
  constructor() { this._mem = new ItemMemory(); this._docs = []; this._id = 0 }

  add_document(text, label = '') {
    const id = this._id++
    this._docs.push({ id, hv: encode(text, this._mem), text, label })
    return id
  }

  search(query, top_k = 5) {
    const q = encode(query, this._mem)
    return JSON.stringify(
      this._docs.map(d => ({ ...d, score: +similarity(q, d.hv).toFixed(4) }))
        .sort((a, b) => b.score - a.score).slice(0, top_k)
        .map(({ id, text, label, score }) => ({ id, text, label, score }))
    )
  }

  similarity_score(a, b) {
    return similarity(encode(a, this._mem), encode(b, this._mem))
  }

  classify(text) {
    const hv = encode(text, this._mem)
    const byLabel = new Map()
    for (const d of this._docs) {
      if (!byLabel.has(d.label)) byLabel.set(d.label, [])
      byLabel.get(d.label).push(d.hv)
    }
    const scores = [...byLabel.entries()]
      .map(([label, vecs]) => ({ label, score: +similarity(hv, bundle(vecs)).toFixed(4) }))
      .sort((a, b) => b.score - a.score)
    const best = scores[0] || { label: '', score: 0 }
    return JSON.stringify({ label: best.label, confidence: best.score, scores })
  }

  analogy(a, b, c, top_k = 5) {
    const result = bind(bind(encode(b, this._mem), encode(a, this._mem)), encode(c, this._mem))
    return JSON.stringify(
      this._docs.map(d => ({ ...d, score: +similarity(result, d.hv).toFixed(4) }))
        .sort((a, b) => b.score - a.score).slice(0, top_k)
        .map(({ text, label, score }) => ({ text, label, score }))
    )
  }

  doc_count()    { return this._docs.length }
  vocab_size()   { return this._mem.size }
  clear_corpus() { this._docs = []; this._id = 0 }
}

export async function loadEngine() {
  // Try WASM — only available after: wasm-pack build --target web in /hdc-engine
  if (typeof window !== 'undefined') {
    try {
      const wasmPath = new URL('../../hdc-engine/pkg/hdc_engine.js', import.meta.url)
      const wasm = await import(/* @vite-ignore */ wasmPath.href)
      await wasm.default()
      console.log('[HDC] Rust WASM engine active')
      return new wasm.HDCEngine()
    } catch {
      console.log('[HDC] JS engine active (build WASM: cd hdc-engine && wasm-pack build --target web --release)')
    }
  }
  return new HDCEngine()
}
