// Builds the Card Scanner's picture library from Scryfall, on GitHub's servers.
// Output (served with the app):
//   lib/version.json  - build stamp, card count
//   lib/cards.json    - compact card list
//   lib/fp.bin        - picture fingerprints (+ neural signatures when available)
// Re-runs only process cards that are new since the last build.
import fs from "node:fs";
import sharp from "sharp";

const OUT = "lib";
const UA = { "User-Agent": "CardScannerLibraryBuilder/1.0", Accept: "application/json" };
const EMB = 256;               // neural signature size (projected)
const REC = 8 + 8 + 27 + EMB;  // bytes per card in fp.bin
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---------- same maths as the scanner (must match exactly) ----------
const COS = []; for (let u = 0; u < 8; u++) { COS[u] = new Float64Array(32); for (let x = 0; x < 32; x++) COS[u][x] = Math.cos((2 * x + 1) * u * Math.PI / 64); }
function phashGray(g) { // g: Float64Array(1024) grayscale 32x32
  const tmp = new Float64Array(8 * 32);
  for (let u = 0; u < 8; u++) for (let y = 0; y < 32; y++) { let s = 0; for (let x = 0; x < 32; x++) s += g[y * 32 + x] * COS[u][x]; tmp[u * 32 + y] = s; }
  const co = new Float64Array(64);
  for (let u = 0; u < 8; u++) for (let w = 0; w < 8; w++) { let s = 0; for (let y = 0; y < 32; y++) s += tmp[u * 32 + y] * COS[w][y]; co[w * 8 + u] = s; }
  const med = Array.from(co).slice(1).sort((a, b) => a - b)[31];
  const h = new Uint32Array(2); for (let i = 1; i < 64; i++) if (co[i] > med) h[i >> 5] |= 1 << (i & 31); return h;
}
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function projMatrix(inDim) { // fixed random projection 1280 -> 256 (same seed in the scanner)
  const r = mulberry32(20261104), m = new Float32Array(inDim * EMB);
  for (let i = 0; i < m.length; i += 2) { const u = Math.max(r(), 1e-12), v = r(); const a = Math.sqrt(-2 * Math.log(u)); m[i] = a * Math.cos(2 * Math.PI * v); if (i + 1 < m.length) m[i + 1] = a * Math.sin(2 * Math.PI * v); }
  return m;
}
function project(vec, M) { const o = new Float32Array(EMB); const n = vec.length; for (let j = 0; j < EMB; j++) { let s = 0; for (let i = 0; i < n; i++) s += vec[i] * M[i * EMB + j]; o[j] = s; }
  let nn = 0; for (const x of o) nn += x * x; nn = Math.sqrt(nn) || 1; const q = new Int8Array(EMB); for (let j = 0; j < EMB; j++) q[j] = Math.max(-127, Math.min(127, Math.round(o[j] / nn * 381))); return q; }

async function region(buf, W, H, x, y, w, h, ow, oh) {
  const left = Math.max(0, Math.round(x)), top = Math.max(0, Math.round(y));
  const width = Math.max(1, Math.min(W - left, Math.round(w))), height = Math.max(1, Math.min(H - top, Math.round(h)));
  return sharp(buf).extract({ left, top, width, height }).resize(ow, oh, { fit: "fill", kernel: "cubic" }).removeAlpha().raw().toBuffer();
}
async function fingerprint(buf) {
  const meta = await sharp(buf).metadata(); const W = meta.width, H = meta.height;
  const gray = raw => { const g = new Float64Array(1024); for (let i = 0; i < 1024; i++) g[i] = raw[i * 3] * .3 + raw[i * 3 + 1] * .59 + raw[i * 3 + 2] * .11; return g; };
  const art = phashGray(gray(await region(buf, W, H, W * .08, H * .11, W * .84, H * .45, 32, 32)));
  const full = phashGray(gray(await region(buf, W, H, 0, 0, W, H, 32, 32)));
  const c9 = await region(buf, W, H, W * .08, H * .11, W * .84, H * .45, 3, 3);
  let m = 0; for (let i = 0; i < 27; i++) m += c9[i]; m = m / 27 || 1;
  const col = new Uint8Array(27); for (let i = 0; i < 27; i++) col[i] = Math.min(255, Math.round(c9[i] / m * 100));
  const artPx = await region(buf, W, H, W * .08, H * .11, W * .84, H * .45, 224, 168);
  return { art, full, col, artPx };
}

// ---------- neural network (optional: the library still works without it) ----------
let tf = null, net = null, M = null;
async function loadNet() {
  try { tf = await import("@tensorflow/tfjs-node"); const mobilenet = await import("@tensorflow-models/mobilenet");
    net = await mobilenet.load({ version: 2, alpha: 1.0 }); M = projMatrix(1280); console.log("Neural network loaded."); }
  catch (e) { console.log("Neural network not available, building fingerprints only:", e.message); net = null; }
}
async function embed(artPx) {
  if (!net) return null;
  const t = tf.tensor3d(artPx, [168, 224, 3], "int32");
  const e = net.infer(t, true); const a = await e.data(); t.dispose(); e.dispose();
  return project(a, M);
}

// ---------- Scryfall ----------
async function getJSON(url) { for (let i = 0; i < 5; i++) { const r = await fetch(url, { headers: UA }); if (r.ok) return r.json(); await sleep(1000 * (i + 1)); } throw new Error("Failed: " + url); }
async function getImg(url) { for (let i = 0; i < 3; i++) { try { const r = await fetch(url, { headers: { "User-Agent": UA["User-Agent"] } }); if (r.ok) return Buffer.from(await r.arrayBuffer()); } catch (e) {} await sleep(500 * (i + 1)); } return null; }

const FLAG = { showcase: 1, extendedart: 2, borderless: 4, basic: 8, token: 16, art: 32, dfc: 64, foil: 128, nonfoil: 256, etched: 512 };
function rowOf(c, face) {
  const f = c.card_faces && c.card_faces[face];
  const img = (c.image_uris && c.image_uris.small) || (f && f.image_uris && f.image_uris.small) || "";
  const fe = c.frame_effects || [], fin = c.finishes || [];
  let flags = 0; if (fe.includes("showcase")) flags |= FLAG.showcase; if (fe.includes("extendedart")) flags |= FLAG.extendedart; if (c.border_color === "borderless") flags |= FLAG.borderless;
  if (/Basic Land/.test(c.type_line || (f && f.type_line) || "")) flags |= FLAG.basic; if (c.layout === "token" || c.layout === "double_faced_token") flags |= FLAG.token; if (c.layout === "art_series") flags |= FLAG.art;
  if (c.card_faces && !c.image_uris) flags |= FLAG.dfc; if (fin.includes("foil")) flags |= FLAG.foil; if (fin.includes("nonfoil")) flags |= FLAG.nonfoil; if (fin.includes("etched")) flags |= FLAG.etched;
  const p = c.prices || {};
  const name = c.layout === "art_series" && c.card_faces ? c.card_faces[0].name : c.name;
  // [key, id, oracle, name, set, number, rarity, flags, usd, usd_foil, image(small, path only), printed_size]
  return [c.id + ":" + face, c.id, c.oracle_id || (c.card_faces && c.card_faces[0].oracle_id) || "", name, c.set, c.collector_number, (c.rarity || "")[0] || "", flags,
    p.usd ? +p.usd : null, p.usd_foil ? +p.usd_foil : (p.usd_etched ? +p.usd_etched : null), img.replace("https://cards.scryfall.io/", "").replace(/\?.*$/, ""), c.printed_size || 0];
}

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  // previous build (reuse its fingerprints)
  const prev = new Map(); let prevHasEmb = false;
  try { const pc = JSON.parse(fs.readFileSync(`${OUT}/cards.json`, "utf8")); const pb = fs.readFileSync(`${OUT}/fp.bin`); prevHasEmb = !!pc.emb;
    pc.cards.forEach((row, i) => prev.set(row[0], pb.subarray(i * REC, (i + 1) * REC))); console.log("Previous library:", prev.size, "cards"); } catch (e) { console.log("No previous library, building from scratch."); }

  const bulk = await getJSON("https://api.scryfall.com/bulk-data");
  const def = bulk.data.find(b => b.type === "default_cards");
  console.log("Downloading Scryfall card data…");
  const all = await getJSON(def.download_uri);
  const sets = {}; const jobs = [];
  for (const c of all) {
    if (c.digital || c.lang !== "en" || c.set_type === "memorabilia" && c.layout !== "art_series") continue;
    const faces = (!c.image_uris && c.card_faces && c.card_faces[0].image_uris) ? c.card_faces.map((_, i) => i) : [0];
    for (const fi of faces) { const row = rowOf(c, fi); if (!row[10]) continue; sets[c.set] = c.set_name; jobs.push(row); }
  }
  console.log("Cards to include:", jobs.length);
  await loadNet();
  const needEmb = !!net;
  const work = jobs.filter(r => !prev.has(r[0]) || (needEmb && !prevHasEmb));
  console.log("New pictures to process:", work.length);

  const recs = new Map(); let done = 0, failed = 0; const t0 = Date.now();
  let idx = 0;
  async function worker() {
    while (idx < work.length) { const row = work[idx++];
      const buf = await getImg("https://cards.scryfall.io/" + row[10]); if (!buf) { failed++; continue; }
      try { const fp = await fingerprint(buf); const e = await embed(fp.artPx);
        const rec = Buffer.alloc(REC); Buffer.from(fp.art.buffer).copy(rec, 0); Buffer.from(fp.full.buffer).copy(rec, 8); Buffer.from(fp.col).copy(rec, 16);
        if (e) Buffer.from(e.buffer).copy(rec, 43); recs.set(row[0], rec); done++;
        if (done % 1000 === 0) console.log(`${done}/${work.length} (${Math.round((Date.now() - t0) / 1000)}s)`);
      } catch (err) { failed++; }
    }
  }
  await Promise.all(Array.from({ length: 8 }, worker));
  console.log("Processed", done, "failed", failed);

  const keep = jobs.filter(r => recs.has(r[0]) || prev.has(r[0]));
  const bin = Buffer.alloc(keep.length * REC);
  keep.forEach((r, i) => { (recs.get(r[0]) || prev.get(r[0])).copy(bin, i * REC); });
  const stamp = new Date().toISOString();
  fs.writeFileSync(`${OUT}/cards.json`, JSON.stringify({ v: stamp, emb: needEmb || prevHasEmb ? EMB : 0, rec: REC, sets, cards: keep }));
  fs.writeFileSync(`${OUT}/fp.bin`, bin);
  fs.writeFileSync(`${OUT}/version.json`, JSON.stringify({ v: stamp, count: keep.length, emb: needEmb || prevHasEmb ? EMB : 0 }));
  console.log("Library written:", keep.length, "cards");
}
main().catch(e => { console.error(e); process.exit(1); });
