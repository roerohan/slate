/**
 * Builds a curated dataset of HTML-snippet / screenshot pairs sourced from
 * HuggingFaceM4/WebSight v0.2.
 *
 * Pipeline:
 *   1. Page through the HF dataset-server REST API collecting candidate rows.
 *   2. Discard anything whose usable snippet exceeds the token-budget char limit.
 *   3. Render each candidate via Cloudflare Browser Rendering and drop blank frames.
 *   4. Persist screenshots + a JSON manifest to disk.
 */

import "dotenv/config";
import fs from "fs";
import path from "path";
import sharp from "sharp";
import Cloudflare from "cloudflare";

import cfg from "./config.js";

// ── Deterministic PRNG (mulberry32) ─────────────────────────────────────────

function createRng(seed: number) {
  let state = seed | 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let z = Math.imul(state ^ (state >>> 15), 1 | state);
    z = (z + Math.imul(z ^ (z >>> 7), 61 | z)) ^ z;
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle<T>(items: T[], rng: () => number): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// ── Logging ─────────────────────────────────────────────────────────────────

function elapsed(startMs: number): string {
  const s = ((performance.now() - startMs) / 1000).toFixed(1);
  return `${s}s`;
}

function log(msg: string) {
  const ts = new Date().toISOString().slice(11, 19);
  console.log(`[${ts}] ${msg}`);
}

// ── HTML parsing helpers ────────────────────────────────────────────────────

const FULL_DOC_RE = /^\s*(<(!doctype|html))/i;
const BODY_RE = /<body[^>]*>([\s\S]*)<\/body>/i;
const STYLE_RE = /<style[^>]*>([\s\S]*?)<\/style>/gi;

function looksLikeFullDocument(src: string): boolean {
  return FULL_DOC_RE.test(src);
}

/** Pull the inner-body markup out of a full HTML document. */
function isolateBody(doc: string): string | undefined {
  return BODY_RE.exec(doc)?.[1]?.trim();
}

/** Collect all <style> blocks and merge them into one. */
function gatherStyles(doc: string): string {
  const blocks: string[] = [];
  let hit: RegExpExecArray | null;
  while ((hit = STYLE_RE.exec(doc))) blocks.push(hit[1]);
  STYLE_RE.lastIndex = 0; // reset stateful regex
  return blocks.length ? `<style>${blocks.join("\n")}</style>` : "";
}

/**
 * For full HTML documents, extract a compact snippet (styles + body content).
 * For fragments, return as-is.  Returns `undefined` when the document can't
 * be reduced under the char budget.
 */
function toSnippet(raw: string): { snippet: string; full: string } | undefined {
  if (!looksLikeFullDocument(raw)) {
    return raw.length <= cfg.maxSnippetLength
      ? { snippet: raw, full: raw }
      : undefined;
  }

  const body = isolateBody(raw);
  if (!body) return undefined;

  const styles = gatherStyles(raw);
  const combined = (styles ? `${styles}\n${body}` : body).trim();
  return combined.length <= cfg.maxSnippetLength
    ? { snippet: combined, full: raw }
    : undefined;
}

// ── HuggingFace dataset-server client ───────────────────────────────────────

const HF_ROWS_URL = "https://datasets-server.huggingface.co/rows";
const HF_PAGE_SIZE = 100; // API maximum

interface DatasetPage {
  rows: { row_idx: number; row: { text: string; [k: string]: unknown } }[];
  num_rows_total: number;
}

async function fetchPage(offset: number, count: number): Promise<DatasetPage> {
  const url = new URL(HF_ROWS_URL);
  url.searchParams.set("dataset", "HuggingFaceM4/WebSight");
  url.searchParams.set("config", "v0.2");
  url.searchParams.set("split", "train");
  url.searchParams.set("offset", String(offset));
  url.searchParams.set("length", String(count));

  const resp = await fetch(url);
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`HF dataset-server ${resp.status}: ${body}`);
  }
  return resp.json() as Promise<DatasetPage>;
}

interface Candidate {
  snippet: string;
  fullHtml: string;
  rowIndex: number;
}

async function collectCandidates(): Promise<Candidate[]> {
  const { scanLimit, keepCount } = cfg.dataset;
  const rng = createRng(cfg.seed);
  const accepted: Candidate[] = [];
  let rejected = 0;
  const t0 = performance.now();

  for (let cursor = 0; cursor < scanLimit; ) {
    const batchSize = Math.min(HF_PAGE_SIZE, scanLimit - cursor);
    log(`fetching HF rows ${cursor}..${cursor + batchSize - 1}`);

    const page = await fetchPage(cursor, batchSize);
    log(`  received ${page.rows.length} rows (dataset has ${page.num_rows_total} total)`);

    for (const entry of page.rows) {
      const result = toSnippet(entry.row.text);
      if (result) {
        accepted.push({
          snippet: result.snippet,
          fullHtml: result.full,
          rowIndex: entry.row_idx,
        });
      } else {
        rejected++;
      }
    }

    cursor += batchSize;
    if (page.rows.length < batchSize) break; // exhausted dataset
  }

  log(`filtering done in ${elapsed(t0)} — ${accepted.length} accepted, ${rejected} rejected`);

  const selected =
    accepted.length > keepCount
      ? shuffle(accepted, rng).slice(0, keepCount)
      : shuffle(accepted, rng);

  log(`selected ${selected.length} candidates for rendering`);
  return selected;
}

// ── Screenshot capture via Cloudflare Browser Rendering ─────────────────────

function wrapFragment(fragment: string): string {
  return [
    "<!DOCTYPE html>",
    "<html><head><meta charset='utf-8'>",
    "<style>body{margin:20px;background:#fff}</style>",
    "</head><body>",
    fragment,
    "</body></html>",
  ].join("\n");
}

async function captureScreenshot(
  cf: Cloudflare,
  accountId: string,
  html: string,
): Promise<Buffer | null> {
  const payload = looksLikeFullDocument(html) ? html : wrapFragment(html);

  try {
    const raw = await cf.browserRendering.screenshot
      .create({
        account_id: accountId,
        html: payload,
        viewport: cfg.viewport,
        screenshotOptions: { fullPage: false },
      })
      .asResponse();

    return Buffer.from(await raw.arrayBuffer());
  } catch (e) {
    return null;
  }
}

// ── Blank-frame detection ───────────────────────────────────────────────────

async function isNonBlank(png: Buffer): Promise<boolean> {
  const { data, info } = await sharp(png)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  let colored = 0;
  for (let off = 0; off < data.length; off += 3) {
    if (data[off] !== 0xff || data[off + 1] !== 0xff || data[off + 2] !== 0xff) {
      colored++;
    }
  }
  return colored / (info.width * info.height) > cfg.blankThreshold;
}

// ── Manifest types ──────────────────────────────────────────────────────────

interface ManifestRecord {
  id: number;
  screenshot: string;
  html: string;
  reference_html: string;
}

// ── Parallel screenshot pipeline ────────────────────────────────────────────

interface RenderResult {
  candidate: Candidate;
  png: Buffer;
}

/**
 * Process candidates through the screenshot pipeline with bounded concurrency.
 * Returns only the candidates that rendered successfully and are non-blank,
 * preserving their original ordering.
 */
async function renderAll(
  candidates: Candidate[],
  cf: Cloudflare,
  accountId: string,
): Promise<RenderResult[]> {
  const concurrency = cfg.screenshotConcurrency;
  const results: (RenderResult | null)[] = new Array(candidates.length).fill(null);

  let completed = 0;
  let failed = 0;
  let blank = 0;

  async function processOne(index: number) {
    const c = candidates[index];
    const png = await captureScreenshot(cf, accountId, c.fullHtml);

    if (!png?.length) {
      failed++;
      log(`  [${completed + failed + blank}/${candidates.length}] row ${c.rowIndex} — render failed`);
      return;
    }

    if (!(await isNonBlank(png))) {
      blank++;
      log(`  [${completed + failed + blank}/${candidates.length}] row ${c.rowIndex} — blank, skipped`);
      return;
    }

    completed++;
    results[index] = { candidate: c, png };

    if (completed % 25 === 0) {
      log(`  progress: ${completed} captured, ${failed} failed, ${blank} blank (${completed + failed + blank}/${candidates.length} processed)`);
    }
  }

  // Bounded-concurrency worker pool
  let nextIndex = 0;

  async function worker() {
    while (nextIndex < candidates.length) {
      const idx = nextIndex++;
      await processOne(idx);
    }
  }

  const workers = Array.from({ length: Math.min(concurrency, candidates.length) }, () =>
    worker(),
  );
  await Promise.all(workers);

  log(`rendering complete: ${completed} good, ${failed} failed, ${blank} blank out of ${candidates.length}`);

  // Filter nulls, preserving original order
  return results.filter((r): r is RenderResult => r !== null);
}

async function buildManifest(
  candidates: Candidate[],
  cf: Cloudflare,
  accountId: string,
): Promise<ManifestRecord[]> {
  fs.mkdirSync(cfg.paths.screenshots, { recursive: true });

  const rendered = await renderAll(candidates, cf, accountId);
  const records: ManifestRecord[] = [];

  for (const { candidate, png } of rendered) {
    const seq = records.length;
    const file = path.join(cfg.paths.screenshots, `${String(seq).padStart(4, "0")}.png`);
    fs.writeFileSync(file, png);

    records.push({
      id: seq,
      screenshot: file,
      html: candidate.snippet,
      reference_html: candidate.fullHtml,
    });
  }

  return records;
}

// ── Stats ───────────────────────────────────────────────────────────────────

function printStats(records: ManifestRecord[]) {
  if (!records.length) return;
  const lens = records.map((r) => r.html.length).sort((a, b) => a - b);
  const sum = lens.reduce((a, b) => a + b, 0);
  log("snippet length distribution:");
  log(`  min=${lens[0]}  max=${lens.at(-1)}  mean=${Math.round(sum / lens.length)}  median=${lens[lens.length >> 1]}`);
}

// ── Entry point ─────────────────────────────────────────────────────────────

async function run() {
  const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
  const apiToken = process.env.CLOUDFLARE_API_TOKEN;

  if (!accountId || !apiToken) {
    console.error("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env");
    process.exit(1);
  }

  fs.mkdirSync(cfg.paths.output, { recursive: true });

  const t0 = performance.now();
  log("starting pipeline");

  log("phase 1/3: collecting candidates from WebSight v0.2");
  const candidates = await collectCandidates();

  log(`phase 2/3: rendering ${candidates.length} screenshots (concurrency=${cfg.screenshotConcurrency})`);
  const cf = new Cloudflare({ apiToken });
  const manifest = await buildManifest(candidates, cf, accountId);

  log("phase 3/3: writing manifest");
  const dest = path.join(cfg.paths.output, "manifest.json");
  fs.writeFileSync(dest, JSON.stringify(manifest, null, 2));

  log(`done in ${elapsed(t0)} — ${manifest.length} pairs written to ${dest}`);
  printStats(manifest);
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
