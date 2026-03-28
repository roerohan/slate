import path from "path";
import { fileURLToPath } from "url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const config = {
  paths: {
    output: path.join(root, "data"),
    screenshots: path.join(root, "data", "screenshots"),
  },

  viewport: { width: 1280, height: 720 },

  dataset: {
    /** How many rows to scan from the HF dataset. */
    scanLimit: 2000,
    /** How many usable pairs we want in the final output. */
    keepCount: 500,
  },

  /** Char budget for HTML snippets (~1024 tokens). */
  maxSnippetLength: 4096,

  /** Fraction of non-white pixels required to consider a render non-blank. */
  blankThreshold: 0.02,

  /** Max concurrent Cloudflare screenshot requests. */
  screenshotConcurrency: 10,

  seed: 42,
} as const;

export default config;
