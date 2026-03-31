# AGENTS.md

## Project Overview

Slate is a **Design2Code** pipeline: given a screenshot of a webpage, produce the HTML that renders it. The project has two phases:

1. **Data preparation** (implemented) -- Curate a dataset of HTML-snippet / screenshot pairs from HuggingFace's WebSight v0.2 dataset.
2. **Model training** (planned) -- Fine-tune a vision-language model on this dataset using reinforcement learning (GRPO/PPO) via [Tinker](https://tinker.com).

## Repository Structure

```
slate/
  src/
    config.ts          # Centralized tunable parameters (viewport, limits, thresholds)
    prepare-data.ts    # 3-phase data pipeline: collect -> render -> manifest
  rl/
    __init__.py
    config.py          # Python-side config (viewport, reward weights, thresholds)
    reward.py          # Reward function: render HTML, compute visual similarity score
  data/                # Generated output (gitignored)
    manifest.json      # Array of {id, screenshot, html, reference_html} records
    screenshots/       # 1280x720 PNGs (0000.png, 0001.png, ...)
  requirements.txt     # Python dependencies (playwright, Pillow, numpy, scikit-image)
  .env                 # Cloudflare + Tinker credentials (gitignored, never commit)
  .env.example         # Template for required env vars
  RESOURCES.md         # Reference papers and datasets
```

## Tech Stack

- **Data pipeline**: Node.js with TypeScript (ESM modules, run via `tsx`)
- **Model training & inference**: Python (Tinker SDK, follows Tinker API docs)
- **Image rendering**: Cloudflare Browser Rendering API (data pipeline), Playwright (reward evaluation)
- **Image analysis**: `sharp` (blank-frame detection in data pipeline), `scikit-image` SSIM (reward)
- **Dataset source**: HuggingFace WebSight v0.2 (REST API)
- **Training** (planned): Tinker with GRPO/PPO

## How the Data Pipeline Works

`npm run prepare-data` runs `src/prepare-data.ts`, which executes three phases:

1. **Collect candidates** -- Fetch up to `scanLimit` (2000) rows from the WebSight dataset API. Extract compact HTML snippets by stripping document wrappers and merging `<style>` blocks. Reject anything over `maxSnippetLength` (4096 chars). Shuffle deterministically (seed 42) and select up to `keepCount` (500).
2. **Render screenshots** -- Send each snippet to Cloudflare Browser Rendering at 1280x720. Run 10 concurrent workers. Filter out blank frames (less than 2% non-white pixels via `sharp`).
3. **Write manifest** -- Save passing pairs to `data/manifest.json` and screenshots to `data/screenshots/`.

The pipeline currently produces ~490 usable pairs out of a target 500.

## Key Configuration (`src/config.ts`)

| Parameter | Value | Purpose |
|---|---|---|
| `viewport` | 1280x720 | Browser viewport for screenshots |
| `dataset.scanLimit` | 2000 | Max rows to scan from HuggingFace |
| `dataset.keepCount` | 500 | Target number of final pairs |
| `maxSnippetLength` | 4096 | Char budget for HTML snippets (~1024 tokens) |
| `blankThreshold` | 0.02 | Min fraction of non-white pixels to accept a render |
| `screenshotConcurrency` | 10 | Max parallel Cloudflare requests |
| `seed` | 42 | PRNG seed for reproducible shuffling |

## Data Format

Each record in `data/manifest.json`:

```json
{
  "id": 0,
  "screenshot": "/absolute/path/to/data/screenshots/0000.png",
  "html": "<style>...</style><div class=\"...\">compact snippet</div>",
  "reference_html": "<!DOCTYPE html><html>...full original document...</html>"
}
```

- `html` is a body-only snippet with merged styles -- this is the target output for the model.
- `reference_html` is the original full document from WebSight.
- The HTML is predominantly **Tailwind CSS** (v2.2.19 via CDN).

## Environment Variables

See `.env.example`. Required:

- `CLOUDFLARE_ACCOUNT_ID` -- For Browser Rendering API
- `CLOUDFLARE_API_TOKEN` -- For Browser Rendering API
- `TINKER_API_KEY` -- For model training

Never commit `.env`. It is gitignored.

## Commands

| Command | Description |
|---|---|
| `npm run prepare-data` | Run the full data generation pipeline |

## Conventions

- **TypeScript** for data pipeline code. Use strict mode, ESM modules (`import`/`export`, not `require`). Run directly with `tsx` -- no compilation step to `dist/`.
- **Python** for anything that interacts with the Tinker SDK (model training, inference, evaluation). Follow Tinker API docs conventions.
- All configurable values live in `src/config.ts`, not scattered as magic numbers.
- Data output goes in `data/` which is gitignored. Never commit generated data.
- Deterministic reproducibility: use seeded PRNG for any randomness.

## Reward Function (`rl/reward.py`)

The reward function scores how well generated HTML visually matches a reference screenshot. It is the foundation for RL training.

**Signature**: Takes a reference screenshot (via its HTML) and generated HTML, renders both via Playwright at 1280x720, and returns a score in [-1, +1].

**Three sub-signals**:

| Signal | Weight | Source | Range |
|---|---|---|---|
| SSIM (visual) | 0.60 | `scikit-image` structural similarity on 256x256 downscaled screenshots | [-1, 1] |
| Text similarity | 0.25 | `SequenceMatcher` on `document.body.innerText` | [0, 1] |
| Color palette | 0.15 | Histogram intersection over quantized (step=32) RGB colors | [0, 1] |

**Content gate** prevents blank-page reward hacking:

```
content_gate = 0.2 + 0.8 * max(text_score, color_score)
gated_ssim   = ssim * content_gate
raw          = 0.60 * gated_ssim + 0.25 * text + 0.15 * color
reward       = 2 * raw - 1
```

A blank page with no text or colour gets `gate=0.2`, crushing SSIM credit even if the background matches. The gradient is continuous everywhere -- no hard -1 cliff.

**Key functions**:

| Function | Purpose |
|---|---|
| `extract_html_from_response(text)` | Parse model output to extract HTML (fenced blocks, raw tags) |
| `extract_ref_info(page, html)` | Render reference HTML, extract text + colours + 256x256 image |
| `extract_gen_info(page, html)` | Render generated HTML, extract same info + full-res image |
| `compute_reward_from_info(ref, gen)` | Compute reward from pre-extracted info, returns `(score, details)` |
| `make_diff_image(ref_img, gen_img)` | Red-overlay diff image for debugging |

**Browser lifecycle**: Callers manage the Playwright browser/page. The reward functions accept a `Page` object. This keeps the module flexible for both batch scoring and RL rollout loops.

## Design Decisions and Constraints

- HTML snippets are capped at 4096 characters to keep training sequences manageable (~1024 tokens).
- Blank-frame detection prevents degenerate training examples (all-white screenshots).
- The pipeline is intentionally decoupled from training so the dataset can be regenerated independently.
- Screenshots use a fixed 1280x720 viewport for consistency across the entire dataset.

## Next Steps / Training Intent

The dataset is designed for a **Design2Code** task: a vision-language model receives a screenshot and must generate the HTML that produced it. Training will use reinforcement learning (GRPO or PPO) via Tinker. See `RESOURCES.md` for reference papers:

- GRPO/PPO for LLM fine-tuning
- WebSight dataset
- Design2Code benchmark
