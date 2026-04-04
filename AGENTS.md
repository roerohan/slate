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
    config.py          # Python-side config (viewport, reward weights, model, eval settings)
    reward.py          # Reward function: render HTML, compute visual similarity score
    eval.py            # Baseline evaluation: VLM inference + reward scoring + artifacts
    training/
      __init__.py
      single_shot.py   # Single-shot GRPO training: screenshot -> HTML, one attempt
      multi_turn.py    # Multi-turn GRPO training: generate -> diff feedback -> correct
  data/                # Generated output (gitignored)
    manifest.json      # Array of {id, screenshot, html, reference_html} records
    screenshots/       # 1280x720 PNGs (0000.png, 0001.png, ...)
    eval/              # Eval artifacts (generated screenshots, diffs, results.json)
    training/          # Training logs, metrics, and checkpoints
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

Before running any Python commands, activate the virtual environment if one is present:

```bash
source venv/bin/activate
```

| Command | Description |
|---|---|
| `npm run prepare-data` | Run the full data generation pipeline |
| `python -m rl.eval` | Evaluate base VLM on Design2Code (50 examples by default) |
| `python -m rl.eval --n 20` | Evaluate on 20 examples |
| `python -m rl.eval --model-path tinker://...` | Evaluate a fine-tuned model checkpoint |
| `python -m rl.eval --out data/eval_run2` | Custom output directory |
| `python -m rl.training.single_shot` | Run single-shot GRPO training (20 batches default) |
| `python -m rl.training.single_shot --batches 10` | Train for 10 batches |
| `python -m rl.training.single_shot --resume` | Resume from last checkpoint |
| `python -m rl.training.single_shot --log-path data/training/run2` | Custom log directory |
| `python -m rl.training.multi_turn` | Run multi-turn GRPO training (20 batches, 3 turns) |
| `python -m rl.training.multi_turn --turns 2` | Limit to 2 turns per episode |
| `python -m rl.training.multi_turn --resume` | Resume from last checkpoint |

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

## Evaluation (`rl/eval.py`)

Measures VLM performance on the Design2Code task. Validates the full pipeline end-to-end: Tinker inference, HTML extraction, reward scoring, artifact generation.

**Pipeline**:

1. **Load dataset** -- Selects N examples from `data/manifest.json` (shuffled with seed 42).
2. **Build prompts** -- Each prompt contains a system instruction + the reference screenshot image, sent to the VLM via Tinker.
3. **Sample HTML** -- Async concurrent requests to Tinker (default 8 parallel). The VLM generates HTML in a ```html code block.
4. **Score with reward function** -- Renders both reference and generated HTML via Playwright, computes reward (SSIM + text + color).
5. **Save artifacts** -- Per-example: generated screenshot (`NNNN_gen.png`), diff image (`NNNN_diff.png`), generated HTML (`NNNN_gen.html`). Aggregate: `results.json` with metrics + per-example breakdown.

**Model**: `Qwen/Qwen3.5-4B` with `qwen3_5_disable_thinking` renderer (configurable in `rl/config.py`).

**Key config** (all in `rl/config.py`):

| Parameter | Value | Purpose |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3.5-4B` | Base VLM for inference |
| `RENDERER_NAME` | `qwen3_5_disable_thinking` | Tinker renderer (no thinking tokens) |
| `MAX_TOKENS` | 8192 | Max generated tokens |
| `TEMPERATURE` | 0.7 | Sampling temperature |
| `EVAL_SUBSET` | 50 | Default number of eval examples |
| `EVAL_CONCURRENCY` | 8 | Max parallel Tinker requests |
| `MAX_IMAGE_SIZE` | 480 | Longest side of screenshot sent to VLM |

**Expected output** (base model): The reference project saw avg SSIM ~0.536, reward ~-0.677 for a base 4B model. Poor performance is expected and is exactly what RL training is meant to fix.

## Single-Shot GRPO Training (`rl/training/single_shot.py`)

Trains the VLM to improve at screenshot-to-HTML using Group Relative Policy Optimization (GRPO). "Single-shot" means one screenshot in, one HTML out -- no multi-turn correction loop.

**Per-batch loop**:

1. **Select examples** -- Pick `BATCH_SIZE` (4) screenshots from the shuffled dataset.
2. **Sample completions** -- For each screenshot, sample `GROUP_SIZE` (4) HTML completions from the current policy via Tinker.
3. **Score** -- Render each completion in Playwright, compute reward using the reward function.
4. **GRPO advantages** -- For each group, `advantage_i = reward_i - mean(rewards)`. This is the RL signal: completions better than the group average get positive advantage, worse ones get negative.
5. **Build datums** -- Construct Tinker training datums with prompt tokens, response tokens, sampling logprobs, and per-token advantages. Groups where all rewards are identical are skipped (zero gradient).
6. **Gradient step** -- `importance_sampling` loss (GRPO-style) + AdamW optimizer via Tinker's LoRA training API.
7. **Checkpoint** -- Save weights every `SAVE_EVERY` (5) batches. Final checkpoint saves both state (for resume) and sampler weights (for inference/eval).

**Key config** (all in `rl/config.py`):

| Parameter | Value | Purpose |
|---|---|---|
| `LEARNING_RATE` | 4e-5 | AdamW learning rate |
| `LORA_RANK` | 32 | LoRA adapter rank |
| `GROUP_SIZE` | 4 | Completions per prompt (for GRPO variance reduction) |
| `BATCH_SIZE` | 4 | Prompts per training batch |
| `TRAIN_TEMPERATURE` | 1.0 | Sampling temperature (higher than eval for exploration) |
| `SAVE_EVERY` | 5 | Checkpoint every N batches |

**Resuming**: Use `--resume` to continue from the last checkpoint in the log directory. Loads both weights and optimizer state.

**After training**: Run `python -m rl.eval --model-path tinker://...` with the checkpoint path from the final save to measure improvement.

## Multi-Turn GRPO Training (`rl/training/multi_turn.py`)

Extends single-shot training with a self-correction loop. Instead of one attempt per screenshot, the model gets multiple turns:

1. **Turn 1**: See screenshot, generate HTML
2. **Turn 2**: See diff image (red overlay showing pixel differences), fix HTML
3. **Turn 3**: See updated diff, fix again

**Episode flow** (per group member):

1. Build initial prompt (system + screenshot + task)
2. Sample HTML from the model
3. Render the HTML, score it, produce a diff image vs the reference
4. If not the last turn: append the model's response + diff image + feedback prompt to the conversation
5. Sample corrected HTML from the model (it sees the full conversation history)
6. Repeat until MAX_TURNS or scoring fails

**GRPO advantage**: Computed from the **final turn's reward** only. All turns in the episode share the same advantage. This incentivises the model to produce the best possible output by the end, regardless of how many correction rounds it took.

**Datum structure**: Each turn produces a **separate Datum** because the `qwen3_5_disable_thinking` renderer strips thinking tokens from conversation history, breaking the prefix-extension property. Turn 2's tokenized prompt is re-tokenized from scratch, not appended to turn 1's tokens.

**Key config** (in `rl/config.py`):

| Parameter | Value | Purpose |
|---|---|---|
| `MAX_TURNS` | 3 | Max conversation turns per episode |
| `FEEDBACK_PROMPT` | "Your HTML does not match..." | User message sent with each diff image |

**vs single-shot**: Multi-turn produces more datums per example (one per turn per group member) and takes longer per batch due to sequential turns, but teaches the model to self-correct based on visual feedback.

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
