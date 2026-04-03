"""
Evaluate base Qwen 3.5 4B on the Design2Code task.

Sends each screenshot from the dataset to the VLM, collects generated HTML,
scores it against the reference using the reward function, and saves
per-example artifacts + aggregate metrics.

Usage:
    source venv/bin/activate
    python -m rl.eval                       # defaults: 50 examples
    python -m rl.eval --n 20                # evaluate 20 examples
    python -m rl.eval --model-path tinker://run-id/weights/checkpoint-001
    python -m rl.eval --out data/eval_run2  # custom output directory
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import dotenv
import numpy as np
import tinker

# Load .env so TINKER_API_KEY (and any other secrets) are available.
dotenv.load_dotenv()
from PIL import Image
from playwright.sync_api import sync_playwright
from tinker import types

from tinker_cookbook.image_processing_utils import get_image_processor, resize_image
from tinker_cookbook.renderers import (
    ImagePart,
    Message,
    TextPart,
    get_renderer,
    get_text_content,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from rl.config import (
    EVAL_CONCURRENCY,
    EVAL_SUBSET,
    MAX_IMAGE_SIZE,
    MAX_TOKENS,
    MODEL_NAME,
    RENDERER_NAME,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    VIEWPORT_H,
    VIEWPORT_W,
)
from rl.reward import (
    compute_reward_from_info,
    extract_gen_info,
    extract_html_from_response,
    extract_ref_info,
    make_diff_image,
    render_html_to_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "manifest.json"
DEFAULT_EVAL_DIR = ROOT / "data" / "eval"


# ── Dataset loading ──────────────────────────────────────────────────────────


def load_eval_subset(n: int, seed: int = 42) -> list[dict]:
    """Load *n* examples from the manifest, shuffled deterministically."""
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(manifest))[:n]
    return [manifest[i] for i in indices]


# ── Prompt construction ──────────────────────────────────────────────────────


def build_prompt(
    renderer,
    screenshot_path: str,
) -> tinker.ModelInput:
    """Build a VLM prompt: system instruction + screenshot image + task."""
    pil_image = Image.open(screenshot_path).convert("RGB")
    pil_image = resize_image(pil_image, max_size=MAX_IMAGE_SIZE)

    messages: list[Message] = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(
            role="user",
            content=[
                ImagePart(type="image", image=pil_image),
                TextPart(
                    type="text",
                    text="Generate the HTML that produces this webpage screenshot.",
                ),
            ],
        ),
    ]
    return renderer.build_generation_prompt(messages)


# ── Tinker inference (async) ─────────────────────────────────────────────────


async def sample_html(
    sampling_client: tinker.SamplingClient,
    renderer,
    model_input: tinker.ModelInput,
    sampling_params: types.SamplingParams,
) -> str | None:
    """Send a prompt to Tinker and return the decoded text response."""
    response = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )
    tokens = response.sequences[0].tokens
    parsed_message, _ok = renderer.parse_response(tokens)
    raw_text = get_text_content(parsed_message)
    return extract_html_from_response(raw_text)


# ── Reward scoring (sync, needs Playwright) ──────────────────────────────────


def score_example(
    page,
    reference_html: str,
    generated_html: str | None,
    out_dir: Path,
    example_id: int,
) -> dict:
    """
    Score one example: render reference + generated HTML, compute reward,
    save artifacts (generated screenshot, diff image, generated HTML).
    """
    gen_screenshot_path = out_dir / f"{example_id:04d}_gen.png"
    diff_path = out_dir / f"{example_id:04d}_diff.png"
    html_path = out_dir / f"{example_id:04d}_gen.html"

    # Save generated HTML
    if generated_html:
        html_path.write_text(generated_html, encoding="utf-8")

    # Handle failed extraction
    if generated_html is None:
        render_html_to_file(page, None, str(gen_screenshot_path))
        return {
            "id": example_id,
            "reward": -1.0,
            "ssim": 0.0,
            "text": 0.0,
            "color": 0.0,
            "content_gate": 0.2,
            "extraction_failed": True,
        }

    # Extract reference info
    ref_info = extract_ref_info(page, reference_html)

    # Render generated HTML and extract info
    gen_info = extract_gen_info(page, generated_html)

    # Save generated screenshot
    gen_full = gen_info.get("image_full")
    if gen_full is not None:
        Image.fromarray(gen_full).save(str(gen_screenshot_path))

    # Compute reward
    reward, details = compute_reward_from_info(ref_info, gen_info)

    # Save diff image
    try:
        diff_img = make_diff_image(ref_info["image"], gen_info["image"])
        Image.fromarray(diff_img).save(str(diff_path))
    except Exception:
        pass  # diff image is optional

    return {
        "id": example_id,
        "reward": reward,
        "ssim": details["ssim"],
        "text": details["text"],
        "color": details["color"],
        "content_gate": details["content_gate"],
        "extraction_failed": False,
    }


# ── Main eval loop ───────────────────────────────────────────────────────────


def run_eval(
    n: int = EVAL_SUBSET,
    model_path: str | None = None,
    out_dir: Path = DEFAULT_EVAL_DIR,
) -> dict:
    """
    Run the full evaluation pipeline:
      1. Connect to Tinker, create sampling client
      2. Build prompts for N examples
      3. Sample HTML from VLM (async, concurrent)
      4. Score each with reward function (sync, Playwright)
      5. Save artifacts + report metrics
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ─────────────────────────────────────────────────────
    examples = load_eval_subset(n)
    log.info("Loaded %d examples for evaluation", len(examples))

    # ── Set up Tinker client ─────────────────────────────────────────────
    log.info("Connecting to Tinker (model=%s)", model_path or MODEL_NAME)
    service_client = tinker.ServiceClient()

    if model_path:
        sampling_client = service_client.create_sampling_client(
            model_path=model_path,
        )
    else:
        sampling_client = service_client.create_sampling_client(
            base_model=MODEL_NAME,
        )

    # ── Set up renderer ──────────────────────────────────────────────────
    log.info("Loading tokenizer + image processor for %s", MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    image_processor = get_image_processor(MODEL_NAME)
    renderer = get_renderer(
        name=RENDERER_NAME,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    sampling_params = types.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        stop=renderer.get_stop_sequences(),
    )

    # ── Phase 1: Build prompts ───────────────────────────────────────────
    log.info("Building prompts...")
    prompts = []
    for ex in examples:
        prompts.append(build_prompt(renderer, ex["screenshot"]))

    # ── Phase 2: Sample from VLM (async) ─────────────────────────────────
    log.info(
        "Sampling HTML from VLM (%d examples, concurrency=%d)...",
        len(examples),
        EVAL_CONCURRENCY,
    )
    t0 = time.time()

    async def _sample_all() -> list[str | None]:
        sem = asyncio.Semaphore(EVAL_CONCURRENCY)

        async def _bounded(prompt):
            async with sem:
                return await sample_html(
                    sampling_client, renderer, prompt, sampling_params
                )

        tasks = [asyncio.create_task(_bounded(p)) for p in prompts]
        return await asyncio.gather(*tasks)

    generated_htmls = asyncio.run(_sample_all())
    sample_time = time.time() - t0
    log.info("Sampling complete in %.1fs", sample_time)

    extraction_ok = sum(1 for h in generated_htmls if h is not None)
    log.info("HTML extraction: %d/%d successful", extraction_ok, len(generated_htmls))

    # ── Phase 3: Score with reward function (Playwright) ─────────────────
    log.info("Scoring with reward function (Playwright)...")
    t0 = time.time()
    results = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(
            viewport={"width": VIEWPORT_W, "height": VIEWPORT_H}
        )

        for i, (ex, gen_html) in enumerate(zip(examples, generated_htmls)):
            result = score_example(
                page=page,
                reference_html=ex["reference_html"],
                generated_html=gen_html,
                out_dir=out_dir,
                example_id=ex["id"],
            )
            results.append(result)

            if (i + 1) % 10 == 0 or i == len(examples) - 1:
                log.info(
                    "  scored %d/%d  (last reward=%.3f)",
                    i + 1,
                    len(examples),
                    result["reward"],
                )

        browser.close()

    score_time = time.time() - t0
    log.info("Scoring complete in %.1fs", score_time)

    # ── Phase 4: Aggregate metrics ───────────────────────────────────────
    rewards = [r["reward"] for r in results]
    ssims = [r["ssim"] for r in results]
    texts = [r["text"] for r in results]
    colors = [r["color"] for r in results]
    failed = sum(1 for r in results if r["extraction_failed"])

    metrics = {
        "n_examples": len(results),
        "n_extraction_failed": failed,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "reward_median": float(np.median(rewards)),
        "ssim_mean": float(np.mean(ssims)),
        "text_mean": float(np.mean(texts)),
        "color_mean": float(np.mean(colors)),
        "sample_time_s": sample_time,
        "score_time_s": score_time,
        "model": model_path or MODEL_NAME,
    }

    # ── Phase 5: Save results ────────────────────────────────────────────
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"metrics": metrics, "per_example": results}, f, indent=2)

    # ── Print report ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVAL RESULTS")
    print("=" * 60)
    print(f"  Model:              {metrics['model']}")
    print(f"  Examples:           {metrics['n_examples']}")
    print(f"  Extraction failed:  {metrics['n_extraction_failed']}")
    print()
    print(f"  Reward:  {metrics['reward_mean']:+.3f} +/- {metrics['reward_std']:.3f}")
    print(f"           min={metrics['reward_min']:+.3f}  max={metrics['reward_max']:+.3f}  median={metrics['reward_median']:+.3f}")
    print(f"  SSIM:    {metrics['ssim_mean']:.3f}")
    print(f"  Text:    {metrics['text_mean']:.3f}")
    print(f"  Color:   {metrics['color_mean']:.3f}")
    print()
    print(f"  Sampling time:  {metrics['sample_time_s']:.1f}s")
    print(f"  Scoring time:   {metrics['score_time_s']:.1f}s")
    print(f"  Artifacts:      {out_dir}")
    print("=" * 60 + "\n")

    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a VLM on the Design2Code task"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=EVAL_SUBSET,
        help=f"Number of examples to evaluate (default: {EVAL_SUBSET})",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Tinker model path (tinker://...) for fine-tuned weights. "
        "Omit to use the base model.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_EVAL_DIR),
        help=f"Output directory for artifacts (default: {DEFAULT_EVAL_DIR})",
    )
    args = parser.parse_args()
    run_eval(n=args.n, model_path=args.model_path, out_dir=Path(args.out))


if __name__ == "__main__":
    main()
