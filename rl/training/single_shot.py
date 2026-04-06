"""
Single-shot GRPO training for Design2Code.

For each training batch:
  1. Pick BATCH_SIZE screenshots from the dataset
  2. Sample GROUP_SIZE HTML completions per screenshot from the current policy
  3. Score each with the reward function (Playwright + SSIM/text/color)
  4. Compute GRPO advantages: reward_i - mean(rewards) within each group
  5. Build training datums and run a PPO-style gradient step via Tinker

The model sees one screenshot and produces one HTML output -- hence "single shot."
Multi-turn (generate -> diff -> fix) comes in a later phase.

Usage:
    source venv/bin/activate
    python -m training.single_shot                     # defaults
    python -m training.single_shot --batches 20        # 20 training batches
    python -m training.single_shot --resume             # resume from last checkpoint
    python -m training.single_shot --log-path data/training/run2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import dotenv
import numpy as np
import tinker
import torch
from PIL import Image
from playwright.sync_api import sync_playwright
from tinker import types
from tinker.types.tensor_data import TensorData

dotenv.load_dotenv()

from tinker_cookbook import checkpoint_utils
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
    BATCH_SIZE,
    GROUP_SIZE,
    LEARNING_RATE,
    LOG_PATH,
    LORA_RANK,
    MAX_IMAGE_SIZE,
    MAX_TOKENS,
    MODEL_NAME,
    RENDERER_NAME,
    SAVE_EVERY,
    SYSTEM_PROMPT,
    TRAIN_TEMPERATURE,
    VIEWPORT_H,
    VIEWPORT_W,
)
from rl.reward import (
    compute_reward_from_info,
    extract_gen_info,
    extract_html_from_response,
    extract_ref_info,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = ROOT / "data" / "manifest.json"


# ── Dataset ──────────────────────────────────────────────────────────────────


def load_training_data(seed: int = 42) -> list[dict]:
    """Load and shuffle the full manifest for training."""
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    rng = np.random.default_rng(seed)
    rng.shuffle(manifest)
    return manifest


# ── Prompt construction ──────────────────────────────────────────────────────


def build_prompt(renderer, screenshot_path: str) -> tinker.ModelInput:
    """Build a VLM prompt: system instruction + screenshot + task."""
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


# ── Reward scoring ───────────────────────────────────────────────────────────


def score_html(page, reference_html: str, generated_html: str | None) -> float:
    """Score a single generated HTML against reference. Returns reward in [-1, +1]."""
    if generated_html is None:
        return -1.0

    try:
        ref_info = extract_ref_info(page, reference_html)
        gen_info = extract_gen_info(page, generated_html)
        reward, _details = compute_reward_from_info(ref_info, gen_info)
        return reward
    except Exception as e:
        log.warning("Reward scoring failed: %s", e)
        return -1.0


# ── Main training loop ───────────────────────────────────────────────────────


def train(
    n_batches: int = 20,
    resume: bool = False,
    log_path: str = LOG_PATH,
):
    """
    Run single-shot GRPO training.

    Each batch:
      - Picks BATCH_SIZE examples
      - Samples GROUP_SIZE completions per example from the current policy
      - Scores each with the reward function
      - Computes GRPO advantages (reward centering within group)
      - Runs importance_sampling loss + optimizer step
    """
    log_path_obj = Path(log_path)
    log_path_obj.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ─────────────────────────────────────────────────────
    dataset = load_training_data()
    n_total = len(dataset)
    log.info("Loaded %d training examples", n_total)

    total_possible_batches = n_total // BATCH_SIZE
    n_batches = min(n_batches, total_possible_batches)
    log.info("Will train for %d batches (batch_size=%d, group_size=%d)",
             n_batches, BATCH_SIZE, GROUP_SIZE)

    # ── Set up renderer ──────────────────────────────────────────────────
    log.info("Loading tokenizer + image processor for %s", MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    image_processor = get_image_processor(MODEL_NAME)
    renderer = get_renderer(
        name=RENDERER_NAME,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    stop_sequences = renderer.get_stop_sequences()

    sampling_params = types.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TRAIN_TEMPERATURE,
        stop=stop_sequences,
    )

    adam_params = types.AdamParams(
        learning_rate=LEARNING_RATE,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    # ── Set up Tinker client ─────────────────────────────────────────────
    service_client = tinker.ServiceClient()

    start_batch = 0
    if resume:
        resume_info = checkpoint_utils.get_last_checkpoint(str(log_path_obj))
        if resume_info:
            log.info("Resuming from checkpoint at batch %d", resume_info.batch)
            training_client = (
                service_client.create_training_client_from_state_with_optimizer(
                    resume_info.state_path
                )
            )
            start_batch = resume_info.batch
        else:
            log.info("No checkpoint found, starting fresh")
            training_client = service_client.create_lora_training_client(
                base_model=MODEL_NAME, rank=LORA_RANK,
            )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=MODEL_NAME, rank=LORA_RANK,
        )

    log.info("Training client ready (LoRA rank=%d)", LORA_RANK)

    # ── Launch Playwright for reward scoring ─────────────────────────────
    pw_context = sync_playwright().start()
    browser = pw_context.chromium.launch()
    page = browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})

    # ── Metrics tracking ─────────────────────────────────────────────────
    all_metrics: list[dict] = []

    try:
        for batch_idx in range(start_batch, start_batch + n_batches):
            t_start = time.time()

            # ── Checkpoint ───────────────────────────────────────────
            if (
                SAVE_EVERY > 0
                and batch_idx > start_batch
                and (batch_idx - start_batch) % SAVE_EVERY == 0
            ):
                log.info("Saving checkpoint at batch %d...", batch_idx)
                checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"{batch_idx:06d}",
                    log_path=str(log_path_obj),
                    kind="state",
                    loop_state={"batch": batch_idx},
                    ttl_seconds=604800,  # 7 days
                )

            # ── Select batch examples ────────────────────────────────
            batch_start = (batch_idx * BATCH_SIZE) % n_total
            batch_end = batch_start + BATCH_SIZE
            if batch_end > n_total:
                # Wrap around -- take what's left + from the start
                batch_examples = dataset[batch_start:] + dataset[:batch_end - n_total]
            else:
                batch_examples = dataset[batch_start:batch_end]

            log.info(
                "Batch %d: examples %d-%d",
                batch_idx, batch_start, batch_start + len(batch_examples) - 1,
            )

            # ── Get current policy's sampling client ─────────────────
            sampling_client = training_client.save_weights_and_get_sampling_client()

            # ── Build prompts + fire off sampling requests ───────────
            prompts = []
            sample_futures = []
            for ex in batch_examples:
                prompt = build_prompt(renderer, ex["screenshot"])
                prompts.append(prompt)

                # Sample GROUP_SIZE completions per prompt in one call
                future = sampling_client.sample(
                    prompt=prompt,
                    num_samples=GROUP_SIZE,
                    sampling_params=sampling_params,
                )
                sample_futures.append(future)

            # ── Collect samples + compute rewards ────────────────────
            t_sample = time.time()
            datums: list[types.Datum] = []
            batch_rewards: list[float] = []
            batch_skipped = 0

            for ex, prompt, future in zip(batch_examples, prompts, sample_futures):
                sample_result = future.result()

                # Decode each completion and compute reward
                rewards_G: list[float] = []
                tokens_G: list[list[int]] = []
                logprobs_G: list[list[float]] = []

                for sequence in sample_result.sequences:
                    tokens_G.append(sequence.tokens)
                    assert sequence.logprobs is not None
                    logprobs_G.append(sequence.logprobs)

                    # Decode to text, extract HTML
                    parsed_msg, _ok = renderer.parse_response(sequence.tokens)
                    raw_text = get_text_content(parsed_msg)
                    generated_html = extract_html_from_response(raw_text)

                    # Score with reward function
                    reward = score_html(page, ex["reference_html"], generated_html)
                    rewards_G.append(reward)

                sample_time = time.time() - t_sample

                # ── GRPO advantage computation ───────────────────────
                mean_reward = sum(rewards_G) / len(rewards_G)
                advantages_G = [r - mean_reward for r in rewards_G]
                batch_rewards.append(mean_reward)

                # Skip if all advantages are zero (all completions got same reward)
                if all(a == 0.0 for a in advantages_G):
                    batch_skipped += 1
                    continue

                # ── Build datums ─────────────────────────────────────
                for sampled_tokens, logprobs, advantage in zip(
                    tokens_G, logprobs_G, advantages_G
                ):
                    ob_len = prompt.length - 1
                    model_input = prompt.append(
                        types.EncodedTextChunk(tokens=sampled_tokens[:-1])
                    )

                    target_tokens = [0] * ob_len + sampled_tokens
                    padded_logprobs = [0.0] * ob_len + logprobs
                    padded_advantages = (
                        [0.0] * ob_len
                        + [advantage] * (model_input.length - ob_len)
                    )

                    assert (
                        model_input.length
                        == len(target_tokens)
                        == len(padded_logprobs)
                        == len(padded_advantages)
                    ), (
                        f"Length mismatch: input={model_input.length}, "
                        f"target={len(target_tokens)}, "
                        f"logprobs={len(padded_logprobs)}, "
                        f"advantages={len(padded_advantages)}"
                    )

                    datum = types.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(
                                torch.tensor(target_tokens)
                            ),
                            "logprobs": TensorData.from_torch(
                                torch.tensor(padded_logprobs)
                            ),
                            "advantages": TensorData.from_torch(
                                torch.tensor(padded_advantages)
                            ),
                        },
                    )
                    datums.append(datum)

            # ── Training step ────────────────────────────────────────
            if len(datums) == 0:
                log.warning(
                    "Batch %d: all advantages zero, skipping training step",
                    batch_idx,
                )
            else:
                log.info(
                    "Batch %d: training on %d datums (%d prompts skipped)...",
                    batch_idx, len(datums), batch_skipped,
                )

                # Pipeline: enqueue forward_backward + optim_step together
                fwd_bwd_future = training_client.forward_backward(
                    datums, loss_fn="importance_sampling"
                )
                optim_future = training_client.optim_step(adam_params)

                _fwd_bwd_result = fwd_bwd_future.result()
                optim_result = optim_future.result()

            # ── Log metrics ──────────────────────────────────────────
            elapsed = time.time() - t_start
            mean_batch_reward = (
                sum(batch_rewards) / len(batch_rewards)
                if batch_rewards
                else 0.0
            )

            metrics = {
                "batch": batch_idx,
                "reward_mean": mean_batch_reward,
                "reward_min": min(batch_rewards) if batch_rewards else 0.0,
                "reward_max": max(batch_rewards) if batch_rewards else 0.0,
                "n_datums": len(datums),
                "n_skipped": batch_skipped,
                "time_s": elapsed,
                "lr": LEARNING_RATE,
            }
            all_metrics.append(metrics)

            log.info(
                "Batch %d complete: reward=%.3f  datums=%d  skipped=%d  time=%.1fs",
                batch_idx,
                mean_batch_reward,
                len(datums),
                batch_skipped,
                elapsed,
            )

        # ── Final checkpoint ─────────────────────────────────────────
        final_batch = start_batch + n_batches
        log.info("Saving final checkpoint...")
        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=str(log_path_obj),
            kind="both",  # Save both state (for resume) and sampler (for inference)
            loop_state={"batch": final_batch},
            ttl_seconds=None,  # Keep forever
        )

    finally:
        browser.close()
        pw_context.stop()

    # ── Save metrics ─────────────────────────────────────────────────────
    metrics_path = log_path_obj / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────────
    all_rewards = [m["reward_mean"] for m in all_metrics]
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:       {MODEL_NAME}")
    print(f"  Batches:     {len(all_metrics)}")
    print(f"  Group size:  {GROUP_SIZE}")
    print(f"  LoRA rank:   {LORA_RANK}")
    print(f"  LR:          {LEARNING_RATE}")
    print()
    if all_rewards:
        # Show reward trend: first 3 vs last 3
        early = all_rewards[:3]
        late = all_rewards[-3:]
        print(f"  Reward (early avg):  {sum(early)/len(early):+.3f}")
        print(f"  Reward (late avg):   {sum(late)/len(late):+.3f}")
        print(f"  Reward (overall):    {sum(all_rewards)/len(all_rewards):+.3f}")
    print()
    print(f"  Checkpoints: {log_path_obj}")
    print(f"  Metrics:     {metrics_path}")
    print("=" * 60 + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Single-shot GRPO training for Design2Code"
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=20,
        help="Number of training batches (default: 20)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint in --log-path",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=LOG_PATH,
        help=f"Directory for checkpoints and logs (default: {LOG_PATH})",
    )
    args = parser.parse_args()
    train(
        n_batches=args.batches,
        resume=args.resume,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
