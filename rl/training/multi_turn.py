"""
Multi-turn GRPO training for Design2Code.

Unlike single_shot.py where the model gets one attempt, here the model gets
multiple turns per screenshot:

  Turn 1: See screenshot -> generate HTML
  Turn 2: See diff image showing what's wrong -> fix HTML
  Turn 3: See updated diff -> fix again
  ...up to MAX_TURNS

The model learns to self-correct. GRPO advantages are computed on the final
turn's reward, so the model is incentivised to produce the best possible
output by the end of the conversation.

Each turn produces a separate training datum because the renderer strips
thinking tokens from conversation history, breaking the prefix-extension
property. This means turn 2's prompt is re-tokenized from scratch (not an
append to turn 1's tokens).

Usage:
    source venv/bin/activate
    python -m rl.training.multi_turn                      # defaults
    python -m rl.training.multi_turn --batches 20         # 20 batches
    python -m rl.training.multi_turn --turns 2            # 2 turns per episode
    python -m rl.training.multi_turn --resume
    python -m rl.training.multi_turn --log-path data/training_multi_turn/run2
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
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
    FEEDBACK_PROMPT,
    GROUP_SIZE,
    LEARNING_RATE,
    LORA_RANK,
    MAX_IMAGE_SIZE,
    MAX_TOKENS,
    MAX_TURNS,
    MODEL_NAME,
    MULTI_TURN_LOG_PATH,
    RENDERER_NAME,
    SAVE_EVERY,
    SSIM_SIZE,
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
    make_diff_image,
    render_html,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = ROOT / "data" / "manifest.json"


# ── Checkpoint helper ────────────────────────────────────────────────────────
# Playwright's sync API runs an asyncio event loop in a background thread.
# The sync wrapper checkpoint_utils.save_checkpoint() calls asyncio.run()
# which fails with "cannot be called from a running event loop".
# We call save_checkpoint_async() directly on the existing loop instead.


def _save_checkpoint(**kwargs) -> dict[str, str]:
    """Call save_checkpoint_async on the running event loop."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(checkpoint_utils.save_checkpoint_async(**kwargs))


# ── Dataset ──────────────────────────────────────────────────────────────────


def load_training_data(seed: int = 42) -> list[dict]:
    """Load and shuffle the full manifest for training."""
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    rng = np.random.default_rng(seed)
    rng.shuffle(manifest)
    return manifest


# ── Prompt construction ──────────────────────────────────────────────────────


def build_initial_prompt(renderer, screenshot_path: str) -> tuple[list[Message], tinker.ModelInput]:
    """
    Build the first-turn prompt: system instruction + screenshot + task.
    Returns both the message list (for appending later turns) and the
    tokenized ModelInput.
    """
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
    model_input = renderer.build_generation_prompt(messages)
    return messages, model_input


def build_feedback_prompt(
    renderer,
    messages: list[Message],
    assistant_text: str,
    diff_image: Image.Image,
) -> tuple[list[Message], tinker.ModelInput]:
    """
    Build a correction-turn prompt: append the model's previous response
    and a user message with the diff image + feedback instructions.
    Returns the updated message list and tokenized ModelInput.
    """
    # Append the model's previous response
    messages = messages + [
        Message(role="assistant", content=assistant_text),
        Message(
            role="user",
            content=[
                ImagePart(type="image", image=diff_image),
                TextPart(type="text", text=FEEDBACK_PROMPT),
            ],
        ),
    ]
    model_input = renderer.build_generation_prompt(messages)
    return messages, model_input


# ── Reward scoring ───────────────────────────────────────────────────────────


def score_and_diff(
    page,
    reference_html: str,
    generated_html: str | None,
    ref_info: dict | None = None,
) -> tuple[float, dict, Image.Image | None]:
    """
    Score generated HTML against reference and produce a diff image.

    Returns (reward, details, diff_pil_image).
    diff_pil_image is None if scoring failed.
    ref_info can be passed to avoid re-rendering the reference each turn.
    """
    if generated_html is None:
        return -1.0, {"ssim": 0, "text": 0, "color": 0, "content_gate": 0.2}, None

    try:
        if ref_info is None:
            ref_info = extract_ref_info(page, reference_html)
        gen_info = extract_gen_info(page, generated_html)
        reward, details = compute_reward_from_info(ref_info, gen_info)

        # Create diff image for feedback
        diff_arr = make_diff_image(ref_info["image"], gen_info["image"])
        # Upscale diff from SSIM_SIZE to something the VLM can see clearly
        diff_pil = Image.fromarray(diff_arr).resize(
            (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS
        )

        return reward, details, diff_pil
    except Exception as e:
        log.warning("Reward scoring failed: %s", e)
        return -1.0, {"ssim": 0, "text": 0, "color": 0, "content_gate": 0.2}, None


# ── Single multi-turn rollout ────────────────────────────────────────────────


def run_episode(
    renderer,
    sampling_client: tinker.SamplingClient,
    sampling_params: types.SamplingParams,
    page,
    example: dict,
    max_turns: int,
) -> tuple[float, list[dict]]:
    """
    Run one multi-turn episode for a single example.

    Returns:
      final_reward: the reward from the last turn
      turn_records: list of per-turn dicts, each with:
        - prompt (ModelInput): tokenized prompt for this turn
        - tokens (list[int]): sampled response tokens
        - logprobs (list[float]): per-token logprobs
        - reward (float): reward at this turn
        - raw_text (str): decoded model output
    """
    messages, model_input = build_initial_prompt(renderer, example["screenshot"])

    # Pre-compute reference info once (avoids re-rendering each turn)
    ref_info = extract_ref_info(page, example["reference_html"])

    turn_records: list[dict] = []

    for turn in range(max_turns):
        # Sample from the model
        response = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

        seq = response.sequences[0]
        assert seq.logprobs is not None

        # Decode response
        parsed_msg, _ok = renderer.parse_response(seq.tokens)
        raw_text = get_text_content(parsed_msg)
        generated_html = extract_html_from_response(raw_text)

        # Score this turn's output
        reward, details, diff_pil = score_and_diff(
            page, example["reference_html"], generated_html, ref_info=ref_info,
        )

        turn_records.append({
            "prompt": model_input,
            "tokens": seq.tokens,
            "logprobs": seq.logprobs,
            "reward": reward,
            "raw_text": raw_text,
        })

        # If this is the last turn, or we can't produce a diff, stop
        if turn == max_turns - 1 or diff_pil is None:
            break

        # Build next turn's prompt with diff feedback
        messages, model_input = build_feedback_prompt(
            renderer, messages, raw_text, diff_pil,
        )

    return turn_records[-1]["reward"], turn_records


# ── Datum construction from multi-turn episode ───────────────────────────────


def episode_to_datums(
    turn_records: list[dict],
    advantage: float,
) -> list[types.Datum]:
    """
    Convert a multi-turn episode into training datums.

    Each turn becomes a separate Datum because the renderer strips thinking
    tokens from history, so turn N's tokenized prompt is NOT a prefix of
    turn N+1's prompt. The advantage (from GRPO) is applied uniformly to
    all action tokens across all turns.
    """
    datums = []

    for record in turn_records:
        prompt: tinker.ModelInput = record["prompt"]
        sampled_tokens: list[int] = record["tokens"]
        logprobs: list[float] = record["logprobs"]

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

    return datums


# ── Main training loop ───────────────────────────────────────────────────────


def train(
    n_batches: int = 20,
    max_turns: int = MAX_TURNS,
    resume: bool = False,
    log_path: str = MULTI_TURN_LOG_PATH,
):
    """
    Run multi-turn GRPO training.

    Each batch:
      - Picks BATCH_SIZE examples
      - Runs GROUP_SIZE multi-turn episodes per example
      - Each episode: model generates HTML, gets diff feedback, corrects
      - Computes GRPO advantages from final-turn rewards
      - Builds one Datum per turn per episode, runs gradient step
    """
    log_path_obj = Path(log_path)
    log_path_obj.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ─────────────────────────────────────────────────────
    dataset = load_training_data()
    n_total = len(dataset)
    log.info("Loaded %d training examples", n_total)

    total_possible_batches = n_total // BATCH_SIZE
    n_batches = min(n_batches, total_possible_batches)
    log.info(
        "Will train for %d batches (batch=%d, group=%d, turns=%d)",
        n_batches, BATCH_SIZE, GROUP_SIZE, max_turns,
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

    # ── Launch Playwright ────────────────────────────────────────────────
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
                _save_checkpoint(
                    training_client=training_client,
                    name=f"{batch_idx:06d}",
                    log_path=str(log_path_obj),
                    kind="state",
                    loop_state={"batch": batch_idx},
                    ttl_seconds=604800,
                )

            # ── Select batch examples ────────────────────────────────
            batch_start = (batch_idx * BATCH_SIZE) % n_total
            batch_end = batch_start + BATCH_SIZE
            if batch_end > n_total:
                batch_examples = dataset[batch_start:] + dataset[:batch_end - n_total]
            else:
                batch_examples = dataset[batch_start:batch_end]

            log.info(
                "Batch %d: examples %d-%d",
                batch_idx, batch_start, batch_start + len(batch_examples) - 1,
            )

            # ── Get current policy ───────────────────────────────────
            sampling_client = training_client.save_weights_and_get_sampling_client()

            # ── Run multi-turn episodes ──────────────────────────────
            # For each example, run GROUP_SIZE episodes.
            # Episodes are sequential (each turn depends on the previous),
            # but we process all group members for an example before moving on.

            datums: list[types.Datum] = []
            batch_final_rewards: list[float] = []
            batch_turn_counts: list[float] = []
            batch_skipped = 0

            for ex in batch_examples:
                # Run GROUP_SIZE episodes for this example
                final_rewards_G: list[float] = []
                episodes_G: list[list[dict]] = []

                for g in range(GROUP_SIZE):
                    final_reward, turn_records = run_episode(
                        renderer=renderer,
                        sampling_client=sampling_client,
                        sampling_params=sampling_params,
                        page=page,
                        example=ex,
                        max_turns=max_turns,
                    )
                    final_rewards_G.append(final_reward)
                    episodes_G.append(turn_records)
                    batch_turn_counts.append(len(turn_records))

                # ── GRPO advantages from final rewards ───────────────
                mean_reward = sum(final_rewards_G) / len(final_rewards_G)
                advantages_G = [r - mean_reward for r in final_rewards_G]
                batch_final_rewards.append(mean_reward)

                # Skip if all advantages are zero
                if all(a == 0.0 for a in advantages_G):
                    batch_skipped += 1
                    continue

                # ── Build datums from all episodes ───────────────────
                for episode, advantage in zip(episodes_G, advantages_G):
                    datums.extend(episode_to_datums(episode, advantage))

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

                fwd_bwd_future = training_client.forward_backward(
                    datums, loss_fn="importance_sampling"
                )
                optim_future = training_client.optim_step(adam_params)

                _fwd_bwd_result = fwd_bwd_future.result()
                optim_result = optim_future.result()

            # ── Log metrics ──────────────────────────────────────────
            elapsed = time.time() - t_start
            mean_final = (
                sum(batch_final_rewards) / len(batch_final_rewards)
                if batch_final_rewards
                else 0.0
            )
            mean_turns = (
                sum(batch_turn_counts) / len(batch_turn_counts)
                if batch_turn_counts
                else 0.0
            )

            metrics = {
                "batch": batch_idx,
                "reward_mean": mean_final,
                "reward_min": min(batch_final_rewards) if batch_final_rewards else 0.0,
                "reward_max": max(batch_final_rewards) if batch_final_rewards else 0.0,
                "avg_turns": mean_turns,
                "n_datums": len(datums),
                "n_skipped": batch_skipped,
                "time_s": elapsed,
                "lr": LEARNING_RATE,
            }
            all_metrics.append(metrics)

            log.info(
                "Batch %d complete: reward=%.3f  turns=%.1f  datums=%d  skipped=%d  time=%.1fs",
                batch_idx,
                mean_final,
                mean_turns,
                len(datums),
                batch_skipped,
                elapsed,
            )

        # ── Final checkpoint ─────────────────────────────────────────
        final_batch = start_batch + n_batches
        log.info("Saving final checkpoint...")
        _save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=str(log_path_obj),
            kind="both",
            loop_state={"batch": final_batch},
            ttl_seconds=None,
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
    all_turns = [m["avg_turns"] for m in all_metrics]
    print("\n" + "=" * 60)
    print("  MULTI-TURN TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:       {MODEL_NAME}")
    print(f"  Batches:     {len(all_metrics)}")
    print(f"  Group size:  {GROUP_SIZE}")
    print(f"  Max turns:   {max_turns}")
    print(f"  LoRA rank:   {LORA_RANK}")
    print(f"  LR:          {LEARNING_RATE}")
    print()
    if all_rewards:
        early = all_rewards[:3]
        late = all_rewards[-3:]
        print(f"  Reward (early avg):  {sum(early)/len(early):+.3f}")
        print(f"  Reward (late avg):   {sum(late)/len(late):+.3f}")
        print(f"  Reward (overall):    {sum(all_rewards)/len(all_rewards):+.3f}")
        print(f"  Avg turns/episode:   {sum(all_turns)/len(all_turns):.1f}")
    print()
    print(f"  Checkpoints: {log_path_obj}")
    print(f"  Metrics:     {metrics_path}")
    print("=" * 60 + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Multi-turn GRPO training for Design2Code"
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=20,
        help="Number of training batches (default: 20)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=MAX_TURNS,
        help=f"Max conversation turns per episode (default: {MAX_TURNS})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint in --log-path",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=MULTI_TURN_LOG_PATH,
        help=f"Directory for checkpoints and logs (default: {MULTI_TURN_LOG_PATH})",
    )
    args = parser.parse_args()
    train(
        n_batches=args.batches,
        max_turns=args.turns,
        resume=args.resume,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
