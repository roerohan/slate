"""
Centralized configuration for the RL reward pipeline.

Mirrors relevant values from src/config.ts to keep the two halves
of the project consistent.
"""

# ── Viewport (must match the data-pipeline screenshots) ──────────────────────

VIEWPORT_W = 1280
VIEWPORT_H = 720

# ── SSIM comparison resolution ───────────────────────────────────────────────
# Screenshots are downscaled to this square size before computing SSIM.
# 256x256 is a good speed/quality tradeoff — SSIM is robust to resolution.

SSIM_SIZE = 256

# ── Reward weights ───────────────────────────────────────────────────────────
# raw = W_SSIM * gated_ssim + W_TEXT * text + W_COLOR * color
# reward = 2 * raw - 1   →  [-1, +1]

W_SSIM = 0.60
W_TEXT = 0.25
W_COLOR = 0.15

# ── Content gate ─────────────────────────────────────────────────────────────
# SSIM is multiplied by a content gate to prevent blank-page reward hacking.
# gate = GATE_FLOOR + (1 - GATE_FLOOR) * max(text_score, color_score)
# A blank page (text=0, color=0) gets gate=GATE_FLOOR, crushing SSIM credit.

GATE_FLOOR = 0.2

# ── Color quantization ──────────────────────────────────────────────────────
# Colors are bucketed into bins of this width before histogram comparison.

COLOR_QUANT_STEP = 32

# ── Tailwind CDN ─────────────────────────────────────────────────────────────
# Used when wrapping HTML snippets into full documents for rendering.

TAILWIND_CDN = "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"

# ── Model / inference ────────────────────────────────────────────────────────
# Used by eval.py and (later) train.py for Tinker VLM inference.

MODEL_NAME = "Qwen/Qwen3.5-4B"
RENDERER_NAME = "qwen3_5_disable_thinking"

# ── Generation parameters ────────────────────────────────────────────────────

MAX_TOKENS = 8192          # Upper bound on generated HTML tokens
TEMPERATURE = 0.7          # Sampling temperature for generation
TOP_P = 1.0
TOP_K = -1                 # -1 = disabled

# ── Eval settings ────────────────────────────────────────────────────────────

EVAL_SUBSET = 50           # Number of examples to evaluate on
EVAL_CONCURRENCY = 8       # Max parallel Tinker sampling requests
MAX_IMAGE_SIZE = 480       # Longest side of screenshot sent to the VLM

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert web developer. Given a screenshot of a webpage, "
    "generate the HTML code that would produce this exact visual output. "
    "Use Tailwind CSS classes for styling. "
    "Output ONLY the HTML code inside a ```html code block. "
    "Do not include <!DOCTYPE>, <html>, <head>, or <body> tags — "
    "just the body content with inline <style> blocks if needed."
)

# ── Training settings ────────────────────────────────────────────────────────

LEARNING_RATE = 4e-5       # AdamW learning rate
LORA_RANK = 32             # LoRA adapter rank (higher = more capacity, more VRAM)
GROUP_SIZE = 4             # GRPO: number of completions per prompt
BATCH_SIZE = 4             # Prompts per training batch (total datums = BATCH_SIZE * GROUP_SIZE)
TRAIN_TEMPERATURE = 1.0    # Higher than eval to encourage exploration during RL
SAVE_EVERY = 5             # Save checkpoint every N batches (0 = disabled)
LOG_PATH = "data/training" # Directory for training logs and checkpoints
