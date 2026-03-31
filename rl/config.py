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
