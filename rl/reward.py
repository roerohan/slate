"""
Reward function for Design2Code reinforcement learning.

Scores how well generated HTML visually matches a reference screenshot.
Uses content-gated SSIM + text similarity + color palette similarity.

    reward = 2 * (W_SSIM * gated_ssim + W_TEXT * text + W_COLOR * color) - 1

SSIM is gated by a content signal so blank pages cannot exploit high SSIM
against light-background references.  The gradient is continuous everywhere
— no hard -1 cliffs.

Usage:
    from playwright.sync_api import sync_playwright
    from rl.reward import extract_ref_info, extract_gen_info, compute_reward_from_info

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})

        ref  = extract_ref_info(page, reference_html)
        gen  = extract_gen_info(page, generated_html)
        reward, details = compute_reward_from_info(ref, gen)

        browser.close()
"""

from __future__ import annotations

import io
import re
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
from PIL import Image
from playwright.sync_api import Page
from skimage.metrics import structural_similarity as ssim

from rl.config import (
    COLOR_QUANT_STEP,
    GATE_FLOOR,
    SSIM_SIZE,
    TAILWIND_CDN,
    VIEWPORT_H,
    VIEWPORT_W,
    W_COLOR,
    W_SSIM,
    W_TEXT,
)

# ── HTML extraction from model responses ─────────────────────────────────────


def extract_html_from_response(text: str) -> str | None:
    """
    Extract HTML from a model's text response.

    Tries, in order:
      1. ```html ... ``` fenced block
      2. ``` ... ``` fenced block starting with a tag
      3. Raw text that is entirely an HTML fragment
      4. First recognisable HTML tag onward
    Returns None if nothing looks like HTML.
    """
    # 1. Explicit ```html fence
    match = re.search(r"```html\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. Generic fence starting with a tag
    match = re.search(r"```\s*(<!?[^`]*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 3. Entire response is HTML
    stripped = text.strip()
    if stripped.startswith("<") and stripped.endswith(">"):
        return stripped

    # 4. First recognisable tag onward
    match = re.search(
        r"(<(?:style|div|span|h[1-6]|p|section|header|nav|main|footer|html|!DOCTYPE)[^>]*>.*)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    return None


# ── HTML rendering via Playwright ────────────────────────────────────────────


def is_full_html(html: str) -> bool:
    """Check whether *html* looks like a complete document (has doctype or <html>)."""
    lowered = html.strip().lower()
    return lowered.startswith("<!doctype") or lowered.startswith("<html")


def _wrap_snippet(snippet: str) -> str:
    """Wrap a body-only HTML snippet in a full document shell with Tailwind CDN."""
    return (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'>\n"
        f'<link href="{TAILWIND_CDN}" rel="stylesheet">\n'
        "<style>body{margin:20px;background:#fff}</style>\n"
        "</head><body>\n"
        f"{snippet}\n"
        "</body></html>"
    )


def render_html(page: Page, html: str) -> None:
    """
    Render *html* in a Playwright page.

    Wraps snippets automatically.  Waits for network idle so the Tailwind
    CDN stylesheet has time to load.
    """
    payload = html if is_full_html(html) else _wrap_snippet(html)
    try:
        page.set_content(payload, timeout=5_000)
    except Exception:
        # Timeout on set_content — reset to blank and return
        page.set_content("<html><body></body></html>", timeout=2_000)
        return

    try:
        page.wait_for_load_state("networkidle", timeout=3_000)
    except Exception:
        # Network didn't settle — give it a short grace period
        page.wait_for_timeout(200)


def render_html_to_image(
    page: Page, html: str, size: int = SSIM_SIZE
) -> np.ndarray:
    """Render HTML and return a (size x size) RGB numpy array."""
    render_html(page, html)
    buf = page.screenshot()
    img = Image.open(io.BytesIO(buf)).convert("RGB").resize((size, size))
    return np.array(img)


def render_html_to_file(
    page: Page,
    html: str | None,
    save_path: str,
    full_page: bool = True,
) -> bool:
    """
    Render HTML and save the screenshot to *save_path*.

    Returns True on success, False if *html* was None or rendering failed
    (a grey placeholder is written instead).
    """
    if html is None:
        Image.new("RGB", (VIEWPORT_W, VIEWPORT_H), (240, 240, 240)).save(save_path)
        return False
    try:
        render_html(page, html)
        page.wait_for_timeout(100)
        page.screenshot(path=save_path, full_page=full_page)
        return True
    except Exception:
        Image.new("RGB", (VIEWPORT_W, VIEWPORT_H), (240, 240, 240)).save(save_path)
        return False


# ── Visual diff image ────────────────────────────────────────────────────────


def make_diff_image(
    ref_img: np.ndarray,
    gen_img: np.ndarray,
    threshold: int = 25,
) -> np.ndarray:
    """
    Create a diff image highlighting pixel differences in red.

    Pixels where ref and gen differ by more than *threshold* (per channel)
    are overlaid with a semi-transparent red.  Returns an RGB numpy array
    the same size as *ref_img*.
    """
    if ref_img.shape != gen_img.shape:
        gen_img = np.array(
            Image.fromarray(gen_img).resize((ref_img.shape[1], ref_img.shape[0]))
        )

    diff = np.abs(ref_img.astype(int) - gen_img.astype(int))
    mask = np.any(diff > threshold, axis=2)

    result = gen_img.copy()
    result[mask] = (
        result[mask].astype(int) * 0.3 + np.array([255, 0, 0]) * 0.7
    ).astype(np.uint8)
    return result


# ── DOM extraction (single JS round-trip) ────────────────────────────────────

# Tags we consider "meaningful" when analysing layout blocks.
MEANINGFUL_TAGS = frozenset(
    {
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "a", "button", "input", "img",
        "nav", "header", "footer", "main", "section", "article",
        "li", "td", "th", "label", "span", "textarea", "select",
    }
)

# JavaScript executed inside the browser to extract text, layout blocks, and
# dominant colours — all in one evaluate() call to avoid round-trip overhead.
_EXTRACT_JS = """(tagsList) => {
    const meaningfulTags = new Set(tagsList);
    const blocks = [];
    const colors = [];
    const els = document.querySelectorAll('*');

    for (const el of els) {
        const rect = el.getBoundingClientRect();
        if (rect.width < 5 || rect.height < 5) continue;

        const tag = el.tagName.toLowerCase();
        const style = getComputedStyle(el);

        // Skip invisible elements
        if (style.display === 'none' || style.visibility === 'hidden' ||
            parseFloat(style.opacity) === 0) continue;

        // ── Colors (from all visible elements with real area) ──
        if (rect.width >= 10 && rect.height >= 10) {
            const bg = style.backgroundColor;
            if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
                const m = bg.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                if (m) colors.push([parseInt(m[1]), parseInt(m[2]), parseInt(m[3])]);
            }
            const fg = style.color;
            if (fg) {
                const m = fg.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                if (m) colors.push([parseInt(m[1]), parseInt(m[2]), parseInt(m[3])]);
            }
        }

        // ── Layout blocks (meaningful elements only) ──
        // Skip elements that fill the entire viewport (html, body wrappers)
        if (rect.width >= window.innerWidth * 0.99 &&
            rect.height >= window.innerHeight * 0.99) continue;

        const hasBg = style.backgroundColor !== 'rgba(0, 0, 0, 0)' &&
                       style.backgroundColor !== 'transparent';
        const hasBorder = style.borderWidth && style.borderWidth !== '0px';

        let directText = '';
        for (const node of el.childNodes) {
            if (node.nodeType === Node.TEXT_NODE) {
                directText += node.textContent;
            }
        }
        directText = directText.trim();

        if (meaningfulTags.has(tag) || directText.length > 0 || hasBg || hasBorder) {
            blocks.push({
                tag: tag,
                x: rect.x,
                y: rect.y,
                w: rect.width,
                h: rect.height,
                text: directText.substring(0, 200),
                fontSize: parseFloat(style.fontSize) || 0,
                fontWeight: style.fontWeight || '400',
                color: style.color || '',
                bgColor: style.backgroundColor || '',
                borderRadius: style.borderRadius || '0px',
                padding: style.padding || '0px',
            });
        }
    }

    return {
        text: document.body ? document.body.innerText.trim() : '',
        blocks: blocks,
        colors: colors,
    };
}"""


def extract_dom_info(page: Page) -> dict:
    """
    Extract text content, layout blocks, and colour palette from the current
    page state.  Runs a single JS evaluate() call — no redundant round-trips.
    """
    return page.evaluate(_EXTRACT_JS, list(MEANINGFUL_TAGS))


def extract_ref_info(
    page: Page, reference_html: str, size: int = SSIM_SIZE
) -> dict:
    """
    Render *reference_html* and extract all information needed for reward
    computation: text, layout blocks, colours, and a (size x size) screenshot.

    Call once per prompt; reuse across all rollouts in the group.
    """
    render_html(page, reference_html)
    info = extract_dom_info(page)
    buf = page.screenshot()
    img = Image.open(io.BytesIO(buf)).convert("RGB").resize((size, size))
    info["image"] = np.array(img)
    return info


def extract_gen_info(
    page: Page, generated_html: str, size: int = SSIM_SIZE
) -> dict:
    """
    Render *generated_html* and extract DOM info + screenshots.

    Returns the same fields as ``extract_ref_info`` plus ``image_full``
    (the full-resolution screenshot, useful for blank-frame checks or
    diff-image generation).
    """
    render_html(page, generated_html)
    info = extract_dom_info(page)
    buf = page.screenshot()
    full_img = Image.open(io.BytesIO(buf)).convert("RGB")
    info["image_full"] = np.array(full_img)
    info["image"] = np.array(full_img.resize((size, size)))
    return info


# ── Comparison functions ─────────────────────────────────────────────────────


def text_similarity(ref_text: str, gen_text: str) -> float:
    """
    Global text-content similarity via SequenceMatcher.

    Returns a float in [0, 1].
    """
    if not ref_text and not gen_text:
        return 1.0
    if not ref_text or not gen_text:
        return 0.0
    return SequenceMatcher(None, ref_text.lower(), gen_text.lower()).ratio()


def _quantize_color(c: tuple[int, ...]) -> tuple[int, ...]:
    step = COLOR_QUANT_STEP
    return (c[0] // step * step, c[1] // step * step, c[2] // step * step)


def color_palette_similarity(ref_colors: list, gen_colors: list) -> float:
    """
    Histogram-intersection similarity over quantized colour palettes.

    Returns a float in [0, 1].
    """
    if not ref_colors and not gen_colors:
        return 1.0
    if not ref_colors or not gen_colors:
        return 0.0

    ref_hist = Counter(_quantize_color(tuple(c)) for c in ref_colors)
    gen_hist = Counter(_quantize_color(tuple(c)) for c in gen_colors)

    all_colors = set(ref_hist.keys()) | set(gen_hist.keys())
    intersection = sum(
        min(ref_hist.get(c, 0), gen_hist.get(c, 0)) for c in all_colors
    )
    total = max(sum(ref_hist.values()), sum(gen_hist.values()))

    return float(intersection / total) if total > 0 else 0.0


def visual_similarity(ref_img: np.ndarray, gen_img: np.ndarray) -> float:
    """
    Structural similarity (SSIM) between two images.

    Both images should be the same shape (use SSIM_SIZE x SSIM_SIZE).
    Returns a float in [-1, 1] (in practice almost always [0, 1]).
    """
    if ref_img.shape != gen_img.shape:
        gen_img = np.array(
            Image.fromarray(gen_img).resize(
                (ref_img.shape[1], ref_img.shape[0])
            )
        )
    return float(ssim(ref_img, gen_img, channel_axis=2, data_range=255))


# ── Combined reward ──────────────────────────────────────────────────────────


def compute_reward_from_info(
    ref_info: dict, gen_info: dict
) -> tuple[float, dict]:
    """
    Compute the Design2Code reward from pre-extracted reference and generated
    page information.

    Returns
    -------
    reward : float
        Score in [-1, +1].
    details : dict
        Breakdown of sub-scores for logging / debugging:
        ``ssim``, ``text``, ``color``, ``content_gate``, ``reward``.

    Reward formula
    --------------
    ::

        content_gate = GATE_FLOOR + (1 - GATE_FLOOR) * max(text, color)
        gated_ssim   = ssim * content_gate

        raw    = W_SSIM * gated_ssim + W_TEXT * text + W_COLOR * color
        reward = 2 * raw - 1
    """
    details = {
        "ssim": visual_similarity(ref_info["image"], gen_info["image"]),
        "text": text_similarity(ref_info["text"], gen_info["text"]),
        "color": color_palette_similarity(ref_info["colors"], gen_info["colors"]),
    }

    # Content gate: SSIM only gets full credit when the page has real content.
    # Blank page (text=0, color=0) → gate=GATE_FLOOR, SSIM contribution crushed.
    # Real attempt (text>0 or color>0) → gate opens smoothly toward 1.0.
    content = max(details["text"], details["color"])
    content_gate = GATE_FLOOR + (1.0 - GATE_FLOOR) * content
    gated_ssim = details["ssim"] * content_gate

    raw = W_SSIM * gated_ssim + W_TEXT * details["text"] + W_COLOR * details["color"]

    # Map [0, 1] → [-1, +1]
    reward = 2.0 * raw - 1.0

    details["content_gate"] = content_gate
    details["reward"] = reward
    return float(reward), details
