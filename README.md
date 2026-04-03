# Slate

Slate is my first attempt at training a vision-language model (VLM) to generate HTML from a screenshot of a webpage. You give it a picture of a site, it writes the code.

## How it works

1. **Dataset**: ~500 screenshot/HTML pairs from HuggingFace's WebSight dataset. All Tailwind CSS, rendered at 1280x720.

2. **Reward function**: Renders the model's HTML in a browser, compares it to the original screenshot using SSIM (visual similarity), text matching, and color palette comparison. Score from -1 to +1.

3. **Evaluation**: Sends screenshots to the model via [tinker](https://tinker-docs.thinkingmachines.ai/), scores the generated HTML, saves artifacts (generated screenshots, diff images, per-example scores).

4. **Training**: GRPO (Group Relative Policy Optimization). For each screenshot, the model writes multiple HTML attempts. The ones that score above average get reinforced, the ones below average get discouraged. Uses LoRA so we only train a small adapter, not the full model.

## Setup

```bash
npm install
pip install -r requirements.txt
playwright install chromium
```

Copy `.env.example` to `.env` and fill in your Cloudflare and Tinker API keys.

## Usage

```bash
# Generate the dataset
pnpm run prepare-data # or use npm

# Evaluate the base model
source venv/bin/activate
python -m rl.eval

# Train
python -m rl.training.single_shot

# Evaluate the trained model
python -m rl.eval --model-path tinker://...
```

## Stack

- TypeScript (data pipeline)
- Python (training, eval, reward)
- Tinker SDK (model inference + LoRA training)
- Playwright (HTML rendering for reward scoring)
- Qwen 3.5-4B (base VLM, you can change it!)

## License

MIT
