# Flourishing AI — Paragraph Scorer & Regenerator

A Streamlit app that scores any paragraph across the seven **Flourishing AI (FAI)**
dimensions using a [Gloo AI](https://platform.ai.gloo.com) model, then regenerates
the paragraph — live — to raise the score without losing its intent.

FAI (by Gloo + Valkyrie Intelligence) measures whether text *actively supports
human flourishing*, not just whether it avoids harm. The seven dimensions:

**Character · Relationships · Happiness · Meaning · Health · Finances · Faith**

## How it works

1. **Score** — a Gloo model rates the text 0–100 on each dimension using that
   dimension's published rubric.
2. **Aggregate** — the seven scores combine by **geometric mean**
   (`FAI = ⁷√(χ·ρ·η·μ·ψ·φ·σ)`). This is *non-compensatory*: the weakest
   dimension holds the total back, so a high score requires being well-rounded.
3. **Regenerate** — the app finds the weakest dimensions, rewrites the paragraph
   to strengthen them (preserving your topic, voice and intent — no forced
   content), rescores, and repeats up to 3 rounds until it hits the target.

The UI shows the whole loop live: `weak spot → rewrite → rescore`, with
before/after dimension bars.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then paste your Gloo API key into .env
streamlit run app.py
```

## Project layout

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `gloo_client.py` | Minimal Gloo AI client (chat completions v2, JSON mode) |
| `fai/dimensions.py` | The seven dimensions, symbols and judging rubrics |
| `fai/scoring.py` | FAI geometric-mean scoring and the supporting maths |
| `fai/judge.py` | Scores a paragraph on all seven dimensions via a Gloo model |
| `fai/rewrite.py` | Iterative, intent-preserving regeneration toward a target |
| `demo_math.py` | Runs the scoring maths on synthetic data (no API key needed) |

## Notes

- Requires a Gloo AI API key (`sk_...`). It is read from `.env` and never
  committed (`.env` is gitignored).
- The FAI scoring maths here is a faithful reconstruction of the published
  framework; the full 807-item benchmark question set is not public.
