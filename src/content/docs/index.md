---
title: "LocoBench"
---

**I have X GB of VRAM -- what's the best model I can run?**

HuggingFace tells you which small model is best at full precision. The Open LLM Leaderboard tells you which large model is best on datacenter GPUs. **LocoBench tells you which model is best for your actual card.**

Within each VRAM tier, full-precision small models compete head-to-head against quantized larger models. A BF16 SmolLM2-1.7B and a Q4_K_M Qwen3-4B both fit in 4GB -- but which one actually wins?

:::note[Sample Data]
These charts currently show **simulated data** generated with realistic degradation curves. They will be replaced with real benchmark results once the full evaluation matrix completes.
:::

:::tip[Key Finding]
Q4_K_M consistently sits in the efficiency sweet spot -- retaining 90-95% of BF16 quality at ~30% of the file size. Below Q3_K_M, quality collapses sharply for knowledge-heavy tasks. But for the smallest VRAM budgets, full-precision sub-2B models may outperform heavily quantized larger ones.
:::

## Composite Score vs File Size

Each point is one model at one precision level (full-precision or quantized). The **orange line** traces the Pareto frontier -- the best quality achievable at each file size. Points above and to the left are more efficient.

<div id="chart-overview-scatter" class="plotly-chart"></div>

## Top 10 Variants by Composite Score

Composite score averages across MMLU, HellaSwag, GSM8K, TruthfulQA, and ARC-Challenge.

<div id="chart-overview-leaderboard" class="plotly-chart"></div>

---

## Explore the Details

- **[Quality Analysis](quality)** -- Per-task scores and quantization degradation curves
- **[Speed Analysis](speed)** -- Generation speed, prompt processing, and time-to-first-token
- **[Bang per Bit](bang-per-bit)** -- Pareto efficiency, quality-speed tradeoffs, and task sensitivity

---

## Why This Exists

Most published benchmarks compare models under ideal conditions -- full precision, datacenter GPUs. Nobody systematically compares everything that fits within a given VRAM budget on consumer hardware. LocoBench fills that gap.

The organising principle is the hardware constraint. Models are grouped by VRAM tier (4GB, 6GB, 8GB, 12GB, 24GB), and within each tier, every model that fits -- whether full-precision or quantized -- competes on quality, speed, and efficiency.

The data is useful for anyone choosing a model for local deployment, and particularly for projects like [LocoLLM](https://locollm.org) that build on top of small models for consumer hardware.

**Methodology:** All quality benchmarks use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Speed benchmarks use [llama-bench](https://github.com/ggml-org/llama.cpp) on CPU-only (0 GPU layers). See the [Benchmarking Guide](guide) for full details.
