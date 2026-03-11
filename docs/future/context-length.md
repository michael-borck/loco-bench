---
title: "Context Length Effects on Small Model Inference"
status: future-spinoff
---

!!! note "Future Research"
    This document captures open questions about how context length affects quality, speed, and VRAM usage for models in the 1–7B range. These experiments are planned but not yet implemented.

Context length is one of the least-explored dimensions in small model benchmarking. Most benchmarks evaluate at short context (under 1K tokens), but real-world usage — chat, RAG, document summarisation — routinely pushes models to 4K–32K tokens. The effects on small models are likely more severe than on large models, but systematic measurement is sparse.

---

## Three Dimensions of Context Effects

### 1. VRAM Usage

The KV cache grows linearly with context length and consumes VRAM beyond the model weights themselves. vram-bench's tier tables assume short context (~512 tokens), where KV cache overhead is minimal (~0.5 GB). At longer contexts, the picture changes:

| Context Length | Approximate KV Cache Overhead |
|---|---|
| 512 tokens | ~0.1–0.5 GB |
| 2K tokens | ~0.3–1.0 GB |
| 8K tokens | ~1.0–2.5 GB |
| 32K tokens | ~3.0–8.0 GB |

Exact overhead depends on the model's architecture — number of attention heads, head dimension, number of layers, and whether the model uses grouped-query attention (GQA) or multi-query attention (MQA), both of which significantly reduce KV cache size.

**The vram-bench question:** At what context length does a model that "fits" in a tier stop fitting? This directly affects the practical value of our tier recommendations.

### 2. Generation Speed

Token generation speed degrades as context grows. Each new token must attend to all previous tokens, so the cost per token increases. For bandwidth-bound inference on consumer GPUs, this manifests as:

- **Prompt processing** (prefill) slows proportionally with prompt length
- **Token generation** slows as the KV cache grows — each attention step reads more data from VRAM
- **Time-to-first-token** increases with longer prompts

`llama-bench` already supports testing at different context lengths via its prompt and generation length parameters. A systematic sweep across 512 / 2K / 8K / 16K / 32K would quantify the degradation curve for each model.

**The vram-bench question:** How much throughput do you lose at realistic context lengths? If a model generates 40 t/s at 512 tokens but 15 t/s at 8K, that changes the practical recommendation.

### 3. Quality Degradation

Small models claim large context windows — Qwen3-4B supports 32K, Llama 3.2 claims 128K — but quality degrades well before the architectural maximum, especially for models in our size range.

**"Lost in the middle"** is a documented phenomenon: models attend well to the beginning and end of their context but lose information placed in the middle. This is more pronounced in smaller models with fewer attention heads and layers.

Three quality effects to measure:

- **Retrieval accuracy at position** — can the model find a specific fact placed at various positions in a long context? (needle-in-a-haystack)
- **Reasoning over long context** — does multi-step reasoning quality degrade as supporting information is spread across a longer context?
- **Effective context vs. claimed context** — at what context length does the model's performance become meaningfully worse than at short context?

**The vram-bench question:** What is the *effective* context length for each model — the point beyond which quality drops below a useful threshold? A 4B model with 32K advertised context but 4K effective context is a very different recommendation than the spec sheet suggests.

---

## Proposed Experiment Design

### Phase 1: VRAM and Speed (Measurable Now)

These experiments use existing tooling and require no new evaluation harnesses:

1. **KV cache VRAM measurement** — Load each model at 512 / 2K / 8K / 16K / 32K context and measure peak VRAM. Record which models fall out of which tiers at each context length.

2. **Speed degradation curves** — Run `llama-bench` at each context length and plot tokens/sec vs context length per model. This produces a "speed at context" chart complementing the existing short-context speed benchmarks.

3. **Updated tier tables** — Add a column or footnote to tier tables showing maximum context length that still fits within the tier's VRAM budget.

### Phase 2: Quality at Context (Requires New Evaluation)

These experiments need a different evaluation approach than standard benchmarks:

1. **Needle-in-a-haystack** — Insert a target fact at various positions (start, 25%, 50%, 75%, end) in contexts of increasing length. Measure retrieval accuracy. This is a standard test with existing implementations.

2. **Long-context reasoning** — Tasks that require synthesising information from multiple positions in the context. Less standardised, but benchmarks like SCROLLS and LongBench provide starting points.

3. **Perplexity at position** — Measure per-token perplexity as a function of position in the context. Rising perplexity in the middle of the context window is a signal of the "lost in the middle" effect.

---

## Why This Matters for vram-bench

vram-bench's core question is "what's the best model for your VRAM budget?" Context length adds a crucial qualifier: **best for what context length?**

A user doing short-prompt Q&A cares about different models than a user doing RAG with 8K context documents. If a model that wins at short context loses its VRAM advantage or quality advantage at longer context, that changes the recommendation.

At minimum, vram-bench should document the context assumptions behind its tier tables. Ideally, it should provide context-aware recommendations — but that requires the experiments above.

---

*Part of the [vram-bench](https://github.com/michael-borck/vram-bench) project. Related: [VRAM Tier Reference](../tiers.md), [Speed Analysis](../speed.md).*
