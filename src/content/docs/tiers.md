---
title: "VRAM Tier Reference"
---

The organising principle of loco-bench: **models are grouped by what fits in your VRAM budget, not by model family or parameter count.**

Within each tier, every model that fits — at any precision level — competes on quality, speed, and efficiency.

## Size Estimation

GGUF file sizes can be estimated from the parameter count and precision level:

```
File size (GB) ≈ parameters (B) × bits_per_weight / 8 × 1.05
```

VRAM usage during inference is higher than file size due to KV cache and runtime overhead:

```
VRAM needed ≈ file_size × 1.2 + 0.5 GB
```

These are conservative estimates. Actual usage depends on context length, batch size, and framework.

:::caution[Context Length and VRAM]
The estimates above assume **short context** (~512 tokens), which matches the standard benchmarks (MMLU, HellaSwag, GSM8K, etc.). Longer contexts increase VRAM usage through the KV cache -- at 8K-32K tokens, KV cache can add 1-4 GB depending on model architecture. A model that fits comfortably in a tier at short context may not fit at longer context lengths. Context length effects on small models are an active area of research in the lab.
:::

## Tier Definitions

### 2GB Tier — GTX 950

**Usable VRAM:** ~1.5 GB (after OS/driver reservation)
**Benchmark GPU:** GTX 950 (105 GB/s) — Tortuga

The absolute floor. 2 GB Maxwell-era VRAM severely limits model selection. Most mainstream models will not fit, but sub-1B models at aggressive quantisation are testable. This tier exists to document whether the floor is usable at all.

| Model | Precision | File Size | VRAM Est. | Fits? |
|---|---|---|---|---|
| SmolLM2-135M | BF16 | 0.3 GB | 0.9 GB | Yes |
| SmolLM2-360M | BF16 | 0.7 GB | 1.3 GB | Yes |
| TinyLlama-1.1B | Q4_K_M | 0.7 GB | 1.3 GB | Yes |
| Llama 3.2-1B | Q4_K_M | 0.7 GB | 1.3 GB | Yes |
| Qwen3-1.7B | Q2_K | 0.7 GB | 1.3 GB | Tight |

**Key question:** Is there any useful inference at 2 GB? If a Q4_K_M 1B model produces coherent conversational responses at >3 tok/s, the floor is lower than most guides claim.

---

### 3GB Tier — GTX 1060 3GB

**Usable VRAM:** ~2.5 GB
**Benchmark GPU:** GTX 1060 3 GB (192 GB/s) — Tortuga

An unusual tier -- 3 GB sits between the 2 GB floor and the 4 GB entry point. The GTX 1060 3 GB is faster than the GTX 950 (192 vs 105 GB/s bandwidth) despite only 1 GB more VRAM. This tier tests whether the bandwidth advantage opens up models that the 2 GB tier cannot run.

| Model | Precision | File Size | VRAM Est. | Fits? |
|---|---|---|---|---|
| SmolLM2-135M | BF16 | 0.3 GB | 0.9 GB | Yes |
| SmolLM2-360M | BF16 | 0.7 GB | 1.3 GB | Yes |
| TinyLlama-1.1B | BF16 | 2.2 GB | 3.1 GB | Tight |
| Gemma 3-1B | Q4_K_M | 0.7 GB | 1.3 GB | Yes |
| Llama 3.2-1B | BF16 | 2.0 GB | 2.9 GB | Tight |
| Qwen3-1.7B | Q4_K_M | 1.1 GB | 1.8 GB | Yes |

**Key comparison:** BF16 TinyLlama-1.1B (~2.2 GB) at 192 GB/s vs the same model at Q4_K_M on the 2 GB tier at 105 GB/s. Does full precision on a faster card beat aggressive quantisation on a slower one?

---

### 4GB Tier — GTX 1050 Ti

**Usable VRAM:** ~3.5 GB (after OS/driver reservation)
**Benchmark GPU:** GTX 1050 Ti (112 GB/s)

| Model | Precision | File Size | VRAM Est. | Fits? |
|---|---|---|---|---|
| SmolLM2-135M | BF16 | 0.3 GB | 0.9 GB | Yes |
| SmolLM2-360M | BF16 | 0.7 GB | 1.3 GB | Yes |
| TinyLlama-1.1B | BF16 | 2.2 GB | 3.1 GB | Tight |
| Gemma 3-1B | BF16 | 2.0 GB | 2.9 GB | Yes |
| Llama 3.2-1B | BF16 | 2.0 GB | 2.9 GB | Yes |
| Qwen3-1.7B | Q4_K_M | 1.1 GB | 1.8 GB | Yes |
| SmolLM2-1.7B | Q4_K_M | 1.1 GB | 1.8 GB | Yes |
| DeepSeek-R1-1.5B | Q4_K_M | 1.0 GB | 1.7 GB | Yes |
| Llama 3.2-3B | Q4_K_M | 2.0 GB | 2.9 GB | Yes |
| Ministral 3B | Q4_K_M | 1.9 GB | 2.8 GB | Yes |
| Qwen3-4B | Q4_K_M | 2.5 GB | 3.5 GB | Tight |
| Phi-4-Mini (3.8B) | Q4_K_M | 2.4 GB | 3.4 GB | Tight |
| Gemma 3-4B | Q4_K_M | 2.5 GB | 3.5 GB | Tight |

**Key comparison:** BF16 Gemma-3-1B (~2.0 GB) vs Q4_K_M Qwen3-4B (~2.5 GB). Both fit. Does the 4× larger model at reduced precision beat the tiny model at full precision?

---

### 6GB Tier — GTX 1060 6GB

**Usable VRAM:** ~5.3 GB
**Benchmark GPU:** GTX 1060 6GB (192 GB/s)

Everything from the 4GB tier, plus:

| Model | Precision | File Size | VRAM Est. | Fits? |
|---|---|---|---|---|
| Qwen3-1.7B | BF16 | 3.4 GB | 4.6 GB | Yes |
| SmolLM2-1.7B | BF16 | 3.4 GB | 4.6 GB | Yes |
| DeepSeek-R1-1.5B | BF16 | 3.0 GB | 4.1 GB | Yes |
| All 3-4B models | Q5_K_M+ | 2.7-3.2 GB | 3.7-4.3 GB | Yes |
| DeepSeek-R1-7B | Q4_K_M | 4.4 GB | 5.8 GB | Tight |
| DeepSeek-R1-7B | Q3_K_M | 3.5 GB | 4.7 GB | Yes |

**Key comparison:** BF16 SmolLM2-1.7B (~3.4 GB) vs Q4_K_M DeepSeek-R1-7B (~4.4 GB). The 7B distilled reasoning model at aggressive quantization vs the 1.7B "overtrained" model at full precision.

---

### 8GB Tier — RTX 2060 Super

**Usable VRAM:** ~7.2 GB
**Benchmark GPU:** RTX 2060 Super (448 GB/s)

Everything from lower tiers, plus:

| Model | Precision | File Size | VRAM Est. | Fits? |
|---|---|---|---|---|
| Llama 3.2-3B | BF16 | 6.4 GB | 8.2 GB | Tight |
| Ministral 3B | BF16 | 6.0 GB | 7.7 GB | Tight |
| Qwen2.5-Coder-3B | BF16 | 6.0 GB | 7.7 GB | Tight |
| All 3-4B models | Q8_0 | 4.0-4.2 GB | 5.3-5.5 GB | Yes |
| DeepSeek-R1-7B | Q5_K_M | 5.2 GB | 6.7 GB | Yes |
| DeepSeek-R1-7B | Q4_K_M | 4.4 GB | 5.8 GB | Yes (comfortable) |

**Key comparison:** BF16 Llama-3.2-3B (~6.4 GB) vs near-lossless Q8_0 Qwen3-4B (~4.2 GB) vs Q5_K_M DeepSeek-R1-7B (~5.2 GB). Three different size classes, three precision levels, same VRAM budget.

---

### 12GB Tier — RTX 3060

**Usable VRAM:** ~11 GB
**Benchmark GPU:** RTX 3060 (360 GB/s)

Everything from lower tiers, plus:

| Model | Precision | File Size | VRAM Est. | Fits? |
|---|---|---|---|---|
| Phi-4-Mini (3.8B) | BF16 | 7.6 GB | 9.6 GB | Yes |
| Gemma 3-4B | BF16 | 8.0 GB | 10.1 GB | Yes |
| Qwen3-4B | BF16 | 8.0 GB | 10.1 GB | Yes |
| DeepSeek-R1-7B | Q8_0 | 7.4 GB | 9.4 GB | Yes |
| DeepSeek-R1-7B | BF16 | 14.0 GB | 17.3 GB | No |

**Key comparison:** BF16 Qwen3-4B (~8.0 GB) vs Q8_0 DeepSeek-R1-7B (~7.4 GB). Full-precision 4B vs near-lossless 7B. Does the extra 3B parameters at Q8_0 beat full precision?

---

### 16GB Tier — Tesla P100

**Usable VRAM:** ~15 GB
**Benchmark GPU:** Tesla P100 16 GB HBM2 (732 GB/s) — Colmena

The P100 is an unusual card in the lineup -- datacenter Pascal with HBM2 instead of GDDR. No Tensor Cores, but 732 GB/s memory bandwidth is faster than every consumer card in the lab except the RTX 3090. This makes it a compelling inference card for bandwidth-bound LLM workloads, not just a training card.

| Model | Precision | File Size | VRAM Est. | Fits? |
|---|---|---|---|---|
| Qwen3-4B | BF16 | 8.0 GB | 10.1 GB | Yes |
| DeepSeek-R1-7B | BF16 | 14.0 GB | 17.3 GB | Tight |
| DeepSeek-R1-7B | Q8_0 | 7.4 GB | 9.4 GB | Yes (comfortable) |
| Llama 3.1-8B | Q8_0 | 8.5 GB | 10.7 GB | Yes |
| Llama 3.1-8B | Q4_K_M | 4.9 GB | 6.4 GB | Yes |
| Qwen3-8B | Q8_0 | 8.5 GB | 10.7 GB | Yes |
| Llama 3.1-13B | Q4_K_M | 7.9 GB | 10.0 GB | Yes |

**Key comparison:** Q8_0 Llama 3.1-8B on the P100 (732 GB/s) vs the same model on an RTX 4060 Ti 16 GB (288 GB/s, planned). Same VRAM, same precision, 2.5x bandwidth difference. The P100 lacks Tensor Cores; the 4060 Ti has them. Which factor dominates for inference?

---

### Pooled Tiers (Multi-GPU)

These tiers pool VRAM across matched cards via llama.cpp `--tensor-split`. Each pooled tier has a monolithic single-card counterpart for direct comparison. Both GTX (Tortuga) and RTX (Colmena) configurations are tested -- the GTX track provides a no-Tensor-Core control group.

**RTX pooled (Colmena):**

| Tier | Configuration | Usable VRAM | Per-Card BW | Monolithic Counterpart |
|---|---|---|---|---|
| 16 GB (pooled) | 2× RTX 2060 Super | ~14 GB | 448 GB/s each | Tesla P100 16 GB (732 GB/s) |
| 24 GB (pooled) | 3× RTX 2060 Super | ~21 GB | 448 GB/s each | RTX 3090 24 GB (936 GB/s, planned) |

**GTX pooled (Tortuga):**

| Tier | Configuration | Usable VRAM | Per-Card BW | Monolithic Counterpart |
|---|---|---|---|---|
| 12 GB (pooled) | 2× GTX 1060 6 GB | ~10 GB | 192 GB/s each | GTX Titan X 12 GB (336 GB/s) |
| 18 GB (pooled) | 3× GTX 1060 6 GB | ~16 GB | 192 GB/s each | Tesla P100 16 GB (732 GB/s) |

*Tortuga has three GTX 1060 6 GB cards. All pooled configurations are available without additional acquisition.*

**Why this matters:** The RTX 2060 Super's 448 GB/s memory bandwidth exceeds the RTX 3060 (360 GB/s), RTX 4060 (272 GB/s), and RTX 4060 Ti (288 GB/s). Token generation is bandwidth-bound, so pooled 2060 Supers may outperform newer single cards at the same total VRAM -- despite the PCIe splitting overhead.

The GTX pooled track asks a different question: can pre-RTX hardware scale into useful VRAM ranges? 18 GB of pooled GTX 1060 VRAM would run models that no single pre-RTX consumer card can fit. Whether the 192 GB/s per-card bandwidth and PCIe overhead make this practical is an open question.

**Key comparisons:**

| Comparison | What It Tests |
|---|---|
| 2× RTX 2060 Super (16 GB) vs P100 (16 GB) | Pooled Turing vs monolithic Pascal at same VRAM |
| 3× RTX 2060 Super (24 GB) vs RTX 3090 (24 GB) | Pooled commodity vs monolithic high-end |
| 2× GTX 1060 6 GB (12 GB) vs GTX Titan X (12 GB) | Pooled Pascal vs monolithic Maxwell at same VRAM |
| 3× GTX 1060 6 GB (18 GB) vs P100 (16 GB) | More pooled VRAM (no Tensor Cores) vs less monolithic VRAM (no Tensor Cores, but 4x bandwidth) |

The full multi-GPU experiment design is documented in LocoConvoy: [pooling experiment](https://lococonvoy.org/docs/multi-gpu/) and [tiered inference experiment](https://lococonvoy.org/docs/tiered-inference-experiment/).

These tiers will be added once single-card benchmarks are complete and the multi-GPU overhead is quantified.

---

## Which Models at Which Tiers?

Summary view — the maximum parameter count that comfortably fits at each precision level:

| Tier | BF16 (full) | Q8_0 | Q4_K_M | Q2_K |
|---|---|---|---|---|
| 2 GB | — | — | ≤1B | ≤1.7B |
| 3 GB | ≤1B | ≤1B | ≤1.7B | ≤3B |
| 4 GB | ≤1B | ≤1.7B | ≤4B | ≤7B |
| 6 GB | ≤1.7B | ≤3B | ≤7B | ≤7B+ |
| 8 GB | ≤3B | ≤4B | ≤7B | ≤7B+ |
| 12 GB | ≤4B | ≤7B | ≤14B | ≤14B+ |
| 16 GB | ≤7B | ≤8B | ≤14B+ | ≤14B+ |

This is the decision matrix loco-bench produces data for. The question is always: **within your VRAM budget, which combination of model size and precision gives the best results?**
