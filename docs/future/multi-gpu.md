---
title: "Multi-GPU Inference on Consumer Hardware"
status: future-spinoff
---

!!! note "Future Spinoff"
    This experiment is tracked as a potential standalone project or as a future vram-bench tier extension (e.g., `--tier pooled-16gb`). Multi-GPU pooling represents a natural extension of the single-card VRAM tier framework.

A proposed experiment to measure the real-world throughput of splitting LLM inference across multiple consumer GPUs, and to determine whether pooled VRAM meaningfully compensates for the bandwidth penalty of PCIe interconnect.

---

## The Question

Consumer GPUs communicate over PCIe, not NVLink. NVLink -- Nvidia's high-speed GPU interconnect -- is reserved for datacenter cards like the A100 and H100, where inter-GPU bandwidth reaches hundreds of GB/s. PCIe 4.0 x16, the best connection available to consumer cards, peaks at around 32 GB/s bidirectional. That is roughly one-tenth the memory bandwidth of a single RTX 3090.

The question is whether that interconnect penalty makes multi-GPU consumer inference practical or pointless -- and for which use cases the answer differs.

**Pooled VRAM is the obvious draw.** Two RTX 3060s give 24 GB of combined VRAM at a total cost well below a single RTX 3090. If inference scales reasonably across them, that is a compelling budget path to larger models. If the PCIe bottleneck throttles throughput badly enough, the pooled VRAM is not worth the complexity.

---

## Why This Is Underexplored

Most multi-GPU inference documentation covers datacenter hardware where NVLink removes the interconnect bottleneck. Consumer multi-GPU benchmarks have historically focused on gaming (SLI/NVLink for rendering), which is a different workload entirely. Systematic measurement of LLM inference across consumer multi-GPU configurations on realistic hardware is sparse.

The community assumption is that consumer multi-GPU is not worth the trouble. That assumption may be correct -- but it has not been rigorously tested across the VRAM tier range that Colmena covers, and the answer likely varies by model size, quantisation level, and which framework is doing the splitting.

---

## Framework Support

Multi-GPU LLM inference on consumer hardware is supported but uneven across frameworks:

**llama.cpp** has tensor-parallel support and can split layers across GPUs. It does not require NVLink and will use PCIe. Performance at PCIe bandwidth has not been systematically documented for consumer cards.

**Ollama** has been adding multi-GPU support progressively. Behaviour varies by version and is less predictable than llama.cpp for experimental configurations.

**vLLM** targets datacenter workloads and assumes NVLink for tensor parallelism. Consumer PCIe configurations are outside its design assumptions.

The most useful experiments for Colmena are llama.cpp-based, where the configuration is explicit and reproducible.

---

## Proposed Experiment Design

### Configuration Matrix

The natural starting point is identical cards in the same machine, where VRAM pooling is the only variable -- compute and bandwidth per card are held constant.

| Configuration | Total VRAM | Per-Card BW | Interconnect |
|---------------|------------|-------------|--------------|
| 1x RTX 2060 Super | 8 GB | 448 GB/s | -- |
| 2x RTX 2060 Super | 16 GB | 448 GB/s each | PCIe 4.0 |
| 1x RTX 3060 | 12 GB | 360 GB/s | -- |
| 2x RTX 3060 | 24 GB | 360 GB/s each | PCIe 4.0 |

Colmena already has three RTX 2060 Supers. Two-card and three-card configurations are available without additional acquisition.

### What to Measure

For each configuration, run vram-bench's standard prompt set and record:

- Token generation rate (t/s) -- primary metric
- Time to first token -- captures the synchronisation overhead at prompt processing
- Throughput at different model sizes -- does splitting help more for larger models?
- Scaling efficiency -- tokens/s per card in multi-GPU vs single-card baseline

### The Scaling Efficiency Number

The useful output is not raw throughput but scaling efficiency: what fraction of linear scaling does the configuration achieve?

A two-card setup with perfect scaling would double single-card throughput. In practice, PCIe overhead will reduce this. If two RTX 2060 Supers achieve 60% of double the single-card rate, that is a meaningful result -- the VRAM pooling costs 40% of theoretical throughput but unlocks model sizes impossible on a single card.

If scaling efficiency falls below 40-50%, the configuration is probably not worth the complexity for users who could instead wait for a single card with sufficient VRAM.

---

## The Asymmetric Case

More interesting than identical-card configurations is mixed-card splitting -- a common real-world scenario where a user has two different cards and wants to use both.

The bandwidth mismatch between cards complicates layer splitting. llama.cpp will attempt to balance layers but the slower card becomes the bottleneck. Measuring this documents how much worse mixed configurations perform compared to matched pairs, which is directly useful to Colmena readers who have accumulated heterogeneous hardware.

A RTX 2060 Super paired with a RTX 3060 is a realistic scenario and testable with Colmena's existing hardware.

---

## The VRAM Pooling Value Proposition

The core question for a reader with two modest cards is: should I run them together for pooled VRAM, or use the better card alone?

The answer depends on the model. For a model that fits comfortably in a single card's VRAM, splitting across two cards adds interconnect overhead for no benefit -- single card will be faster. For a model that does not fit in a single card but fits in the pooled total, splitting is the only option that avoids system RAM offload, which is dramatically slower than either configuration.

The interesting threshold is the model that *just barely* fits in a single card versus the same model running split across two cards with comfortable headroom. Does comfortable headroom on a split configuration outperform tight-fit on a single card? That is a question vram-bench can answer with existing hardware.

---

## Expected Findings (Hypothesis)

The hypothesis is that consumer multi-GPU inference is worthwhile in exactly one scenario: when the target model does not fit on a single card but fits in the pooled VRAM of two cards, and when the alternative is system RAM offload rather than a single larger card.

Outside that scenario -- for models that fit comfortably on a single card -- the PCIe overhead will make multi-GPU configurations slower than single-card, and the complexity will not be justified.

If this holds, the practical recommendation for readers becomes: exhaust single-card VRAM options before reaching for multi-GPU, but if you already have two cards and the model you want exceeds single-card capacity, splitting is worth attempting before giving up on local inference entirely.

---

## Connection to the "Best From What You Have" Philosophy

Multi-GPU inference is the clearest expression of LocoLLM's core question: what can you actually do with the hardware in front of you? A reader with two older cards sitting in a workstation should know whether they can pool those cards effectively, and Colmena's floor-card methodology makes this measurable rather than speculative.

The results will also inform whether multi-GPU deserves its own tier in Colmena's benchmark structure, or whether it is better treated as a footnote to the single-card tiers.

---

*Part of the [vram-bench](https://github.com/michael-borck/vram-bench) project. For hardware context see the [Colmena spec sheet](../colmena.md). Related: [Perceived Intelligence experiment](perceived-intelligence.md), [LocoLLM](https://github.com/michael-borck/loco-llm).*
