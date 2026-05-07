---
title: "Advanced Reference: Scaling MoE+RAM Beyond the Floor"
description: For readers who want to push the MoE+RAM thesis up the hardware ladder — variable-bit-weight quantization, 3090-class targets, and consumer-board RAM nuances.
---

LocoBench's primary mission is the floor — the worst card per VRAM tier on hardware most people already own. This page is for readers who want to push the same MoE+RAM thesis higher up the ladder: 3090-class GPUs, 128GB consumer RAM, and the specialised software that makes models in the 500B-parameter range tractable on a desktop.

This is reference, not the focus. If you're starting from scratch, the [Benchmarking Guide](guide) and the LocoBench tier matrix are where you should look first.

## What scales up

The MoE+RAM trick that gets a Qwen3-30B-A3B running on a GTX 1060 6GB scales up almost unchanged. The same two ingredients matter at the higher band:

- **VRAM holds the active model** (the experts that fire per token plus the KV cache).
- **System RAM holds the total model** (the experts at rest).

Drop a 24GB GPU and 128GB system RAM into the equation and you can host MoE models in the 500-billion-parameter range at quantised sizes around 100-120GB on disk. DeepSeek-R1 and DeepSeek-V3 are the obvious targets at this tier.

## What changes at the higher band

Three things matter beyond LocoBench's primary scope:

**Variable-bit-weight quantization.** Standard `llama.cpp` quantizes uniformly across model weights. Specialised forks (notably the **IK llama.cpp** fork, sometimes called Krawall's fork, with the variable-bit-weight feature) preserve higher precision on the weights that carry the most reasoning while compressing the rest. The result: a quantised 500B model that retains more of the original's reasoning depth than a uniformly-quantized version of the same size, and substantially more than a smaller distilled "student" model trained to imitate it. This distinction matters when comparing across the upper tier — a quantized 500B and a distilled 70B "from" 500B are not the same artifact.

**Consumer-board RAM topology.** AM5 and modern Intel desktop boards are dual-channel and have four DIMM slots. Populating all four often forces the memory controller to downclock, throttling the very bandwidth the workload needs. **2x64GB beats 4x32GB** at the same total. (Note: this is a *consumer-board* rule. Server boards like X99 are quad-channel and benefit from populating all four slots — see the LocoBench primary rig spec.)

**Vulkan backend for long contexts.** On long-context inference (32K+), the Vulkan backend in `llama.cpp` has been observed to outperform CUDA on the same Nvidia hardware. Worth trying if you're hitting throughput walls at high context lengths.

## Where this fits

| Tier | Target | Hardware order | LocoBench coverage |
|---|---|---|---|
| **Floor** (LocoBench primary) | Qwen3-30B-A3B at reading speed | GTX 1050 Ti / 1060 6GB + 32-64GB DDR4 server RAM, ~AUD $200 total | Primary [MoE-on-a-budget design](https://github.com/michael-borck/loco-bench/blob/main/docs/superpowers/specs/2026-05-06-locobench-moe-budget-design.md) |
| **Mid** (LocoBench stretch) | Qwen3-Next-80B-A3B | RTX 2060 Super 8GB + 64GB DDR4, ~AUD $400 total | Stretch goal in primary spec |
| **Upper** (this page) | DeepSeek-R1 / DeepSeek-V3 class | RTX 3090 24GB + 128GB DDR5, ~AUD $1500 total | Acknowledged here, not actively benchmarked |

The upper tier is a valid path if you have the hardware. It is not LocoBench's mission. The mission is to characterise the floor honestly, so most people can participate without buying anything.

## Sources

This page distils technical content from two community references on the upper-tier desktop AI setup. The references themselves are framed as digital-sovereignty manifestos; the engineering content survives the framing.

- *The Home-Grown Giant: A Guide to Running Massive AI Models on Your Desktop*
- *The Personal Thinking Machine: A Student's Primer on Local AI Hardware*

If you go up this path, the [LocoLabo umbrella](https://locolabo.org) tracks how the upper-tier work informs the primary mission — particularly under the "engineer before hardware" principle, which holds at every tier.
