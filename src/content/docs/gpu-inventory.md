---
title: "GPU Inventory"
---

Every GPU currently in the LocoLab project, organised by machine. Cards move between machines as experiments require -- this is the current assignment.

For specifications, acquisition guidance, and generation-level analysis see the [Nvidia GPU Reference](nvidia-gpu-reference).

---

## Colmena

The RTX-era benchmark platform. An 8-GPU enclosed mining rig running LocoBench. Cards here are chosen as **floor representatives** -- the worst card per VRAM tier, so results are honest baselines. Matched trios give repeat-measurement discipline at their tier.

| Card | VRAM | Bandwidth | Tier Role |
|------|------|-----------|-----------|
| GTX 1060 6 GB (x3) | 6 GB | 192 GB/s | Floor of 6 GB tier (Pascal, no Tensor Cores); bridges into Tortuga's pre-RTX coverage |
| RTX 2060 Super (x3) | 8 GB | 448 GB/s | Floor of 8 GB Turing tier (Tensor Cores). Three cards for result-consistency validation |
| RTX 4060 Ti | 16 GB | 288 GB/s | Floor of 16 GB consumer tier -- documents the memory-bus penalty |
| RTX 5050 | 8 GB | 320 GB/s | Blackwell floor of 8 GB tier (GDDR6, FP4 support). Isolates architecture effect vs bandwidth deficit |

---

## Tortuga

Swappable bench cards. Cards rotate through Tortuga to fill out LocoBench tier coverage, particularly the older Maxwell and Pascal generations.

| Card | VRAM | Bandwidth | Tier Role |
|------|------|-----------|-----------|
| GTX 950 | 2 GB | 105 GB/s | Floor of 2 GB tier (Maxwell). TinyLlama 1.1B only. Quality cliff reference point |
| GTX 960 | 4 GB | 112 GB/s | Floor of 4 GB tier (Maxwell, Compute 5.2, Ollama only) |
| GTX 980 Ti | 6 GB | 336 GB/s | Floor of 6 GB Maxwell tier (Ollama only) |
| GTX Titan X | 12 GB | 336 GB/s | Floor of 12 GB Maxwell tier (Ollama only) |
| GTX 750 Ti | 2 GB | 86 GB/s | Floor of 2 GB Maxwell tier. TinyLlama 1.1B only. Pre-Pascal baseline |
| GTX 1050 Ti | 4 GB | 112 GB/s | Floor of 4 GB tier (Pascal, no Tensor Cores) |
| GTX 1060 3 GB | 3 GB | 192 GB/s | Floor of 3 GB tier (Pascal). Quality cliff reference point |
| GTX 1060 6 GB | 6 GB | 192 GB/s | Additional 6 GB Pascal card |

---

## Hidra

Full-bandwidth multi-GPU research, server GPU benchmarking, and GPU onboarding. X99 dual-Xeon platform with 4x PCIe x16 slots in an open-frame chassis for rapid card swaps. The server GPUs are the primary LocoBench payload at their native VRAM tiers; consumer cards rotate through for onboarding and fill-in benchmarks.

**Server GPUs:**

| Card | VRAM | Bandwidth | Tier Role |
|------|------|-----------|-----------|
| Tesla V100 | 16 GB | 900 GB/s | 16 GB server tier (Volta, HBM2, Tensor Cores) |
| Tesla P100 | 16 GB | 732 GB/s | 16 GB server tier (Pascal, HBM2, no Tensor Cores) |
| Tesla M40 | 24 GB | 288 GB/s | 24 GB server floor (Maxwell, CC 5.2, Ollama only). **Incoming** |
| Tesla P40 | 24 GB | 346 GB/s | 24 GB server tier (Pascal, full modern stack). **Incoming** |
| Tesla M4 | 8 GB | 88 GB/s | 8 GB server floor (Maxwell, CC 5.2, Ollama only). **Incoming** |
| Tesla P4 | 8 GB | 192 GB/s | 8 GB server tier (Pascal, CC 6.1, full modern stack). **Incoming** |

**Consumer rotation (onboarding + small-card benchmarks):**

| Card | VRAM | Bandwidth | Tier Role |
|------|------|-----------|-----------|
| GTX 1070 | 8 GB | 256 GB/s | 8 GB Pascal (no Tensor Cores). Bandwidth comparison against 2060 Super |
| RTX 3050 | 8 GB | 224 GB/s | 8 GB Ampere. Lowest bandwidth Tensor Core card in the 8 GB tier |
| RTX 3060 AORUS Elite | 12 GB | 360 GB/s | Floor of 12 GB tier (Ampere, Tensor Cores) |

---

## Puente

Student-facing host. Ryzen 5 2600 desktop running the "closing the gap" minimal LocoPuente PoC and most LocoEnsayo rehearsal chatbots on a single card.

| Card | VRAM | Bandwidth | Tier Role |
|------|------|-----------|-----------|
| RTX 3090 | 24 GB | 936 GB/s | Sole card for LocoPuente + LocoEnsayo (LLM, image gen, voice, chatbots) |

---

## Búho

Dedicated AI inference and LocoLLM adapter training. Single-Xeon X99 workstation with 32 GB DDR4. One card, one job -- patient and nocturnal.

| Card | VRAM | Bandwidth | Tier Role |
|------|------|-----------|-----------|
| Tesla V100 | 32 GB | 900 GB/s | 32 GB server tier (Volta, HBM2, Tensor Cores). Primary adapter-training card; single-card inference at 32 GB |

---

## Hormiga

Low-profile / office deployment.

| Card | VRAM | Bandwidth | Tier Role |
|------|------|-----------|-----------|
| GTX 1050 Ti LP | 4 GB | 112 GB/s | Low-profile 4 GB card for constrained chassis |

---

## Unassigned

| Card | VRAM | Bandwidth | Notes |
|------|------|-----------|-------|
| GTX 1650 OC LP | 4 GB | 128 GB/s | Turing, no Tensor Cores. Low-profile. No current assignment |
| RTX 2060 Super (x2) | 8 GB | 448 GB/s | Awaiting assignment |
| RTX 5070 Ti | 16 GB | 896 GB/s | Blackwell, GDDR7. Awaiting assignment |

---

## Summary

| Machine | Cards | Primary Role |
|---------|-------|-------------|
| Colmena | 8 | RTX-era LocoBench consumer tier benchmarking |
| Tortuga | 8 | Swappable tier cards for pre-RTX LocoBench coverage |
| Hidra | 5 | LocoConvoy multi-GPU + server GPU benchmarking + onboarding (2 server cards installed, 4 incoming) |
| Puente | 1 | LocoPuente PoC + LocoEnsayo chatbots (RTX 3090 24 GB) |
| Búho | 1 | Dedicated LocoLLM adapter training and single-card inference |
| Hormiga | 1 | SFF floor node / office deployment |
| Unassigned | 4 | — |
| **Total installed** | **28** | (+ 4 incoming server cards for Hidra) |

*RTX 5060 Ti: not confirmed in fleet — omitted pending verification.*

---

*For GPU generation details and acquisition guidance see [Nvidia GPU Reference](nvidia-gpu-reference). For benchmarking methodology see the loco-bench documentation.*
