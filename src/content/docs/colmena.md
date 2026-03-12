---
title: "Colmena: Benchmark Reference Machine"
---

Colmena is the dedicated hardware platform used for all loco-bench benchmarks. Understanding its specifications is essential for interpreting results and extrapolating to your own hardware.

## System Specifications

| Component | Detail |
|---|---|
| **Chassis** | WEIHO 8-GPU enclosed mining rig (72x42x18cm, steel, blue lid) |
| **Motherboard** | Intel LGA1155 socket, likely B75/H61 chipset |
| **CPU** | Intel i3-3220 (Ivy Bridge, dual core) |
| **RAM** | 8GB DDR3 SODIMM (board-confirmed 8GB ceiling) |
| **OS Storage** | 128GB mSATA |
| **Model Storage** | WD Scorpio Blue 750GB SATA (via `OLLAMA_MODELS` env var) |
| **PSU** | Integrated 2000-3300W unit |
| **Cooling** | 4x 120mm fans (to be replaced with Arctic P12 PWM) |
| **GPU Slots** | 8 native PCIe slots (no risers needed) |
| **Form Factor** | Enclosed chassis, not open frame |

## GPU Lineup

| VRAM Tier | GPU | Bandwidth | Architecture | Tensor Cores | Role |
|---|---|---|---|---|---|
| 4GB | GTX 1050 Ti | 112 GB/s | Pascal | No | Floor of 4GB tier |
| 6GB | GTX 1060 6GB | 192 GB/s | Pascal | No | Floor of 6GB tier (pending acquisition) |
| 8GB | RTX 2060 Super | 448 GB/s | Turing | Yes | Floor of 8GB tier |
| 12GB | RTX 3060 AORUS Elite | 360 GB/s | Ampere | Yes | Floor of 12GB tier |
| 24GB | RTX 3090 | 936 GB/s | Ampere | Yes | Reference ceiling (reserved, work budget) |
| -- | 3 slots reserved | -- | -- | -- | Future expansion |

## Philosophy: Deliberately Constrained

Colmena is a deliberately constrained machine. The i3-3220 CPU, 8GB RAM ceiling, and modest storage exist by design, not accident.

The CPU's job is to boot the OS and manage the PCIe bus. The GPUs do the work. Over-speccing the host system would make Colmena a *worse* research instrument -- loco-bench benchmarks GPU capability on modest hardware, which is what most users actually have.

The RAM constraint means sequential rather than fully parallel benchmarking. Results are identical -- same hardware, same models -- the runs just don't happen simultaneously. For CloudCore inference serving, one or two active instances at a time is realistic for student load anyway.

## Why Nvidia Only?

The entire local LLM toolchain -- Ollama, llama.cpp, PyTorch, bitsandbytes, Unsloth -- targets CUDA first. AMD's ROCm stack exists and is improving, but driver support is narrower, community troubleshooting is thinner, and the tooling friction is meaningfully higher. Intel Arc is earlier still. For a lab that needs to work reliably with minimal sysadmin overhead, CUDA is the only practical choice today.

The secondhand market reinforces this. The cryptocurrency mining boom flooded resale channels with Nvidia consumer cards at accessible prices. AMD equivalents at the same VRAM tiers are rarer and less standardised. And the overwhelming majority of users running local LLMs on consumer hardware are on Nvidia -- loco-bench floor cards need to represent what people actually have.

Apple Silicon is the exception, and Poco covers that path via Metal and MLX. If ROCm matures to the point where an AMD card is a genuine drop-in for Ollama inference, it becomes a candidate for a Colmena slot. That day isn't today.

What matters for replication is capability tier, not specific parts. Match the VRAM range and CUDA support, source whatever is available locally at the time.

## Colmena as Reference Baseline

Colmena generates the controlled, repeatable reference results. Community members running the same loco-bench suite on their own hardware extend coverage across GPUs Colmena will never have. See the [Community Contributions](guide#community-contributions) section in the benchmarking guide for how to submit results.

## Benchmark Philosophy: Floor of Tier

Each VRAM tier is represented by the **worst-in-class GPU** for that tier, not the best available. This gives a conservative baseline with a clear promise:

> "If it runs here, it runs on your card."

Community submissions extend each tier upward. The bandwidth delta within each tier is documented in `nvidia-gpu-reference.md`, allowing readers to extrapolate to their specific card.

### Why the RTX 3090?

The 3090 sits in an awkward market position -- too old for enthusiasts, too expensive for budget builders. But for loco-bench it serves as the **reference ceiling** for consumer secondhand hardware:

- 24GB VRAM is the consumer ceiling for secondhand GPUs
- Validates whether floor-tier results scale predictably upward
- 936 GB/s bandwidth provides genuinely interesting comparative data
- Most loco-bench users have 8GB cards or less -- the 3090 result tells them what they're leaving on the table, and in many cases the answer will be "not as much as you'd think"

The 3090 is framed as a research instrument, not an aspirational purchase. Reserved via work research budget with a patient acquisition strategy.
