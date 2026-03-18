---
title: "Benchmark Hardware"
---

LocoBench runs across Tortuga (pre-RTX legacy tiers) and Hormiga (SFF reference node). Colmena's primary role has shifted to LocoConvoy multi-GPU experiments and LocoLLM fine-tuning, though its GPU specifications are documented here for cross-reference. Mesa serves as the overflow and GPU onboarding platform. Understanding the specifications is essential for interpreting results and extrapolating to your own hardware.

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

## Colmena GPU Lineup

Two matched trios for multi-GPU experiments, plus the Tesla P100 for fine-tuning. Colmena's primary role is now LocoConvoy/LocoLLM, not benchmarking.

| VRAM Tier | GPU | Bandwidth | Architecture | Tensor Cores | Role |
|---|---|---|---|---|---|
| 6 GB | GTX 1060 6 GB x3 | 192 GB/s | Pascal | No | Multi-GPU pooling (18 GB); pre-RTX scaling baseline |
| 8 GB | RTX 2060 Super x3 | 448 GB/s | Turing | Yes | Multi-GPU pooling (24 GB); RTX-era scaling testbed |
| 16 GB | Tesla P100 | 732 GB/s | Pascal | No | Fine-tuning; HBM2 bandwidth (arriving) |

## Tortuga GPU Lineup

Pre-RTX cards without Tensor Cores. Powered on for benchmark runs only.

| VRAM Tier | GPU | Bandwidth | Architecture | Tensor Cores | Role |
|---|---|---|---|---|---|
| 2 GB | GTX 950 | 105 GB/s | Maxwell | No | Absolute floor |
| 4 GB | GTX 960 | 112 GB/s | Maxwell | No | Maxwell 4 GB tier |
| 4 GB | GTX 1050 Ti | 112 GB/s | Pascal | No | Pascal 4 GB tier; cross-ref with Hormiga |
| 3 GB | GTX 1060 3 GB | 192 GB/s | Pascal | No | Unusual tier between 2 GB and 4 GB |
| 6 GB | GTX 1060 6 GB | 192 GB/s | Pascal | No | Floor of 6 GB tier |
| 6 GB | GTX 980 Ti | 336 GB/s | Maxwell | No | Legacy high-end; bandwidth outlier |
| 12 GB | GTX Titan X | 336 GB/s | Maxwell | No | Maxwell 12 GB; counterpoint to RTX 3060 |

## Hormiga

| VRAM Tier | GPU | Bandwidth | Architecture | Tensor Cores | Role |
|---|---|---|---|---|---|
| 4 GB | GTX 1050 Ti LP | 112 GB/s | Pascal | No | SFF reference; minimum viable inference |

## Philosophy: Deliberately Constrained

Both chassis are deliberately constrained machines. Colmena's i3-3220 CPU, 8 GB RAM ceiling, and modest storage exist by design, not accident. Tortuga is similar.

The CPU's job is to boot the OS and manage the PCIe bus. The GPUs do the work. Over-speccing the host system would make the benchmarks less representative -- LocoBench measures GPU capability on modest hardware, which is what most users actually have.

The RAM constraint means sequential rather than fully parallel benchmarking. Results are identical -- same hardware, same models -- the runs just don't happen simultaneously.

## Why Nvidia Only?

The entire local LLM toolchain -- Ollama, llama.cpp, PyTorch, bitsandbytes, Unsloth -- targets CUDA first. AMD's ROCm stack exists and is improving, but driver support is narrower, community troubleshooting is thinner, and the tooling friction is meaningfully higher. Intel Arc is earlier still. For a lab that needs to work reliably with minimal sysadmin overhead, CUDA is the only practical choice today.

The secondhand market reinforces this. The cryptocurrency mining boom flooded resale channels with Nvidia consumer cards at accessible prices. AMD equivalents at the same VRAM tiers are rarer and less standardised. And the overwhelming majority of users running local LLMs on consumer hardware are on Nvidia -- loco-bench floor cards need to represent what people actually have.

Apple Silicon is the exception, and Poco covers that path via Metal and MLX. If ROCm matures to the point where an AMD card is a genuine drop-in for Ollama inference, it becomes a candidate for a Colmena slot. That day isn't today.

What matters for replication is capability tier, not specific parts. Match the VRAM range and CUDA support, source whatever is available locally at the time.

## Reference Baselines

Colmena and Tortuga together generate the controlled, repeatable reference results -- RTX-era tiers from Colmena, pre-RTX tiers from Tortuga, SFF validation from Hormiga. Community members running the same LocoBench suite on their own hardware extend coverage across GPUs the lab will never have. See the [Community Contributions](guide#community-contributions) section in the benchmarking guide for how to submit results.

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
