# LocoBench MoE-on-a-Budget — Design Spec

**Status:** Draft, awaiting user review
**Date:** 2026-05-06
**Owner:** Michael Borck
**Repo:** loco-bench
**Related:** [LocoBench README](../../../README.md), upstream LocoLab AI-last thesis

---

## 1. Background

### The seed

A YouTube creator demonstrated running **Qwen3-30B-A3B** (a Mixture-of-Experts model) on an 8-year-old GTX 1060 6GB at **17 tokens/sec with 256K context**, using five `llama.cpp` flags. The "satellite-phone" baseline was 3 tok/s; flag-by-flag they reached "reading speed" with full context. Stable, reproducible, single Docker command.

The five flags:

1. `--n_cpu_moe <N>` — pin every layer's expert blocks to CPU; keep attention/SSM on GPU
2. `--no-mmap` — load entire model into RAM up front; eliminate disk page faults during inference
3. **GPU/CPU split tuning** — sweep `n_cpu_moe` to maximize VRAM utilization without OOM
4. **Turbo Quant KV cache** (`--cache-type-k q4_0 --cache-type-v q3_0`) — asymmetric KV compression exploiting Grouped Query Attention's 8:1 ratio
5. `--mlock` (with container/Docker `IPC_LOCK` capability) — prevent kernel from paging experts back to disk during idle

### The opportunity for LocoBench

LocoBench's central question is: *"I have X GB of VRAM — what's the best model I can run?"* The current matrix tops out at quantized 4B dense models in the 4GB and 6GB tiers. If a **30B-parameter MoE** can be coaxed onto the same cards by spending **system RAM instead of VRAM**, the whole tier matrix needs a rethink.

Two structural insights this spec investigates:

- **VRAM determines how big your *active* model can be.** For Qwen3-30B-A3B that's 3B active params (~1.7 GB at Q4) — comfortably within 4GB and 6GB cards.
- **System RAM determines how big your *total* model can be.** With 64GB RAM, an 8GB GPU can run **Qwen3-Next-80B-A3B**. With 128GB, larger MoEs become candidates until *active* params become the new ceiling.

This is a second axis LocoBench doesn't currently track. The spec adds it.

### Architectural notes that affect methodology

- The article's expert-routing pattern (256 experts total, top-8 per token) **matches Qwen3-30B-A3B**, not Qwen3-Next-80B-A3B (which uses different expert counts). The article's separate claim of "30 of 40 layers are SSM" suggests Qwen3-Next architecture, which is **inconsistent with the 256/8 expert count of standard Qwen3-30B-A3B**. Likely explanations: (a) auto-caption transcription error on the SSM claim, (b) the speaker referenced architectural background from a different model. **The spec resolves this ambiguity by pinning model identity to GGUF SHA-256 before any benchmark run** (see §10 open questions).
- If the model under test does have SSM layers (Qwen3-Next family), this affects parallelism: SSM layers compute one position at a time and cannot be parallelized across speculative-decoding draft windows. The article reports speculative decoding **slowed the setup from 17 → 11 tok/s**. Whether the cause is SSM serialization or pure MoE memory thrash, the negative result still applies and is worth documenting.
- MoE expert routing is content-dependent. Per token, 8 of 256 experts wake up; in a batch of N tokens, up to 8N distinct experts may be needed. This is why CPU-offloaded MoE inference is **memory-bandwidth-bound**, not compute-bound.

---

## 2. Goal & scope

### Primary goal

Determine, for each consumer-GPU VRAM tier from 2GB to 8GB:

1. Whether a small-active MoE model **outperforms the strongest dense model** that fits the same VRAM, on standard quality benchmarks at comparable inference speed.
2. **How much of the article's 5-flag speedup applies** to each tier and each model class (default-vs-optimized speedup curve).
3. Whether the article's **17 tok/s @ 256K context** result on a 1060 6GB **reproduces** under our methodology (validates the setup).
4. Whether **system RAM is the binding constraint** that unlocks larger MoEs on small VRAM cards (the "RAM is the second axis" finding).

### Headline questions

| # | Question | Tier | Why it matters |
|---|---|---|---|
| 1 | Does **OLMoE-1B-7B beat TinyLlama-1.1B** on a 2GB card? | 2GB | Same active compute, 7× stored knowledge — clean test of MoE thesis |
| 2 | Does **Qwen3-30B-A3B fit a GTX 1050 Ti 4GB** with the 5 flags? | 4GB | Article assumed 6GB floor; 4GB would extend the result |
| 3 | Can we **reproduce 17 tok/s on the GTX 1060 6GB**? | 6GB | Methodology validation |
| 4 | Does **24GB → 64GB RAM upgrade unlock Qwen3-Next-80B-A3B** on an RTX 2060 Super 8GB? | 8GB | Establishes RAM-as-second-axis story |
| 5 | What is the **per-flag attribution** of speedup from default to optimized? | All | Lets readers prioritize which flags to apply |

### Non-goals (explicitly out of scope)

- Training, fine-tuning, or LoRA adapter work (that's LocoLLM, not LocoBench)
- Routing or orchestration across models (LocoLLM territory)
- Datacenter GPUs (V100/A100/H100) — V100 included only as a stretch transfer experiment, see §7
- Quantization research beyond Q4_K_M weights and Q4/Q3 KV — no GPTQ/AWQ/EXL2/etc.
- Beating frontier models. The bar is *"useful on hardware you already own."*
- Dual-Xeon NUMA experiments — single-Xeon X99 is the canonical rig. Dual-Xeon noted as optional appendix only.
- Models above 8GB active params — won't fit any tier in scope

### Success criteria

The spec is successful if, after execution, we can publish:

1. A clear **answer per headline question** with measured numbers (not just qualitative claims)
2. A **per-tier results table** showing best MoE vs best dense, on quality and speed
3. A **default-vs-optimized speedup curve** per tier with per-flag attribution
4. A **standalone "five flags" recipe doc** anyone can copy-paste
5. **Reproducibility manifest**: GGUF hashes, exact flags, hardware fingerprint, container image — every run reproducible from `config.yaml` alone
6. The **article's 17 tok/s claim either reproduced or refuted** with explanation

---

## 3. Hardware matrix

### Primary rig (canonical configuration)

- **Motherboard:** **X99M-A** (single-socket, 4 DIMM slots). Chosen over the user's dual-socket X99 board because (a) MoE expert lookup is random-access across hundreds of blocks, so NUMA penalty actively hurts; (b) `numactl`-pinning a dual board to socket 0 just leaves the second CPU idle — no benefit for this workload; (c) 128GB ceiling on X99M-A is sufficient for every model in scope, including the Qwen3-Next-80B-A3B unlock test (~45GB at Q4). The X79 alternative is rejected: DDR3 + 32GB practical cap means it cannot host the 80B-A3B test.
- **CPU:** single Xeon (LGA 2011-3), full memory bandwidth on quad-channel DDR4 from one socket
- **RAM:** DDR4, ECC if board/CPU support it. **1 DIMM per channel** for max bandwidth. RAM matrix per §3 below.
- **PCIe slot:** x16 Gen3, CPU-served (not chipset-routed)
- **OS / runtime:** Linux + Docker, llama.cpp inside container per article's deployment shape
- **Storage:** any SSD; not in scope as a variable for the primary spec (NVMe-as-RAM-extension is future work, see §7)

### GPU swap matrix

The rig is held constant; GPUs are physically swapped between runs. This controls every variable except VRAM/architecture.

| Tier | GPU | Architecture | Year | Role |
|---|---|---|---|---|
| **2GB** | GTX 950 | Maxwell | 2015 | Absolute floor; OLMoE vs TinyLlama showdown |
| **4GB** | GTX 1050 Ti | Pascal | 2016 | Article-architecture continuity; Qwen3-30B-A3B floor test |
| **6GB** | GTX 1060 6GB | Pascal | 2016 | **Article reproduction target: ≥17 tok/s** on Qwen3-30B-A3B optimized |
| **8GB** | RTX 2060 Super | Turing | 2019 | "Sleeper card" — headroom for 30B-A3B; Qwen3-Next-80B-A3B at 64GB RAM |

### RAM matrix

User starts at **32GB (4×8GB)** currently installed and will source **4×16GB = 64GB** to enable the 80B-A3B unlock test. 128GB is conditional on results.

| RAM | Source | Tiers run at this RAM | Target outcome |
|---|---|---|---|
| **32GB** (current) | 4×8GB DDR4 | All four GPU tiers (2GB → 8GB) | Baseline runs for all primary models except Qwen3-Next-80B-A3B |
| **64GB** (source 4×16GB) | swap from 32GB after baseline complete | 8GB tier (RTX 2060 Super) re-run | **Qwen3-Next-80B-A3B unlock test** — the headline RAM-as-second-axis finding |
| **128GB** (stretch, 4×32GB) | conditional purchase | 8GB tier only | Run only if 64GB result with 80B-A3B is compelling enough to justify DIMM purchase. Targets larger MoEs (Phi-3.5-MoE, GRIN-MoE) where active params still fit 8GB |

The 32GB → 64GB → 128GB progression is the **RAM-as-second-axis** experimental story. By holding GPU constant (RTX 2060 Super 8GB) and varying RAM, the spec isolates RAM as the lever that unlocks larger total-model classes.

### Optional sidebars (one-shot data points, not full sweeps)

- **GTX 960 4GB (Maxwell)** — does the 4GB-tier finding hold across architectures? Single optimized run for cross-architecture validation. Cut if results clutter the chart.
- **GTX 980 Ti 6GB (Maxwell)** — Maxwell-at-6GB sidebar. Same purpose. Cut if not useful.
- **Dual-Xeon NUMA penalty quantification** — if the user temporarily moves the GPU and RAM into the dual-Xeon X99 board for one run, this gives a single appendix data point: same model, same GPU, dual-socket without `numactl` pinning vs single-socket on X99M-A. Demonstrates *why* we excluded dual-socket. Strictly optional; skip if it adds calendar drag.

### Cards explicitly excluded from primary scope

GTX 1080, 1660, RTX 2060 (non-Super), RTX 5050, RTX 5060 Ti 16GB. Available for follow-up work; would dilute the low-end narrative if included now.

---

## 4. Model matrix

All weights in **GGUF Q4_K_M** for apples-to-apples comparison. KV cache defaults to **Q8** (lossless); **Q4/Q3 Turbo Quant** is the optimized variant.

### Per-tier lineup

| Tier | MoE candidates | Dense baseline | Headline result the matrix produces |
|---|---|---|---|
| **2GB** | OLMoE-1B-7B (7B total / 1B active) | TinyLlama-1.1B | "Same active compute, 7× knowledge — does MoE specialization beat the dense floor?" |
| **4GB** | Qwen1.5-MoE-A2.7B (14B/2.7B), DeepSeek-V2-Lite (16B/2.4B), **Qwen3-30B-A3B (floor test)** | Qwen3-4B-Instruct | "Three MoEs vs strongest dense — does the article's 30B sneak onto a 1050 Ti?" |
| **6GB** | **Qwen3-30B-A3B (article reproduction)** | Qwen3-4B-Instruct | "17 tok/s replication + 5-flag speedup curve" |
| **8GB** | Qwen3-30B-A3B, **Qwen3-Next-80B-A3B (at 64GB RAM)** | Phi-4-Mini-Instruct | "Sleeper card story: 8GB + 64GB DDR4 runs an 80B model at reading speed" |

### Total distinct models

- **MoE:** OLMoE-1B-7B, Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite, Qwen3-30B-A3B, Qwen3-Next-80B-A3B (5 models)
- **Dense baselines:** TinyLlama-1.1B, Qwen3-4B-Instruct, Phi-4-Mini-Instruct (3 models)
- **Total:** 8 models

### Why these dense baselines

| Tier | Baseline | Reason |
|---|---|---|
| 2GB | TinyLlama-1.1B | The model OLMoE is explicitly trying to dethrone; near-identical active compute |
| 4GB / 6GB | Qwen3-4B-Instruct | Strongest small dense that fits comfortably; same family as the article's MoE so quality differences are attributable to architecture |
| 8GB | Phi-4-Mini-Instruct | Strong reasoning; different model family from Qwen, gives external validity |

### Verification step before benchmarking

Before any benchmark run, confirm exact GGUF source, quantization recipe, and hash for each model. The article's "Qwen 2.5 35B A3B" naming was a transcription error for **Qwen3-30B-A3B**; the model identity must be locked unambiguously to the actual HuggingFace repo and file hash before the spec executes.

### Deferred to follow-up (NOT in this spec)

- Mixtral 8x7B (12.9B active — won't fit 8GB)
- Phi-3.5-MoE-instruct, GRIN-MoE (6.6B active — borderline 8GB; needs own sub-study)
- DeepSeek-V3, Qwen3-Coder-480B-A35B (datacenter MoEs)

---

## 5. Benchmark harness

### Architecture: split inference from evaluation

```
                    +--------------------------------+
                    |  llama.cpp server              |
                    |  (OpenAI-compatible API)       |
                    |  ← config preset sets flags    |
                    +--------------------------------+
                                  ↑
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
+----------------+      +-------------------+     +-------------------+
| lm-eval-harness|      | llama-bench       |     | resource probe    |
| --model        |      | (tok/s, TTFT)     |     | (VRAM/RAM/mlocked)|
| openai-comp    |      |                   |     |                   |
+----------------+      +-------------------+     +-------------------+
   quality              speed                     memory + stability
```

Why this shape: `llama.cpp` flags are the lever the spec is testing. Splitting the server from the evaluator gives us free flag control on the inference side without touching the evaluation code.

### Config presets

The model-side variable. Each (model, tier) cell of the matrix runs every applicable preset.

| Preset | Applies to | Flags |
|---|---|---|
| **default** | All models, all tiers | Stock llama.cpp; auto `-ngl` split, mmap on, no mlock, KV cache Q8 |
| **optimized-moe** | MoE models only | `--n_cpu_moe <tuned per tier>`, `--no-mmap`, `--mlock`, `--cache-type-k q4_0 --cache-type-v q3_0` |
| **optimized-dense** | Dense models, optional | `--no-mmap`, `--mlock`. Skip Turbo Quant KV — gains marginal for dense; flag-on adds risk of quality regression. Apply only when `default` is borderline OOM at high context |

### Per-flag attribution sweeps

For Qwen3-30B-A3B on the 1060 6GB (the article's exact rig), run an **incremental flag sweep** to attribute speedup to each flag independently. This reproduces the article's narrative arc and gives readers per-flag value:

1. baseline (`-ngl 20`, no other flags) → expected ~3 tok/s
2. + `--n_cpu_moe 41` → expected ~10 tok/s
3. + `--no-mmap` → expected ~13.5 tok/s
4. + `--n_cpu_moe 35` (VRAM rebalance, more layers on GPU) → expected ~17 tok/s, context drops to ~64K
5. + Turbo Quant KV (Q4/Q3), with `--n_cpu_moe 36` to fit larger KV → expected 17 tok/s, context restored to 256K
6. + `--mlock` → 17 tok/s, stable through 72h idle

Other tiers run only `default` and `optimized-moe` (full bundle). The full attribution sweep is a 6GB-only artifact that exists to reproduce the article and break out the contribution per flag.

### Multiple benchmarks per (model × tier × preset) cell

#### Quality (via lm-eval-harness against the llama.cpp server)

- **MMLU** — broad knowledge; tests whether expert routing helps factual recall
- **GSM8K** — math reasoning; tests whether sparse activation hurts chain-of-thought
- **HellaSwag** — commonsense; cheap-to-answer baseline expected to favor MoE
- **TruthfulQA** — factuality; catches Turbo-Quant degradation if any
- **ARC-Challenge** — science reasoning
- **NIAH (needle-in-a-haystack)** — long-context retrieval at **1K, 8K, 64K, 256K context**. Validates that Turbo-Quant'd KV cache actually retrieves correctly at 256K, not just runs without OOM

#### Speed (via llama-bench + custom prompts)

- tokens/sec at **1K, 8K, 64K, 256K** context lengths (the curve, not just one number)
- TTFT (time-to-first-token) at each context length

Two prompts to vary: short-question-long-context (NIAH-shaped) vs long-generation (creative writing) — the second stresses generation throughput, the first stresses prompt processing.

#### Memory

- Peak VRAM via `nvidia-smi --query-gpu=memory.used --format=csv --loop`
- Peak system RAM via `/proc/<pid>/status` (`VmRSS`, `VmHWM`)
- Mlocked bytes via `/proc/<pid>/status` (`VmLck`)
- Active-vs-stored ratio (validates MoE thesis empirically)

#### Stability (only on optimized-moe preset, 6GB and 8GB tiers)

- Tok/s at t=0 (warm-up done), then re-measure after **1h, 24h, 72h** idle
- 72h is the article's "day 3 slowdown" timeline
- Pass criterion: <5% degradation at 72h with `--mlock`; degradation expected without it

The user has confirmed rigs can be tied up for 72h tests — these are research machines.

### What we expect to fail (and document)

- **Speculative decoding** on Qwen3-Next-80B-A3B — article reports 17 → 11 tok/s regression due to (a) memory thrash from batched expert lookups, (b) SSM layers preventing draft-window parallelism. Spec includes one negative-result run to confirm and document.
- **Turbo Quant KV on dense models without GQA at 8:1** — may degrade quality; only apply to MoE in primary scope.
- Any flag that "should work" but doesn't, gets a paragraph in the writeup. The article's transparency on speculative-decoding failure is part of why it's credible; we match that standard.

### Reuse vs new code

| Component | Status | Effort |
|---|---|---|
| Existing LocoBench `lm_eval --model hf` flow | Reused for the dense baseline `default` preset path | None |
| llama.cpp server boot wrapper (named preset → flags → wait `/health` → return URL) | New | ~100 lines Python |
| Resource probe (poll nvidia-smi + /proc/status, write JSON) | New | ~50 lines Python |
| Stability runner (boot → warmup → idle → re-measure → log) | New | ~80 lines Python |
| Per-flag attribution sweep driver | New | ~60 lines Python |
| NIAH long-context task | Add as lm-eval task or use existing implementation | Small |
| Result storage in `results/moe-budget/...` | New directory tree, JSON outputs | None code-side, just a layout decision |

Estimated total new code: **~300-400 lines of Python orchestration**. No model-specific code; all model differences live in `config.yaml` per cell.

---

## 6. Output format and docs

### Result schema (per cell of matrix)

```
results/moe-budget/<tier>/<model>/<preset>/
  ├── lm_eval.json            # MMLU, GSM8K, HellaSwag, TruthfulQA, ARC-C scores
  ├── niah.json               # needle-in-a-haystack at 1K, 8K, 64K, 256K
  ├── llama_bench.json        # tok/s and TTFT at each context length
  ├── resources.json          # peak VRAM, peak RAM, mlocked bytes
  ├── stability.json          # tok/s at t=0, +1h, +24h, +72h (where run)
  ├── config.yaml             # exact flags, GGUF hash, hardware fingerprint
  └── run.log                 # llama.cpp server stdout
```

`config.yaml` is load-bearing: every run must be reproducible from this file alone.

Example `config.yaml`:

```yaml
run_id: 2026-05-08-1060-qwen3-30b-a3b-optimized-moe
tier: 6gb
gpu:
  model: GTX 1060
  vram_gb: 6
  pci_slot: PCIe x16 Gen3
host:
  motherboard: X99M-A
  cpu: <Xeon LGA 2011-3 model>
  ram_gb: 32
  ram_dimms: 4x8GB
  ram_channels: 4
  ram_speed_mhz: 2400
  ram_type: DDR4 (ECC if board supports)
  os: Ubuntu 22.04
  kernel: 5.15.x
container:
  image: ghcr.io/ggml-org/llama.cpp:server-cuda-<digest>  # confirm current registry path before runs
  ipc_lock: true
model:
  name: Qwen3-30B-A3B
  hf_repo: Qwen/Qwen3-30B-A3B-GGUF
  filename: qwen3-30b-a3b-q4_k_m.gguf
  sha256: <hash>
preset: optimized-moe
flags:
  ngl: 99
  n_cpu_moe: 35
  no_mmap: true
  mlock: true
  cache_type_k: q4_0
  cache_type_v: q3_0
  ctx_size: 262144
```

### Docs deliverables

- **`docs/moe-on-a-budget.md`** — the primary narrative writeup
- **`docs/the-five-flags.md`** — standalone copy-paste recipe doc
- Updated **`README.md`** tier table — adds "Best MoE that fits" column
- **`docs/assets/`** — speedup curves, quality-vs-size scatters, mlock stability traces

### Narrative structure of `docs/moe-on-a-budget.md`

1. **The thesis** — VRAM holds the active model; system RAM holds the rest
2. **The five flags** — recipe summary, link to standalone doc
3. **Tier walkthroughs** — one section per VRAM tier with the headline finding:
   - 2GB: OLMoE-vs-TinyLlama showdown chart
   - 4GB: "Qwen3-30B-A3B does fit a 1050 Ti"
   - 6GB: Article reproduction + per-flag speedup curve
   - 8GB: Sleeper card + Qwen3-Next-80B-A3B unlock at 64GB RAM
4. **Cross-tier findings** — speedup attribution per flag, RAM-as-second-axis chart
5. **What we tried that didn't work** — speculative decoding regression on SSM/MoE; any other dead ends
6. **Transfer to datacenter** — V100 stretch results if completed
7. **Closing: the AI-last frame** — see §8

---

## 7. Stretch goals and future work

### V100 32GB transfer experiment (stretch, in this spec)

Triggered only **after consumer-tier results are in** and validated.

The V100 is a different physics regime: **HBM2 at ~900 GB/s** vs DDR4 at 38–70 GB/s. CPU-RAM expert offload is much slower relative to GPU memory than on consumer cards, which means smaller MoEs benefit less from `--n_cpu_moe` (they already fit in 32GB VRAM directly). The interesting V100 question is at the **upper tail**: can it run models that don't fit even 32GB VRAM?

| What transfers | What changes |
|---|---|
| The MoE active/stored thesis | Sub-30B MoEs fit fully in VRAM → no `--n_cpu_moe` needed → simpler config |
| Quality benchmarks should be near-identical | Speed will be much higher; default-vs-optimized gap may *narrow* for smaller MoEs |
| Turbo Quant KV trick | Less critical at 32GB VRAM; context fits without compression |
| `--mlock` stability concern | Less important on server-grade host with workload management |

#### V100 RAM upgrade decision rule

- **32GB RAM (current):** baseline. Run all primary models that fit in 32GB VRAM. No CPU offload needed.
- **32GB → 64GB:** justified only if Qwen3-Next-80B-A3B on consumer (8GB+64GB RAM) shows compelling 64GB-unlock results.
- **64GB → 128GB:** justified only if 64GB enables a clearly better model class on V100. Likely candidates: Phi-3.5-MoE, GRIN-MoE, possibly Mixtral 8x22B at heavy quant.

### Future work explicitly NOT in this spec (parking lot)

#### NVMe as RAM extension

When the model exceeds available RAM, the next bottleneck is **NVMe page-in cost**. PCIe 4.0 x4 NVMe ≈ 7 GB/s read; DDR4 quad-channel ≈ 70 GB/s. So NVMe is ~10× slower than RAM — but for very-large MoEs (DeepSeek-V3 at Q4 ≈ 350 GB), it may be the only option short of multi-socket server boards.

Possible directions for a follow-up spec:

- **mmap'd model on NVMe + selective expert pinning** — keep frequently-activated experts in RAM page cache, page cold ones from disk
- **Two-level cache:** GPU VRAM (active 8 experts) ← RAM (warm experts) ← NVMe (cold experts)
- **Expert activation profiling:** which experts fire often enough to deserve RAM residency? Per-task or per-domain expert pinning hints
- **Predictive prefetch:** can we pre-load experts the next token will likely need based on routing patterns?

This is its own research direction. Note as future work, not primary scope.

#### Other parking-lot items

- Multi-GPU MoE sharding (expert parallelism across cheap cards)
- Ahead-of-time expert routing analysis (compile-time hints)
- Quantization of expert weights independently from non-expert weights (asymmetric model quantization)
- Comparison of MoE inference engines: llama.cpp vs vLLM vs SGLang vs TensorRT-LLM

---

## 8. The AI-last frame (closing)

The spec's closing message in `docs/moe-on-a-budget.md`:

> *Before you buy a faster card, set five flags. Before you buy more VRAM, buy more RAM. Before you reach for a bigger model, ask whether a sparse one specialised to your task does the job. The cheapest performance is the performance you didn't need to hire.*

This connects to the broader LocoLab thesis. LocoLLM applies AI-last at the *training* layer — small specialist models routed to tasks, instead of one large generalist. This spec applies AI-last at the *deployment* layer — engineering before hardware, RAM before VRAM, sparsity before scale. Same philosophy, different layer of the stack.

---

## 9. Reproducibility and integrity

- Every run produces a complete `config.yaml` with model hash, container image hash, flag set, hardware fingerprint
- Container images pinned by digest, not tag
- GGUF files pinned by SHA-256, not name
- Random seeds set where applicable; otherwise temperature 0 for quality eval
- Hardware fingerprint includes RAM speed, channel population, kernel version, CUDA driver version
- All raw outputs (lm-eval JSON, llama.cpp server logs) committed alongside processed results
- Negative results (speculative decoding failure, OOM cases) reported as data, not omitted

---

## 10. Open questions and risks

### Open questions

1. **Article's actual model identity:** The 256-experts / top-8 routing in the article matches **Qwen3-30B-A3B**, but the "30 of 40 layers are SSM" claim points to Qwen3-Next architecture, which has different expert counts. Resolve by watching the original video for an on-screen GGUF filename, or by attempting both models and seeing which one reproduces 17 tok/s with the article's flags. Lock identity to a SHA-256 before the rest of the spec executes.
2. **Qwen3-Next-80B-A3B GGUF availability:** Confirm that a Q4_K_M GGUF exists with hybrid SSM-attention layers correctly supported in current llama.cpp. If not, fall back to non-Next 80B-class MoE or document the gap.
3. **NIAH implementation:** Use existing lm-eval task if available, or port from a public long-context-eval harness. Decide before implementation.
4. **GTX 950 quantization edge case:** Maxwell lacks native FP16 throughput. Verify that Q4_K_M GGUF inference on GTX 950 actually runs (some quant kernels assume Pascal+). If it doesn't, drop the 2GB tier or note as unsupported.
5. **Article's exact `--n_cpu_moe` values (41 → 35 → 36):** Confirm via the article's source whether 41 is the layer count for the model under test or includes an off-by-one. Affects sweep range.
6. **llama.cpp container registry:** The image moved from `ghcr.io/ggerganov/llama.cpp` to `ghcr.io/ggml-org/llama.cpp`. Confirm current canonical path and pin a specific digest for all runs.

### Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| GTX 950 doesn't run Q4_K_M MoE inference at all | Medium | Drop 2GB tier or use the GTX 1050 (Pascal) at VRAM-cap if it has 2GB variant |
| Qwen3-Next-80B-A3B doesn't fit 64GB RAM at Q4 (size estimate off) | Low-medium | Pre-flight measure model size; fall back to Q3 quant or skip 80B run |
| Article's 17 tok/s doesn't reproduce | Low | This *is* a finding worth reporting. Investigate llama.cpp version, kernel, RAM speed differences |
| Stability test (72h) reveals memory leak unrelated to mlock | Low | Document; file upstream bug if reproducible |
| 72h × multiple runs serializes the calendar | Medium | Schedule stability runs overnight and on weekends; run quality benchmarks during day |
| MoE GGUF format changes mid-spec (upstream churn) | Low | Pin llama.cpp container image by digest |

---

## 11. Out-of-band followups when this spec lands

After execution and writeup, candidate next specs:

1. **NVMe-as-RAM-extension for >128GB MoE inference** (the parking-lot item)
2. **V100 deep dive** if stretch experiment is compelling
3. **MoE inference engine comparison** (llama.cpp vs vLLM vs SGLang) at fixed hardware
4. **Asymmetric expert-weight quantization** (quant non-expert layers heavier than expert layers, or vice versa)

Not in scope. Listed for visibility.

---

## Appendix A: Glossary

- **A3B notation** — "Active 3 Billion": a model with 3B active parameters per token. Total parameter count given separately (e.g., "30B-A3B" = 30B total, 3B active).
- **MoE (Mixture of Experts)** — model architecture where most weights live in expert blocks; only a subset (typically 8 of 256) are activated per token via a learned routing function.
- **GQA (Grouped Query Attention)** — attention variant sharing KV heads across multiple Q heads. The 8:1 ratio in Qwen3 means keys can take heavier KV-cache compression than values.
- **SSM (State Space Model) layers** — alternative to attention that computes one position at a time, depending on the prior state. Cannot be parallelized across speculative-decoding draft windows.
- **`--n_cpu_moe N`** — llama.cpp flag pinning every layer's expert blocks to CPU for N layers, keeping non-expert (attention/SSM) parts on GPU.
- **`--no-mmap`** — load entire model into RAM at startup; eliminates per-inference disk page faults.
- **`--mlock`** — instruct kernel not to page locked memory back to disk; requires container `IPC_LOCK` capability.
- **Turbo Quant KV** — Google DeepMind 2025 paper applying random rotation + aggressive quantization (Q4 keys / Q3 values) to KV cache with near-lossless quality given GQA structure.
- **NIAH (Needle-in-a-Haystack)** — long-context retrieval benchmark: place a known fact ("the needle") in a long document and ask the model to retrieve it.

## Appendix B: References

- LocoBench README — `loco-bench/README.md` — tier philosophy and existing model matrix
- LocoLLM README — `loco-llm/README.md` — adapter training thesis (this spec is the deployment-layer counterpart)
- Article — YouTube video transcript provided by user during brainstorming; five flags, Qwen3-30B-A3B (likely; identity to be confirmed) on GTX 1060 6GB, 17 tok/s @ 256K context. Add explicit URL when added to the docs writeup.
- Turbo Quant paper — Google DeepMind, 2025 — random rotation + aggressive KV quantization
- Qwen3 model family — HuggingFace `Qwen/Qwen3-*`
- OLMoE — Allen AI, `allenai/OLMoE-1B-7B-0924`
- llama.cpp — `ggerganov/llama.cpp`
- lm-evaluation-harness — `EleutherAI/lm-evaluation-harness`
