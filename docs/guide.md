# Benchmarking Guide

This document covers how to run vram-bench benchmarks, what hardware to use, and how to produce the "bang per bit" analysis that fills a genuine gap in the literature.

## What We're Measuring and Why

Most published benchmarks evaluate full-precision models on cloud hardware. Nobody systematically compares everything that fits within a given VRAM budget — full-precision small models against quantized larger models — on consumer hardware. That's the gap vram-bench fills.

We're running two distinct benchmarks that serve different purposes:

**Benchmark A: "What's best for my VRAM budget?"** Within each VRAM tier (4GB, 6GB, 8GB, 12GB, 24GB), compare every model that fits — whether full-precision or quantized — on standard tasks. This answers which model+precision combination gives the most capability for a given hardware constraint.

**Benchmark B: "What's the real user experience?"** Run the top models per tier on actual target hardware and measure tokens/sec, time-to-first-token, and memory usage alongside quality. This connects quality numbers to deployment reality.

## The Test Matrix

### Models

Start with models that have existing full-precision benchmarks for comparison:

| Model | Parameters | Why Include |
|---|---|---|
| Qwen3-4B-Instruct | 4B | distil labs #1 for fine-tuning |
| Qwen3-1.7B | 1.7B | Smallest viable Qwen; tests scaling |
| Llama 3.2-3B-Instruct | 3.2B | Different architecture; strong baseline |
| Llama 3.2-1B-Instruct | 1B | Tests quantization cliff at small scale |
| Phi-4-Mini (3.8B) | 3.8B | Strong reasoning claims |
| Gemma 3-1B-it | 1B | Different tokenizer |
| Gemma-3-4B-it | 4B | Tests scaling directly against Gemma-3-1B-it |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | Distilled reasoning at micro scale |
| SmolLM2-1.7B | 1.7B | HuggingFace on-device contender |
| Ministral 3B | 3B | Mistral edge-optimized |
| Qwen2.5-Coder-3B | 3B | Domain-specific (coding) baseline |
| Phi-4-Mini-Reasoning | 3.8B | Reasoning distillation comparison |
| DeepSeek-R1-Distill-Qwen-7B | 7B | Does a heavily quantized 7B beat a 4B at Q4_K_M? |
| TinyLlama-1.1B | 1.1B | Community baseline |

### Quantization Levels

For each model, quantize at:

| Quant | Approx bpw | Why Include |
|---|---|---|
| BF16 (baseline) | 16 | Reference point; matches published benchmarks |
| Q8_0 | 8 | Near-lossless baseline |
| Q6_K | 6.6 | High quality, moderate compression |
| Q5_K_M | 5.7 | Often cited as best quality/size balance |
| Q4_K_M | 4.8 | The critical data point for local deployment |
| Q4_0 | 4.0 | Simpler quantization; speed comparison |
| Q3_K_M | 3.4 | Tests where quality collapses |
| Q2_K | 2.6 | Extreme compression; likely broken but worth documenting |

That's 14 models x 8 quant levels = 112 model variants. Each takes 1-3 hours to evaluate depending on hardware, so budget accordingly for the full matrix.

### Tasks

Use the same benchmarks as the Open LLM Leaderboard for direct comparability:

**Standard tasks (via lm-evaluation-harness):**
- MMLU (knowledge)
- HellaSwag (commonsense reasoning)
- GSM8K (math reasoning)
- TruthfulQA (factuality)
- ARC-Challenge (science reasoning)

### Speed and Efficiency Metrics

For each model variant, also record:
- File size on disk (MB)
- Peak RAM usage during inference
- Prompt processing speed (tokens/sec at 512 tokens)
- Generation speed (tokens/sec at 128 tokens)
- Time-to-first-token
- Perplexity on a standard corpus (WikiText-2)

## Tools

### Primary: lm-evaluation-harness

The EleutherAI lm-evaluation-harness is the standard. It's the backend for the HuggingFace Open LLM Leaderboard and directly supports GGUF models.

```bash
# Install
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[hf]"

# Evaluate a GGUF model
# IMPORTANT: Always pass a separate tokenizer to avoid multi-hour hangs
lm_eval --model hf \
  --model_args pretrained=/path/to/gguf_folder,gguf_file=qwen3-4b-q4_k_m.gguf,tokenizer=Qwen/Qwen3-4B-Instruct \
  --tasks hellaswag,mmlu,gsm8k,truthfulqa,arc_challenge \
  --device cuda:0 \
  --batch_size 8 \
  --output_path results/qwen3-4b-q4_k_m/
```

The harness outputs structured JSON with per-task scores, making it straightforward to aggregate results across the full matrix.

### Speed Benchmarks: llama-bench

llama.cpp includes a built-in benchmarking tool that measures prompt processing and generation speed:

```bash
# From llama.cpp build directory
./llama-bench \
  -m /path/to/model.gguf \
  -p 512 \    # prompt length
  -n 128 \    # generation length
  -ngl 0      # 0 for pure CPU, or 99 for full GPU offload
```

Run each model variant twice: once with `-ngl 0` (CPU-only, representing the target deployment) and once with `-ngl 99` (full GPU offload, representing best-case scenario).

### Perplexity: llama-perplexity

```bash
./llama-perplexity -m /path/to/model.gguf -f wikitext-2-raw/wiki.test.raw
```

Perplexity is the fastest sanity check for quantization quality. A sharp perplexity jump between Q4_K_M and Q3_K_M confirms the "quantization cliff" at that model size.

### Quantizing Models

Use llama.cpp's quantize tool or HuggingFace's GGUF-my-repo Space:

```bash
# Local quantization
./llama-quantize /path/to/model-f16.gguf /path/to/model-q4_k_m.gguf Q4_K_M

# Or use HuggingFace's online tool (no local compute needed)
# https://huggingface.co/spaces/ggml-org/gguf-my-repo
```

HuggingFace's GGUF-my-repo Space handles conversion and quantization in the cloud for free. Upload a SafeTensors model, select quantization levels, and it produces GGUF files. This is the easiest path for generating the full quant matrix without tying up local hardware.

## Where to Run

All benchmarks run on [Colmena](colmena.md), a deliberately constrained 8-GPU rig. See the [Colmena spec sheet](colmena.md) for full system details, GPU lineup, and benchmark philosophy.

### Recommended Approach

Run the benchmarks in tiers:

**Tier 1 (do first, minimal hardware):** Perplexity sweep of all variants using llama-perplexity on the RTX 2060. This takes minutes per model and immediately shows where the quantization cliffs are. Produces the core "bang per bit" data.

**Tier 2 (core contribution):** lm-eval-harness on Q4_K_M and BF16 for all models. This is the direct comparison that answers "do full-precision rankings hold at Q4_K_M?"

**Tier 3 (full matrix):** Expand to all 8 quant levels for top models. Shows the full degradation curve.

**Tier 4 (deployment reality):** Speed benchmarks on representative hardware (CPU-only, 8GB RAM). Run llama-bench on the 2060 with `-ngl 0` and on an actual student laptop.

## GPU Tier Methodology

vram-bench benchmarks each VRAM tier using the **worst-in-class GPU** for that tier. This is a deliberate choice: the floor of each tier gives a conservative baseline that's more useful than a cherry-picked best case. The implicit promise is:

> "If it runs here, it runs on your card."

Anyone with any card in that VRAM tier is guaranteed to do at least as well as the benchmark result. This is a stronger, more honest statement than testing the best binned sample at optimal conditions.

Within each tier, the model set includes both **full-precision small models** (e.g., SmolLM2-1.7B at BF16) and **quantized larger models** (e.g., Qwen3-4B at Q4_K_M). If they fit in the same VRAM budget, they compete head-to-head.

### Tier Lineup

| VRAM Tier | Benchmark GPU | Bandwidth | Role |
|---|---|---|---|
| 4GB | GTX 1050 Ti | 112 GB/s | Floor of 4GB tier |
| 6GB | GTX 1060 6GB | 192 GB/s | Floor of 6GB tier |
| 8GB | RTX 2060 Super | 448 GB/s | Floor of 8GB Turing tier (all 8GB Turing cards share ~448-496 GB/s) |
| 12GB | RTX 3060 | 360 GB/s | Floor of 12GB tier |
| 24GB | RTX 3090 | 936 GB/s | Reference ceiling (see below) |

Each card answers two questions simultaneously:

1. **What can this VRAM tier run?** (model compatibility — determined by VRAM capacity)
2. **How fast is the worst case?** (inference speed — determined by memory bandwidth)

### Bandwidth Extrapolation

Because the benchmark uses floor-of-tier hardware, users with faster cards in the same tier can extrapolate upward using known bandwidth ratios. Sample footnotes for each tier:

**6GB tier:** "Benchmarked on a GTX 1060 6GB (192 GB/s). Users with a GTX 1660 Super (336 GB/s) or GTX 1660 Ti (288 GB/s) can expect approximately 50-75% higher token throughput on the same models. VRAM capacity and model compatibility results remain directly applicable regardless of card variant."

**12GB tier:** "Benchmarked on an RTX 3060 (360 GB/s). Users with a 3080 12GB (912 GB/s) can expect significantly higher token throughput on the same models. The 3060 represents the entry point of the 12GB tier — model compatibility results apply to all 12GB cards, inference speed will be considerably higher on higher bandwidth variants."

Bandwidth deltas between cards within a tier are documented in `nvidia-gpu-reference.md`.

### The RTX 3090: Reference Ceiling

The RTX 3090 (24GB, 936 GB/s) doesn't fit the floor-of-tier pattern. It's not a card most vram-bench users will have — it sits in an awkward middle ground where it's too expensive for the "accessible hardware" narrative but too old for the "serious AI workstation" crowd.

For vram-bench it serves a different role: the **reference ceiling** for consumer secondhand hardware.

- 24GB VRAM is the consumer ceiling for secondhand GPUs
- It answers "what does the best affordable card unlock?"
- It validates whether the floor-tier results scale predictably upward
- The bandwidth story at 936 GB/s provides genuinely interesting comparative data against the floor cards

The framing is not "here's what you should buy" but **"here's the ceiling of what's possible on consumer secondhand hardware."** The 3090 becomes the reference point everything else is measured against, not a recommendation.

Most vram-bench users have 8GB cards or less. The 3090 result tells them what they're leaving on the table — and in many cases the answer will be "not as much as you'd think for small models." That's a valuable finding that validates the whole project philosophy: if the gap between a $60 floor card and a $600 ceiling card is modest for 3-4B models at Q4_K_M, it proves these models genuinely run well on budget hardware.

### Benchmarking vs Personal Use

This floor-of-tier philosophy serves **benchmarking** — producing conservative, widely applicable baselines. For **personal inference serving** where tokens per second matters, chasing the best-in-class card for your budget is the right strategy. Different goals, different hardware decisions.

### Factory Overclocks

Cards with factory overclocks are run at reference stock clocks for benchmark consistency. On Linux headless servers, `nvidia-smi` is the cleanest tool for this:

```bash
# Lock to stock reference clocks (example: GTX 1060 6GB)
nvidia-smi --applications-clocks=2002,1506
```

OC performance headroom is noted but not measured. This controls for variables rather than ignoring them.

## The "Bang Per Bit" Visualisation

This is the novel chart. Plot performance against cost (size, bits, RAM) to show the efficiency frontier for small quantized models.

### Chart 1: Accuracy vs Model Size (GB)

X-axis: actual file size on disk (GB). Y-axis: composite benchmark score.

Each model appears as a line with points at each quant level. This shows the Pareto frontier: which model+quant combination gives you the most capability per GB.

The interesting finding will be whether the curves cross: does a Q5_K_M 1.7B model ever beat a Q3_K_M 4B model at the same file size? That would directly inform which model to recommend for 4GB machines.

### Chart 2: Accuracy vs Bits Per Weight

X-axis: bits per weight (from 2.6 for Q2_K to 16 for BF16). Y-axis: benchmark score per task.

One line per model, one panel per task. This shows which tasks are most quantization-sensitive and at what precision level each model breaks.

### Chart 3: The "Free Compute" Ratio

X-axis: model size in GB. Y-axis: tokens per second (CPU-only).

Overlay with iso-quality contours. This shows the trade-off space users actually navigate: "I have 8GB of RAM, how much quality can I get and how fast?"

### Chart 4: Quality Recovery Through Fine-Tuning

Once fine-tuned adapters exist (e.g., from [LocoLLM](https://github.com/michael-borck/loco-llm)):

X-axis: quant level. Y-axis: benchmark score. Two lines per model: base and with adapter.

This shows whether fine-tuning recovers quantization losses. If the adapter lines are flatter than the base lines, it proves that task-specific fine-tuning buffers against quantization degradation. That would be a genuinely novel finding.

## Publishing the Results

### As a HuggingFace Dataset

Upload the raw results as a HuggingFace Dataset. This makes the data reproducible and citable.

```
vram-bench/results
  results/
    qwen3-4b-instruct/
      bf16.json
      q8_0.json
      ...
    llama-3.2-3b-instruct/
      ...
  speed/
    rtx2060_gpu.csv
    rtx2060_cpu.csv
    student_laptop_cpu.csv
  metadata.json
```

### As a HuggingFace Space

Build a Gradio or Streamlit dashboard that:
- Shows the bang-per-bit charts interactively
- Lets users filter by model, quant level, and task
- Links to the raw dataset for reproducibility

### As a Technical Report

The benchmark data frames as:

1. **The gap:** Published benchmarks don't cover quantized 3-4B models systematically
2. **The method:** Standard evaluation applied to a controlled matrix of models x quant levels x tasks x hardware
3. **The findings:** Rankings, quantization cliffs, task sensitivity, efficiency frontiers
4. **The implication:** Which configurations are viable for resource-constrained deployment

## Community Contributions

Colmena generates the reference baseline -- controlled, repeatable, well documented. But the real value of vram-bench grows when the community extends coverage across hardware Colmena will never have.

### Why Community Results Matter

- Results from diverse real-world environments are more credible than a single controlled rig
- The dataset grows to cover GPUs nobody in the lab owns -- 3090s, 4090s, AMD, Apple Silicon, laptop GPUs
- Each submission validates whether Colmena's floor-tier results hold on different hardware
- The project shifts from "one person's benchmark" to "a community instrument with a reference implementation"

### How to Contribute

Run the same vram-bench test suite on your hardware and submit results. The goal is one command to run, one command to submit. The harder it is, the fewer submissions we get.

Results should include:

- GPU model and VRAM
- Driver version and CUDA version
- The standard vram-bench output (lm-eval JSON + llama-bench CSV)
- Any relevant system context (CPU, RAM, OS)

Submission format and tooling are being developed. The design constraint is simplicity -- if it takes more than a few minutes to set up and run, it's too complicated.

### Apple Silicon

Apple Silicon results are particularly interesting comparative data. The same vram-bench suite running on M1/M2/M3 hardware via Metal and MLX produces a direct cross-platform comparison that doesn't exist elsewhere in the literature.

## Estimated Time and Cost

| Tier | What | Hardware | Time | Cost |
|---|---|---|---|---|
| 1 | Perplexity sweep (112 variants) | RTX 2060 | ~8 hours | $0 |
| 2 | Core lm-eval (28 variants) | RTX 2060 + cloud | ~48 hours | $0-25 |
| 3 | Full matrix (112 variants) | Cloud A10 | ~96 hours | $50-100 |
| 4 | Speed benchmarks | RTX 2060 + laptop | ~4 hours | $0 |

## Key References

- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness
- **llama.cpp (quantization + benchmarking)**: https://github.com/ggml-org/llama.cpp
- **HuggingFace GGUF-my-repo**: https://huggingface.co/spaces/ggml-org/gguf-my-repo
- **"Which Quantization Should I Use?" (Llama-3.1-8B study)**: https://arxiv.org/html/2601.14277v1
- **LLM Inference Benchmarking Cheat Sheet**: https://llm-tracker.info/howto/LLM-Inference-Benchmarking-Cheat%E2%80%91Sheet-for-Hardware-Reviewers
