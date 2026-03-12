---
title: "Benchmarking Guide"
---

This document covers how to run LocoBench benchmarks, what hardware to use, and how to produce the "bang per bit" analysis that fills a genuine gap in the literature.

## What We're Measuring and Why

Most published benchmarks evaluate full-precision models on cloud hardware. Nobody systematically compares everything that fits within a given VRAM budget -- full-precision small models against quantized larger models -- on consumer hardware. That's the gap LocoBench fills.

We're running two distinct benchmarks that serve different purposes:

**Benchmark A: "What's best for my VRAM budget?"** Within each VRAM tier (4GB, 6GB, 8GB, 12GB, 24GB), compare every model that fits -- whether full-precision or quantized -- on standard tasks. This answers which model+precision combination gives the most capability for a given hardware constraint.

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

That's 14 models x 8 quant levels = 112 model variants.

### Tasks

Use the same benchmarks as the Open LLM Leaderboard for direct comparability:

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
lm_eval --model hf \
  --model_args pretrained=/path/to/gguf_folder,gguf_file=qwen3-4b-q4_k_m.gguf,tokenizer=Qwen/Qwen3-4B-Instruct \
  --tasks hellaswag,mmlu,gsm8k,truthfulqa,arc_challenge \
  --device cuda:0 \
  --batch_size 8 \
  --output_path results/qwen3-4b-q4_k_m/
```

### Speed Benchmarks: llama-bench

```bash
./llama-bench \
  -m /path/to/model.gguf \
  -p 512 \
  -n 128 \
  -ngl 0  # 0 for pure CPU, or 99 for full GPU offload
```

### Perplexity: llama-perplexity

```bash
./llama-perplexity -m /path/to/model.gguf -f wikitext-2-raw/wiki.test.raw
```

## Where to Run

All benchmarks run on [Colmena](colmena), a deliberately constrained 8-GPU rig.

### Recommended Approach

**Tier 1 (do first):** Perplexity sweep of all variants. Minutes per model, immediately shows quantization cliffs.

**Tier 2 (core contribution):** lm-eval-harness on Q4_K_M and BF16 for all models.

**Tier 3 (full matrix):** Expand to all 8 quant levels for top models.

**Tier 4 (deployment reality):** Speed benchmarks on representative hardware.

## Community Contributions

Run the same LocoBench test suite on your hardware and submit results. The goal is one command to run, one command to submit.

Results should include: GPU model and VRAM, driver and CUDA version, standard LocoBench output (lm-eval JSON + llama-bench CSV), and system context (CPU, RAM, OS).

## Key References

- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness
- **llama.cpp**: https://github.com/ggml-org/llama.cpp
- **HuggingFace GGUF-my-repo**: https://huggingface.co/spaces/ggml-org/gguf-my-repo
