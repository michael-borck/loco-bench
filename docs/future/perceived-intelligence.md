---
title: "Perceived Intelligence vs Token Rate"
status: future-spinoff
---

!!! note "Future Spinoff"
    This experiment is tracked as a potential standalone project. It requires human evaluators and a different experimental methodology than vram-bench's automated benchmarks. It will use vram-bench data as input but may spin off into its own repository.

A proposed experiment to test whether users rate a faster small model as smarter than a slower large model when they cannot see what is running behind the interface.

---

## The Question

When a user interacts with a local LLM without knowing the model size, does response quality or response speed dominate their perception of intelligence?

The assumption in the community is that larger models produce better answers and users will prefer them. This is probably true in isolation. But local inference introduces a variable that hosted APIs do not: token generation speed varies significantly by hardware, and slow responses degrade the user experience in ways that may overwhelm any quality advantage a larger model provides.

The question vram-bench is positioned to answer: **at what token rate does a smaller, faster model beat a larger, slower one on perceived intelligence -- even when the larger model's answers are objectively better?**

---

## Why This Matters for LocoLLM

Colmena's hardware tiers produce a natural experiment. The RTX 3060 (12 GB, 360 GB/s) and RTX 4060 Ti (16 GB, 288 GB/s) can both load 13B quantised models that the RTX 2060 Super (8 GB, 448 GB/s) cannot. But the 2060 Super's higher bandwidth means it generates tokens faster on the 7B models it is limited to.

This creates a directly testable scenario across existing Colmena hardware:

| Card | Model | Est. Token Rate | VRAM Headroom |
|------|-------|-----------------|---------------|
| RTX 2060 Super | 7B Q4 | ~35-45 t/s | At ceiling |
| RTX 3060 | 13B Q4 | ~15-20 t/s | Comfortable |
| RTX 4060 Ti 16 GB | 13B Q4 | ~12-16 t/s | Comfortable |

The 2060 Super runs a smaller model faster. The 3060 and 4060 Ti run a larger model slower. If perceived intelligence tracks model size, users should prefer the 13B responses. If it tracks response fluency, they may prefer the 7B at speed.

---

## The Threshold Effect

Token generation speed is not linear in its effect on user experience. There appear to be perceptual thresholds:

- **Above ~25 t/s** -- responses feel fast and natural. Users are unlikely to notice speed as a variable.
- **~10-25 t/s** -- noticeable but tolerable. Users are aware of speed but still engaged with content.
- **Below ~10 t/s** -- responses feel slow. Words visibly trickle. User attention drifts. Perceived quality degrades regardless of actual answer quality.
- **Below ~5 t/s** -- uncomfortable to use. Most users will attribute this to the system being poor, regardless of what it produces.

The 13B models on the 3060 and 4060 Ti are likely to sit in the 10-20 t/s range on typical inference hardware -- right in the zone where speed is perceptible but quality differences between 7B and 13B should also be apparent. This makes the crossover point testable rather than theoretical.

---

## Proposed Experiment Design

### Setup

- Same prompt set delivered to each card configuration
- Responses generated blind -- the evaluation interface shows only the response text and speed, not the model or hardware
- Human raters asked to score each response on perceived quality and perceived intelligence, with no knowledge of what produced it
- Raters also asked: "Would you use this system for real tasks?"

### Variables

**Independent:** model size (7B vs 13B), token rate (derived from hardware tier)

**Dependent:** perceived intelligence rating, perceived quality rating, stated willingness to use

### Controls

- Identical quantisation level (Q4_K_M or equivalent) across model sizes
- Same prompt set, covering factual recall, reasoning, summarisation, and creative tasks
- Prompts chosen to produce responses where 13B models are expected to be meaningfully better, not just marginally so

### What to Look For

- The token rate at which users stop preferring the larger model
- Whether the crossover point varies by task type -- reasoning tasks may favour quality more than factual recall
- Whether users who are told the model size after rating change their stated preference

---

## Why vram-bench Data Is the Right Foundation

vram-bench already captures token generation rates across Colmena's hardware tiers. Extending it to include a blind user evaluation layer adds the perceptual dimension that pure throughput benchmarks cannot capture.

The existing hardware spread -- from 112 GB/s (GTX 1050 Ti) to 936 GB/s (RTX 3090) -- produces a wide enough token rate range that the crossover point, if it exists, is likely to appear somewhere in the dataset. No additional hardware acquisition is needed to run the initial experiment.

---

## Expected Findings (Hypothesis)

The hypothesis is that perceived intelligence tracks token rate more strongly than model size in the 10-25 t/s range, and that this effect is stronger for conversational and factual tasks than for complex reasoning tasks where the quality gap between 7B and 13B is large enough to override the speed disadvantage.

If this holds, it has practical implications for hardware recommendations. A card that runs a 7B model at 40 t/s may produce a better user experience than a card that runs a 13B model at 12 t/s -- even if the 13B answers are objectively more accurate. That is a result worth documenting.

---

*Part of the [vram-bench](https://github.com/michael-borck/vram-bench) project. For hardware context see the [Colmena spec sheet](../colmena.md). Related: [LocoLLM](https://github.com/michael-borck/loco-llm).*
