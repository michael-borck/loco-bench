# The Quality Cliff

Most inference guides recommend a "4 GB minimum" as received wisdom. loco-bench will produce the evidence behind that claim by running a fixed evaluation set across every tier in the stack.

## Methodology

The same 15 prompts -- spanning reasoning, factual recall, instruction following, and multi-step tasks -- will be run at every VRAM tier using the best-fit model for that tier at Q4 quantisation. Results will be scored for coherence and accuracy and published as a comparable chart.

The tier stack deliberately extends down to 2 GB. This is not because 2 GB is a useful inference target -- it isn't -- but because documenting where quality degrades and why is a research output in its own right. Most inference guides assert a "4 GB minimum" without evidence. loco-bench will show the data behind that claim.

## Expected Results

| VRAM | Model | Expected Quality |
|------|-------|-----------------|
| 2 GB | TinyLlama 1.1B Q4 | Functional for simple tasks. Reasoning gaps. Narrow use cases. |
| 3 GB | Phi-2 2.7B Q4 | Noticeably better. Still limited on complex reasoning. |
| 4 GB | Phi-3 Mini 3.8B Q4 | Meaningful step up. Usable for most everyday tasks. |
| 6 GB | Qwen2 4B Q4 | Solid general capability. |
| 8 GB | Llama3 8B Q4 | Comfortable general purpose. Most users stop here. |
| 12 GB | Llama3 13B Q4 | Clear quality improvement. Longer coherent context. |
| 24 GB | Llama3 70B Q4 | Ceiling of consumer hardware usefulness. |

The goal is not to confirm the obvious -- it is to show *where* the cliff is steep and where it is gradual. The 2 GB to 3 GB jump and the 3 GB to 4 GB jump may tell very different stories. That granularity is what makes running the full tier stack worthwhile.

## Connection to Adapter Training

This data is also relevant to adapter-trained model evaluation. If an adapter-trained 1.1B model outperforms the base Phi-2 2.7B on domain tasks despite running on a 2 GB card, that is a meaningful finding about what adapter training can recover at the quality floor.

## Floor Card Principle

Each tier is benchmarked on the worst-in-class GPU for that VRAM level. If it runs here, it runs on your card. Conservative baselines surface optimisations that comfortable hardware conceals.

The bandwidth delta within each tier allows readers to extrapolate to their specific card. For the current card assignments and tier coverage see the [GPU Inventory](https://locolabo.org/docs/gpu-inventory) on the LocoLab site.
