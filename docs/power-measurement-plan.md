# Power Measurement Plan

*Producing energy-per-token figures for the Local AI vs Cloud AI TCO model.
Target: usable data by 21 August 2026. Tool: `scripts/measure_power.py`.*

---

## Why This Is Four Measurements, Not Twenty-Four

The fleet holds 28 installed cards. Characterising every one would take weeks
and answer a question nobody asked. The TCO model has three scenarios, and the
lab already owns a machine that represents each of them.

| Scenario | Machine | Why it fits |
|---|---|---|
| Single power user | Hormiga | Low-profile office node, GTX 1050 Ti LP. The constrained-chassis floor case. |
| Department, ~20 people | **Búho** | Dedicated inference, one card one job, always on. This *is* the deployment the TCO argument is about. |
| Institution, ~500 people | Hidra | Open-frame X99, 4x PCIe x16. Measure incrementally at 1, 2, 3 and 4 cards. |
| Dense upper bound | Colmena | 8-GPU rig. Only if the institutional scenario needs a denser point. |

Measuring machines rather than cards is also methodologically correct here.
Wall-plug measurement captures CPU, drives, fans and power supply conversion
losses, all of which a TCO model must carry and none of which appear in a
card-level figure.

---

## Order of Work

### 1. Búho, idle. Do this first.

Highest value, lowest effort, no dependencies. Búho is the always-on dedicated
inference box, so its idle draw multiplied by 8760 hours is very likely the
single largest energy term in the whole model. It is a single-card machine, so
there is nothing to isolate. It needs a smart plug and five minutes, with no
model loaded, no workload and no benchmark.

```bash
python scripts/measure_power.py --plug-url http://<ip> --label buho --idle-only --duration 300
```

Do this before anything else. If the fortnight goes badly, this one number is
still worth more to the students than the other 27 cards combined.

### 2. Búho, under load

Gives energy per token for the departmental archetype.

```bash
python scripts/measure_power.py --plug-url http://<ip> --label buho \
    --model models/Qwen3-4B-Q4_K_M.gguf --ngl 99
```

### 3. Hormiga, idle and load

The cheap office floor case. Same two commands, `--label hormiga`.

### 4. Hidra, incremental

Measure at one, two, three and four cards, labelling each run.

```bash
python scripts/measure_power.py --plug-url http://<ip> --label hidra-2card ...
```

This is the most interesting result of the set. Chassis and power supply
overhead amortises as cards are added, and that curve is the quantitative
backbone of any consolidation argument. As far as we know nobody has published
it, which makes it a contribution rather than just an input.

---

## What to Measure With

A **Shelly plug**, Gen1 or Gen2. It exposes a local HTTP endpoint needing no
cloud account and no credentials, which is what lets the script sample it during
a run. Cloud-only plugs such as the TP-Link Tapo range are not supported and are
not worth the authentication work at this price point.

Without a plug, the script falls back to `nvidia-smi`. That measures the card
alone and understates true wall draw by roughly 20 to 35 percent. Acceptable for
comparing cards against each other, not acceptable as a TCO input, and the
script stamps a warning into the JSON saying so.

---

## The Output That Matters

Cloud AI is priced per token. The local side has to reach the same denominator
before the two can be compared, so the unit is **watt-hours per 1000 tokens**.
Everything else is intermediate working.

The script reports two figures:

- **marginal**, the extra energy to serve another thousand tokens on a machine
  already powered on
- **total**, the same as though the machine existed solely to serve those tokens

Neither includes idle time between requests, because that depends on duty cycle,
which is a modelling assumption rather than a measurement. It belongs in the
students' spreadsheet, not in this data.

---

## Expectations, To Be Tested

Recorded so they can be checked against the results rather than quietly
confirmed.

- Idle draw will dominate annual energy use at departmental utilisation.
- Real draw during generation will sit well below manufacturer TDP, because
  token generation is limited by memory bandwidth rather than computation. If
  so, the picker's TDP figures overstate local energy cost, and the students'
  interim model is conservative in the lab's disfavour.
- Electricity will not be where the crossover is decided. Capital cost and
  utilisation likely dominate.

If the second one holds, it is worth publishing on its own. TDP is widely
misused as a proxy for inference power draw.

---

## Handing Over

Results land as JSON under `results/power/`. The students need three things from
it: idle watts per machine, watt-hours per 1000 tokens, and the Hidra scaling
curve. Send the JSON and a one-paragraph summary rather than raw files alone,
and label clearly which machine maps to which of their three scenarios.
