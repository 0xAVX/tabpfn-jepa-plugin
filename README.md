# TabPFN-JEPA Plug

> **Status: prototype merged into [jepa-pfn](https://github.com/0xAVX/jepa-pfn)
> — not a separate submission.** The encoder, relevance ranking and selection
> results live on there with fixed training and a seed-protocol evaluation.

Prototype: the first **self-supervised plugin for frozen TabPFN-3.5**.
A tiny per-dataset JEPA encoder (mask-predict in latent space, no labels, no
augmentations — after T-JEPA, Thimonier et al. 2024; predictive-core framing
after Gen-Verse/JEPA-Anything) gives TabPFN two things it lacks:

1. **select()** — label-free feature relevance → fit wide data into TabPFN's
   width limits without random subsampling.
2. **embed()** — JEPA latents as extra TabPFN inputs (parity check).

## Reproduce

```bash
<venv-python> scripts/run.py   # needs ../playground-series-s6e9/data/*.csv + OpenML
```

Trains the plug self-supervised (minutes, GPU/CPU), ranks features, and runs
TabPFN-3.5 ablations. Results → `figs/jepa.csv`, `figs/arch.csv`.

## Results

NATICUSdroid-86 (wide), TabPFN-3.5: all86 **0.9890** → JEPA-top24 **0.9540**
vs random-24 0.8689. 28% of features, zero labels, +0.085 over random.
Protocol is inductive: JEPA fit + relevance on X-train only, columns frozen,
evaluated on unseen X-test (no transductive peeking).

Architecture (top-24 AUC): depth1/0.4 0.9543, depth2/0.4 0.9518,
depth2/0.6 0.9550, depth3/0.4 0.9456 — shallow is enough here; mask ratio
secondary. (An earlier detached-MSE run overstated depth-3.)

S6E9-13 (narrow), honest negatives: raw 0.9364 vs raw+JEPA 0.9341 (parity);
all13 0.9364 vs JEPA-top6 0.6762 vs random-6 0.8540 (ranking unstable across
runs: 0.82/0.86/0.68 — noise, not signal); JEPA-vs-SHAP rank
ρ=0.07. On narrow all-signal data the plug adds nothing — its value is
wide-data triage, and the entry reports both.

## Literature

- T-JEPA: augmentation-free SSL for tabular (2410.05016) — our encoder lineage.
- TabTune (2511.02802): fine-tunes tabular FMs — we keep the FM frozen instead.
- JEPA-Anything (Gen-Verse, arXiv:2609.20800): factorized predictive core framing.
- No published work plugs JEPA-style SSL into TabPFN — that gap is this entry.
