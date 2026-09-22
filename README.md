# TabPFN-JEPA Plug

Hackathon entry: the first **self-supervised plugin for frozen TabPFN-3.5**.
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

NATICUSdroid-86 (wide), TabPFN-3.5: all86 **0.9890** → JEPA-top24 **0.9512**
vs random-24 0.8689. 28% of features, zero labels, +0.082 over random.

Architecture (top-24 AUC): depth1/0.4 0.9503, depth2/0.4 **0.9704**,
depth2/0.6 0.9539, depth3/0.4 0.9522 — depth-2 sweet spot after fixing the
predictive loss (an earlier detached-MSE run understated training).

S6E9-13 (narrow), honest negatives: raw 0.9364 vs raw+JEPA 0.9341 (parity);
all13 0.9364 vs JEPA-top6 0.8552 vs random-6 0.8540; JEPA-vs-SHAP rank
ρ=0.07. On narrow all-signal data the plug adds nothing — its value is
wide-data triage, and the entry reports both.

## Literature

- T-JEPA: augmentation-free SSL for tabular (2410.05016) — our encoder lineage.
- TabTune (2511.02802): fine-tunes tabular FMs — we keep the FM frozen instead.
- JEPA-Anything (Gen-Verse, arXiv:2609.20800): factorized predictive core framing.
- No published work plugs JEPA-style SSL into TabPFN — that gap is this entry.
