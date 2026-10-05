# Continuation Prompt — Adaptive Granularity WM

You are continuing the research project in:

`C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment`

Read these files before doing anything:

1. `MASTER_PLAN.md` — follow the **Operational Master Plan v2** at the top; the older schedule is historical.
2. `STATUS.md`
3. `PROGRESS.md`
4. `CLAUDE.md`
5. This file.

## Research objective

Turn the proposal into a defensible paper on **uncertainty-guided runtime switching between Causal-rCM c1-1 fine generation (K=1) and c3-3 coarse generation (K=3)**, with recovery and hysteresis. The main questions are uncertainty-vs-baselines, adaptive-vs-fixed quality/latency/compute, and recovery stability.

The verified checkpoints are text-to-video. Set `lambda_action = 0`. Do not claim action-conditioned control. A text-button prompt modifier is optional UI only and is not scientific action conditioning.

## Important links

- Proposal: `C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\03_PROPOSALS\proposal.tex`
- GitHub repository: https://github.com/j1s4nn/adaptive-granularity-wm
- Causal-rCM: https://github.com/NVlabs/rcm
- Kaggle checkpoint dataset: https://www.kaggle.com/datasets/ajjisan/causal-rcm-ckpts
- Kaggle T5 dataset: https://www.kaggle.com/datasets/ajjisan/wan21-t5
- Kaggle DiT/VAE dataset: https://www.kaggle.com/datasets/ajjisan/wan21-dit-vae
- Kaggle account: `ajjisan`

Kaggle authentication is already configured locally. Never read, print, copy, or commit the API token. Never download model files to this laptop. Use Kaggle T4 jobs and keep local results small.

## Current verified facts

- Phase 0b passed: c1-1_step2 and c3-3_step4 load on T4.
- Phase 0c passed: real fp16/SDPA inference works using the documented recipe in `STATUS.md`.
- c1-1 1-step uses c1-1_step2 with `[4, 1]` steps per chunk.
- c3-3 step-4 requires valid latent/chunk-aligned frame counts.
- No EMA weights were found; do not implement or claim `u_ema`.
- The reusable `src` inference and metric layers still contain placeholders until Phase 1 fixes them.

## Phase protocol

Work one phase at a time. At the end of every phase:

1. Save real artifacts and logs.
2. Update `STATUS.md`, `PROGRESS.md`, and this file.
3. State which proposal RQs are covered and which deviations occurred.
4. Run an audit and stop. Report the next phase before continuing.

Never submit a job containing random metrics, dummy frames, repeated placeholder prompts, or unmeasured compute. Include controller, checkpoint-switch, VAE, and decoding overhead in timing. Split calibration and held-out test prompts.

## Phase state

Current phase: **Phase 1 — infrastructure correction and c1-1 → c3-3 → c1-1 smoke test**

Next required evidence:

- real runner derived from `jobs/phase0c/phase0c_infer.py`;
- nested Kaggle mount discovery;
- resumable job metadata and output;
- one valid runtime switch sequence or a documented switch failure;
- measured switching overhead and context/latent alignment status.

When Phase 1 is complete, replace the current phase with Phase 2 and clear only the completed task list. Preserve the audit result and links to artifacts.

## GitHub workflow

The repository name `adaptive-granularity-wm` is professional and should be kept. Commit one coherent phase at a time with a descriptive message, then push only after local validation. Do not force-push or recreate the repository.

