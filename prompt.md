# Continuation Prompt 鈥?Adaptive Granularity WM

You are continuing the research project in:

`C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment`

Read these files before doing anything:

1. `MASTER_PLAN.md` 鈥?follow the **Operational Master Plan v2** at the top; the older schedule is historical.
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

## Completed Phase 1a/1b audit

Phase 1a/1b completed on 2026-10-05. Kaggle kernel `ajjisan/phase-1-runtime-switch-smoke-test-v2` passed a real c1-1 -> c3-3 -> c1-1 sequence on Tesla T4. Metadata: `results/phase1_switch_v2_dl/results/phase1_switch/20261005_031931/20261005_031931_meta.json`. Boundary latent diagnostics are 0.0 in both directions. KV-cache persistence is unavailable and must not be claimed; model loading and switching cost are measured.

Proposal coverage: runtime switching feasibility and speed accounting are covered; uncertainty selection, fixed-mode comparisons, recovery/hysteresis, long-horizon behavior, and paper metrics remain pending. Action conditioning remains deferred with lambda_action=0.

## Phase state

## Completed Phase 2 audit

Phase 2 completed on 2026-10-05: 12/12 real fixed-mode generations passed on Kaggle Tesla T4 (2 prompts × 1 seed × 21/33/45 frames × c1-1/c3-3). Metadata: `results/phase2_baseline/results/phase2_baseline/20261005_042507/20261005_042507_meta.json`; summaries are in the same folder. Mean DiT sampling: c1-1 10.217 s, c3-3 24.027 s. Mean DiT+VAE time was 84.260 s vs 100.550 s (19.3% lower for c1-1, excluding model loading and prompt encoding). Mean recorded NFE was 12 for each mode across the three horizons. Mean decoded adjacent-pixel L1: 0.023899 and 0.024206; interpret as motion/content-dependent variation only, not perceptual quality. VAE dominates runtime. Per-condition timing excludes checkpoint load and prompt encoding; do not claim full end-to-end latency until those are added. Two prompts and one seed cannot resolve the 0.1% target. The hard-coded pass-count bug is documented in `phase2_audit.json`; the raw Kaggle metadata remains unchanged. Source verdict logic is corrected locally but has not been rerun on Kaggle.

Proposal coverage: fixed-mode fine/coarse pilot and temporal variation captured; quality superiority not established; adaptive uncertainty is Phase 3; recovery/hysteresis is later; action remains deferred (`lambda_action=0`).

Current phase: **Phase 3 - signal calibration**

Next required evidence:

- calibrate residual/denoising-discrepancy and warp signals on designated prompts;
- include motion and complexity baselines where feasible;
- keep calibration separate from held-out evaluation;
- report signal distributions and uncertainty ranking without overclaiming quality gains.

Phase 2 is complete at exploratory pilot scale. Begin Phase 3 after this audit; preserve this audit and artifact links.

## GitHub workflow

The repository name `adaptive-granularity-wm` is professional and should be kept. Commit one coherent phase at a time with a descriptive message, then push only after local validation. Do not force-push or recreate the repository.





