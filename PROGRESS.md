# PROGRESS (corrected — tracks reality, not the optimistic Day-1 plan)

## Phase 0a: environment + code probe — DONE (today)
- Kaggle CLI 2.2.4 installed locally, authenticated as ajjisan (KGAT access_token)
- Dataset contents verified remotely (see STATUS.md)
- Fixed: src bundling, recursive checkpoint paths, torch.load weights_only fallback,
  push_and_wait.py CLI 2.x compatibility

## Phase 0b: checkpoint + VRAM probe — RUNNING ON KAGGLE
- Kernel: `ajjisan/phase-0b-checkpoint-vram-probe-t4` (pushed 16:15, RUNNING at 16:55)
- Probe loads c1-1_step2 / c1-1_step4 / c3-3_step4 one at a time, profiles VRAM,
  inspects internals (EMA), sanity-loads VAE (CUDA) + T5 (CPU), writes meta.json
- Awaiting completion → download `results/phase0/`

## Phase 0c: real inference test — NOT STARTED
- Must clone NVlabs/rcm on Kaggle, implement `src/causal_rcm/inference.py`
  (currently placeholders) against the repo, patch attention to SDPA for T4,
  generate a 10-frame rollout, measure real VRAM/latency

## E1 baseline — NOT STARTED (blocked on Phase 0c)
## E2/E3/E4/E5 — NOT STARTED

## Known deviations from MASTER_PLAN
1. No c1-1_step1 checkpoint exists → E1 fine modes are c1-1 step2 and step4
2. Old phase0 kernels errored without running (no quota wasted, ~14s total)
3. Phase 0 split into 0a/0b/0c because inference.py placeholders require
   rcm-repo integration before real generation is possible
