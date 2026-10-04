# PROGRESS (corrected — tracks reality, not the optimistic Day-1 plan)

## Phase 0a: environment + code probe — DONE
- Kaggle CLI 2.2.4 installed locally, authenticated as ajjisan (KGAT access_token)
- Dataset contents verified remotely; old broken phase0 kernels diagnosed and removed
- push_and_wait.py fixed for CLI 2.x

## Phase 0b: checkpoint + VRAM probe — DONE, VERDICT PASS
- Kernel `ajjisan/phase-0b-minimal-probe-checkpoints-only` v3 completed (~4 min wall)
- c3-3_step4 + c1-1_step2 both load on T4; peak VRAM 2.64 GB per model (of 14.56)
- No EMA weights → u_i^EMA dropped (pre-registered); residual + warp scores remain
- Meta: `results/phase0/phase0b_minimal_v3/results/phase0/20261004_092703_meta.json`

## Phase 0c: real inference test — NEXT
- Clone NVlabs/rcm on Kaggle, SDPA patch, fp16, T5 VRAM strategy (CPU-encode or int8),
  10-frame rollout per checkpoint, VRAM + latency + keyframes
- Then rewrite `src/causal_rcm/inference.py` (placeholders) against the working recipe

## E1 baseline — NOT STARTED (blocked on Phase 0c)
## E2/E3/E4/E5 — NOT STARTED

## Known deviations from MASTER_PLAN
1. No c1-1_step1 checkpoint → E1 fine modes are c1-1 step2 and step4
2. No EMA in checkpoints → E2 signals: u_i^res + u_i^warp (+ motion/complexity baselines)
3. Phase 0 split into 0a/0b/0c because inference.py was placeholder code
4. Kaggle script kernels are single-file only → self-contained job scripts
