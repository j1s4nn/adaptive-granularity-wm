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

## Phase 0c: real inference test — DONE, VERDICT PASS (v7, after 6 fix iterations)
- Kernel `ajjisan/phase-0c-real-causal-rcm-inference-test-on-t4` v7 completed
- c1-1 1-step: DiT 4.5s, peak 4.05 GB, VAE 25.5s — frames saved OK
- c3-3 4-step: DiT 14.3s, peak 4.73 GB, VAE 46.6s — frames saved OK
- Working recipe (bf16->fp16 patches, local tokenizer, expandable_segments, meta/assign
  T5 trick, exact configs for all 4 E1 modes) documented in STATUS.md §WORKING RECIPE
- Meta: `results/phase0/phase0c_v7/results/phase0c/20261004_114113/20261004_114113_meta.json`

## E1 baseline — NOT STARTED (needs user approval to submit jobs)
- Design per MASTER_PLAN: 4 modes x 100 VBench prompts x 3 seeds x 10/25/50 frames
- Prompts available in rcm repo: evaluation/vbench_text2video/prompts.json (140 prompts)
- c3-3 frame counts must satisfy (T-3)%3==0: use {21,33,45} px frames (T=6,9,12)

## E2/E3/E4/E5 — NOT STARTED

## Known deviations from MASTER_PLAN
1. No c1-1_step1 checkpoint → 1-step mode = c1-1_step2 ckpt with steps_per_chunk [4,1]
   (repo docs confirm this is the intended 1-step recipe)
2. No EMA in checkpoints → E2 signals: u_i^res + u_i^warp (+ motion/complexity baselines)
3. Phase 0 split into 0a/0b/0c because inference.py was placeholder code
4. Kaggle script kernels are single-file only → self-contained job scripts
5. VAE decode is the latency bottleneck on T4 (25-47s); E1 must budget for it
   (decode after del net + expandable_segments to avoid fragmentation OOM)

