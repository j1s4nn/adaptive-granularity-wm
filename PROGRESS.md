# PROGRESS 鈥?operational plan v2

## Phase 0: feasibility and plan audit 鈥?COMPLETE

- Read the proposal without using its timeline as an execution constraint.
- Confirmed the core paper contribution is runtime fine/coarse switching, not action control.
- Locked text-only scope with `lambda_action=0` because the verified checkpoints expose no action interface.
- Reused all Phase 0b/0c artifacts; no Phase 0 rerun is authorized or needed.
- Replaced the original 86-GPU-hour plan with the two-day reduced plan at the top of `MASTER_PLAN.md`.
- Created `prompt.md` with project paths, Kaggle dataset links, GitHub link, safety rules, phase state, and audit protocol.
- Kept GitHub repository name `j1s4nn/adaptive-granularity-wm`; it is clear and professional.

## Phase 1a/1b: infrastructure correction and runtime-switch smoke test 鈥?COMPLETE

Required before any E1/E2/E3 claim:

- ported the proven single-file Phase 0c inference recipe into a real resumable experiment runner;
- fixed nested Kaggle mount discovery;
- removed the dummy path from the Phase 1 execution route;
- completed c1-1 鈫?c3-3 鈫?c1-1 with context/latent alignment checks;
- measured per-segment load, sampling, VRAM, and combined decode time;
- saved a reproducible PASS artifact and stopped for audit.

### Phase 1 proposal audit

| Requirement | Status |
|---|---|
| Runtime switching feasibility | Covered by real c1-1 -> c3-3 -> c1-1 sequence |
| Context/latent preservation | Covered for transferred boundary; diagnostics 0.0 |
| KV-state preservation | Unavailable; not claimed |
| Speed/compute accounting | Covered with load, sample, decode, VRAM |
| Uncertainty/recovery/action | Pending or deferred; lambda_action=0 |

Artifact: `results/phase1_switch_v2_dl/results/phase1_switch/20261005_031931/20261005_031931_meta.json`.

## Phase 2: fixed-mode baseline - COMPLETE (exploratory)

- Ran real c1-1 and c3-3 T4 inference for 2 prompts, 1 seed, and aligned 21/33/45-frame horizons (12 successful generations).
- Recorded per-condition DiT time, VAE time, peak VRAM, latent/frame variation, and saved first/last frames.
- Found and corrected a hard-coded two-run verdict condition; the raw Kaggle metadata is preserved, while `phase2_audit.json` records the independently verified 12/12 result. Corrected source was syntax-checked but not rerun remotely.
- Added reproducible per-run and aggregated CSV summaries under `results/phase2_baseline/results/phase2_baseline/20261005_042507/`.
- Findings are descriptive only: c1-1 DiT sampling averaged 10.217 s vs 24.027 s for c3-3; mean DiT+VAE generation was 84.260 s vs 100.550 s (19.3% lower for c1-1, excluding load/text setup); VAE decode dominates. Pixel L1 is content/motion dependent and cannot establish quality superiority.
- Limitation: checkpoint load and text encoding are absent from Phase 2 per-condition timing; include these before end-to-end latency claims. Sample size is too small to resolve a 0.1% target.

### Proposal coverage audit

| Proposal item | Phase 2 evidence/status |
|---|---|
| Fixed fine vs fixed coarse | Covered at pilot scale, matched prompts/seeds/horizons |
| Quality improvement | Not established; no reference metric and proxy is descriptive |
| Temporal behavior | Adjacent decoded-pixel and latent L1 recorded; motion-confounded |
| Speed/compute | DiT/VAE and VRAM measured; checkpoint/text setup overhead still to incorporate |
| Adaptive uncertainty | Not covered; Phase 3 |
| Recovery/hysteresis | Not covered; later controller phase |
| Action conditioning | Deferred; `lambda_action=0` |

Metadata: `results/phase2_baseline/results/phase2_baseline/20261005_042507/20261005_042507_meta.json`.

## Next phase: Phase 3 signal calibration

After every phase, update `MASTER_PLAN.md`, `STATUS.md`, `PROGRESS.md`, and `prompt.md`, record proposal coverage/deviations, give a compact audit table, and stop before the next phase.

# Historical progress below

## Phase 0a: environment + code probe 鈥?DONE
- Kaggle CLI 2.2.4 installed locally, authenticated as ajjisan (KGAT access_token)
- Dataset contents verified remotely; old broken phase0 kernels diagnosed and removed
- push_and_wait.py fixed for CLI 2.x

## Phase 0b: checkpoint + VRAM probe 鈥?DONE, VERDICT PASS
- Kernel `ajjisan/phase-0b-minimal-probe-checkpoints-only` v3 completed (~4 min wall)
- c3-3_step4 + c1-1_step2 both load on T4; peak VRAM 2.64 GB per model (of 14.56)
- No EMA weights 鈫?u_i^EMA dropped (pre-registered); residual + warp scores remain
- Meta: `results/phase0/phase0b_minimal_v3/results/phase0/20261004_092703_meta.json`

## Phase 0c: real inference test 鈥?DONE, VERDICT PASS (v7, after 6 fix iterations)
- Kernel `ajjisan/phase-0c-real-causal-rcm-inference-test-on-t4` v7 completed
- c1-1 1-step: DiT 4.5s, peak 4.05 GB, VAE 25.5s 鈥?frames saved OK
- c3-3 4-step: DiT 14.3s, peak 4.73 GB, VAE 46.6s 鈥?frames saved OK
- Working recipe (bf16->fp16 patches, local tokenizer, expandable_segments, meta/assign
  T5 trick, exact configs for all 4 E1 modes) documented in STATUS.md 搂WORKING RECIPE
- Meta: `results/phase0/phase0c_v7/results/phase0c/20261004_114113/20261004_114113_meta.json`

## E1 baseline 鈥?NOT STARTED (needs user approval to submit jobs)
- Design per MASTER_PLAN: 4 modes x 100 VBench prompts x 3 seeds x 10/25/50 frames
- Prompts available in rcm repo: evaluation/vbench_text2video/prompts.json (140 prompts)
- c3-3 frame counts must satisfy (T-3)%3==0: use {21,33,45} px frames (T=6,9,12)

## E2/E3/E4/E5 鈥?NOT STARTED

## Known deviations from MASTER_PLAN
1. No c1-1_step1 checkpoint 鈫?1-step mode = c1-1_step2 ckpt with steps_per_chunk [4,1]
   (repo docs confirm this is the intended 1-step recipe)
2. No EMA in checkpoints 鈫?E2 signals: u_i^res + u_i^warp (+ motion/complexity baselines)
3. Phase 0 split into 0a/0b/0c because inference.py was placeholder code
4. Kaggle script kernels are single-file only 鈫?self-contained job scripts
5. VAE decode is the latency bottleneck on T4 (25-47s); E1 must budget for it
   (decode after del net + expandable_segments to avoid fragmentation OOM)




