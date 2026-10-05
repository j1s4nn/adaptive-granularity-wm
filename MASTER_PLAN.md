# OPERATIONAL MASTER PLAN v2 鈥?TWO-DAY PAPER COMPLETION

> **Status:** This operational plan supersedes the original three-day execution schedule below. The original plan is retained as design history; do not execute its 86-GPU-hour schedule, its placeholder E1 implementation, or its action-conditioned assumptions.

## Mission

Convert the proposal into a defensible empirical paper in the remaining two days. The paper's core contribution is **uncertainty-guided runtime switching between the matched Causal-rCM fine mode (c1-1, K=1) and coarse mode (c3-3, K=3)**, with recovery and hysteresis. The evidence must compare quality proxies, temporal stability, latency, compute, and long-horizon behavior against fixed and signal-based baselines.

The paper will not claim action-conditioned control because the verified checkpoints are text-to-video and contain no action interface. Set `lambda_action = 0`; retain the control term as an explicitly unavailable extension. A text-button prompt modifier may be built only as an optional demonstration and must never be presented as action conditioning or used as evidence for the main hypotheses.

## Non-negotiable evidence rules

1. Reuse the completed Phase 0b/0c artifacts. Do not repeat Phase 0.
2. Every reported generated-video result must come from real Causal-rCM inference using the proven Phase 0c recipe. Random outputs, dummy metrics, repeated placeholder prompts, and paper-only estimates are forbidden.
3. Validate at least one real c1-1 鈫?c3-3 鈫?c1-1 switch before E1/E3. If state transfer fails, stop the runtime-switch experiment and report an oracle/offline selection study instead of claiming online switching.
4. Use only signals available before the next segment is generated. EMA uncertainty is removed because Phase 0 found no EMA weights.
5. Split prompts/seeds into calibration and held-out test sets. Thresholds and AUROC must not be evaluated on the same examples used to choose them.
6. Include uncertainty computation, checkpoint switching, VAE decoding, and model-loading overhead in latency and compute accounting.
7. Text-to-video outputs do not provide ground-truth future frames. Do not report PSNR or reference LPIPS unless a declared reference set exists. Prefer VBench if it is immediately runnable; otherwise use explicitly named temporal-stability and pseudo-reference proxies.
8. A 0.1% improvement is a predeclared practical target, never a guaranteed result. Report paired bootstrap intervals and disclose when the interval cannot resolve 0.1%.
9. Every phase ends with an audit update in `prompt.md`, `STATUS.md`, and `PROGRESS.md`; then stop and report the phase, evidence, deviations from the proposal, and the next phase.
10. Never place Kaggle tokens or credentials in source files, `prompt.md`, Git history, or experiment outputs.

## Paper comparison matrix

The final paper must include all feasible comparisons below under the same T4 protocol:

| Comparison | Purpose | Required measurements |
|---|---|---|
| Fixed c1-1 vs fixed c3-3 | Establish fine/coarse trade-off | quality proxy, temporal stability, NFE, GPU time, FPS, memory |
| Threshold-only controller (R=0) | Isolate switching from recovery | same metrics, mode trace, switch count |
| Recovery controller | Test the main stabilisation mechanism | same metrics, switch cost, long-horizon terminal error |
| Motion controller | Proposal's signal baseline | quality/latency/compute/switching |
| Complexity controller | Proposal's signal baseline | quality/latency/compute/switching |
| Oracle fine/coarse selection | Upper-bound attainable selection benefit | matched-compute quality and cost |
| Causal-rCM literature reference | Contextual comparison only | VBench Total/Quality/Semantic, NFE, FPS, first/second latency, H100/T4 hardware note |

The Causal-rCM literature numbers are not direct replacements for our measurements: the paper uses its own VBench protocol and H100 hardware. Our claims must be stated as improvement over the matched fixed baseline under our declared T4 protocol.

## Reduced execution phases

### Phase 1 鈥?Infrastructure correction and switch smoke test

**Phase 1a status: COMPLETE.** The runner, nested dataset discovery, resumable metadata, and real fp16/SDPA Kaggle execution are verified.

**Phase 1b status: COMPLETE (latent/context smoke test).** A real c1-1 -> c3-3 -> c1-1 sequence completed on a Tesla T4. Boundary latent transfer succeeded in both directions through the I2V continuation API. KV-cache persistence is unavailable in the verified API and is not claimed; model reload and boundary-transfer costs are reported explicitly.

Evidence: `results/phase1_switch_v2_dl/results/phase1_switch/20261005_031931/20261005_031931_meta.json` and accompanying keyframes/log.

Use the proven single-file Phase 0c runner as the source of truth. Replace the placeholder reusable inference path, realign checkpoint loading to nested Kaggle mounts, add a real resumable job metadata file, and run one short c1-1 鈫?c3-3 鈫?c1-1 rollout. Record whether latent/context/KV transfer is valid and measure switching overhead. **Stop for audit.**

### Phase 2 鈥?E1 fixed-mode baseline

Use a small but real held-out prompt set and common valid horizons (21/33/45 frames for c3-3 alignment). Run fixed c1-1 schedules and c3-3 step-4. Compute real timing, NFE, memory, temporal stability, and the selected quality proxy. Produce fixed-mode comparison data and fine-preferred labels. **Stop for audit.**

**Phase 2 status: COMPLETE (exploratory baseline).** On a Tesla T4, 2 prompts × 1 seed × 3 aligned horizons × 2 fixed modes yielded 12/12 successful real generations. Artifacts: `results/phase2_baseline/results/phase2_baseline/20261005_042507/`. Mean DiT sampling was 10.217 s for c1-1 and 24.027 s for c3-3; mean DiT+VAE time was 84.260 s and 100.550 s (19.3% lower for c1-1 in this pilot, excluding model/text setup); mean decoded-frame adjacent-pixel L1 was 0.023899 and 0.024206 respectively. These are descriptive results only: the pixel-difference proxy is motion/content dependent, and does not establish perceptual quality or a statistically supported gain. DiT/VAE timings omit checkpoint load and text encoding; account for them separately before making end-to-end latency claims. No FVD, PSNR, or LPIPS is reported.

### Phase 3 鈥?E2 signal calibration

Compute residual/denoising-discrepancy and warp signals before each segment, plus motion and complexity baselines. Fit thresholds only on calibration prompts and report held-out AUROC with bootstrap intervals. Drop EMA and any unavailable action term. **Stop for audit.**

### Phase 4 鈥?E3 adaptive controller

Run the real controller on held-out prompts. Compare threshold-only, adaptive, fixed modes, and oracle selection at matched compute. Include all controller and switching overhead. Report quality鈥搇atency鈥揷ompute curves and the 0.1% practical target. **Stop for audit.**

### Phase 5 鈥?E4 recovery and final comparisons

Evaluate R=0 versus one or two recovery lengths. Add motion and complexity end-to-end baselines if they are executable under the same protocol. Drop E5 CF/CF++ transfer unless compatible checkpoints and time remain. Generate tables, figures, claims, deviations, and the paper results outline. **Stop for audit.**

## Explicit proposal coverage and deviations

| Proposal item | Operational status |
|---|---|
| RQ1 uncertainty vs motion/complexity | In scope; residual/warp only, EMA removed |
| RQ2 runtime fine/coarse switching | Switching feasibility verified in Phase 1; fixed-mode pilot in Phase 2; adaptive comparison pending |
| RQ3 recovery + hysteresis | In scope |
| RQ4 uncertainty鈥揵enefit relationship | In scope as exploratory reanalysis |
| Action-augmented potential | Not testable; lambda=0 and documented deviation |
| Camera-controlled interactive world model | Not testable with verified text-to-video checkpoints; scope narrowed explicitly |
| E5 CF/CF++ transfer | Optional only; never allowed to block the core paper |
| 100 prompts 脳 3 seeds 脳 all modes | Replaced by a real held-out pilot sized to finish in two days |
| FVD/PSNR/reference LPIPS | Only if valid references/features are available; otherwise replaced by declared proxies |

## Completion standard

The paper is ready when it contains real paired results for fixed fine, fixed coarse, threshold-only, and recovery controllers; a held-out signal comparison; a latency/compute accounting; a long-horizon stability analysis; a literature comparison table; and an explicit limitations section covering text-only conditioning, T4 hardware, sample size, and any failed switching assumption.

# MASTER PLAN: Uncertainty-Guided Adaptive Temporal Granularity
**Research Project**: Adaptive fine/coarse switching for Causal-rCM video world models  
**Student**: Hossen Md Jisan (4th year AI, NUIST)  
**GitHub**: j1s4nn (repo to be created)  
**Goal**: Publishable paper with complete experimental evidence  
**Timeline**: 3 DAYS (72 hours), 12 hours/day hands-on = 36 productive hours  
**Philosophy**: ALL EXPERIMENTS, ZERO COMPROMISES. Speed through parallelization, not shortcuts.

---

## EXECUTIVE SUMMARY

After Stage 0 reconnaissance, I have studied:
- Your 622-line proposal (every section, equation, table)
- Causal-rCM paper and codebase (github.com/NVlabs/rcm)
- CF++ baseline paper
- Your 3 Kaggle datasets (checkpoints verified)
- Kaggle T4 hardware constraints (16GB VRAM, Turing, no bf16)
- Adjacent uncertainty-adaptive methods (Surprise Forcing, DSA, ENkG)

**Core research question**: Can predictive uncertainty guide run-time switching between frame-wise (K=1, fast, drift-prone) and chunk-wise (K=3, stable, slower) generation, and does recovery+hysteresis control long-horizon error?

**Critical open problems identified**:
1. **No action-conditioned checkpoint** for Wan2.1-1.3B exists in your datasets or the release
2. **T4 compatibility**: Turing arch may not support FlashAttention-2; need PyTorch SDPA fallback
3. **Control term 螖a_i is unresolved** without an action model
4. **Camera drift is real**: Causal-rCM's own report鈥攆rame-wise 4-step fails after ~1k iters

**Recommendation**: Start with **text-to-video only** (no action), defer action to future work unless you find a checkpoint. The proposal's control-augmented potential (唯_i = u_i + 位螖a_i) becomes visual-only (唯_i = u_i, 位=0) as the baseline, with action as an aspirational extension.

**3-DAY EXECUTION STRATEGY**:
- **Parallelization**: Run Phase 0 + multiple E1 configs simultaneously on Kaggle (2脳T4 per job, submit 3-4 jobs in parallel)
- **Overlap**: While E1 runs (20 GPU-hours), implement E2/E3/E4 code locally (Day 1 afternoon)
- **No waiting**: Use `scripts/push_and_wait.py` with async polling; start next phase while previous downloads
- **Figure generation**: Automated scripts generate all figures during job runs (no manual post-processing)
- **Writing in parallel**: Outline paper Day 1 evening, fill results as experiments complete Days 2-3

---

## 1. PROJECT BACKBONE (Tree Structure)

```
04_Experiment/
鈹溾攢鈹€ CLAUDE.md                       # Project instructions (already present)
鈹溾攢鈹€ MASTER_PLAN.md                  # This file
鈹溾攢鈹€ TREE.md                         # Auto-generated tree (updated every phase)
鈹溾攢鈹€ INSTRUCTIONS.md                 # Cumulative manual steps
鈹溾攢鈹€ DEVIATIONS.md                   # Tracking proposal changes
鈹溾攢鈹€ CLAIMS.md                       # Claim鈫抏vidence mapping
鈹溾攢鈹€ PREREG.md                       # Pre-registered decision rules
鈹?鈹溾攢鈹€ config/                         # Hyperparameters, thresholds
鈹?  鈹溾攢鈹€ phase0_probe.yaml
鈹?  鈹溾攢鈹€ baseline_fixed.yaml         # E1: fixed c1-1, c3-3
鈹?  鈹溾攢鈹€ uncertainty_score.yaml      # E2: u_i components
鈹?  鈹溾攢鈹€ controller.yaml             # E3/E4: 蟿_high, 蟿_low, R, 位
鈹?  鈹斺攢鈹€ compute_budget.yaml         # Matched-compute normalization
鈹?鈹溾攢鈹€ src/                            # Python packages
鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹溾攢鈹€ causal_rcm/                 # Wrapper for NVlabs/rcm inference
鈹?  鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹?  鈹溾攢鈹€ load_checkpoints.py    # Mount Kaggle datasets, load c1-1/c3-3
鈹?  鈹?  鈹溾攢鈹€ inference.py           # Single-step generate, expose internals
鈹?  鈹?  鈹斺攢鈹€ fallback.py            # PyTorch SDPA if FA2 unavailable
鈹?  鈹?鈹?  鈹溾攢鈹€ uncertainty/                # Proposed component 1
鈹?  鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹?  鈹溾攢鈹€ ema_score.py           # u_i^EMA (v_胃 vs v_胃^-)
鈹?  鈹?  鈹溾攢鈹€ residual_score.py      # u_i^res (step i vs i-1)
鈹?  鈹?  鈹溾攢鈹€ warp_score.py          # u_i^warp (optical flow mismatch)
鈹?  鈹?  鈹溾攢鈹€ composite.py           # g_蠁: combine into u_i
鈹?  鈹?  鈹斺攢鈹€ calibrate.py           # E2: AUROC calibration
鈹?  鈹?鈹?  鈹溾攢鈹€ controller/                 # Proposed components 2 & 3
鈹?  鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹?  鈹溾攢鈹€ potential.py           # 唯_i = u_i + 位螖a_i (位=0 for text-only)
鈹?  鈹?  鈹溾攢鈹€ threshold.py           # Eq.(7) two-threshold switching
鈹?  鈹?  鈹溾攢鈹€ recovery.py            # Eq.(8) recovery counter r_i
鈹?  鈹?  鈹斺攢鈹€ policy.py              # Combined controller
鈹?  鈹?鈹?  鈹溾攢鈹€ baselines/                  # E5 comparison systems
鈹?  鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹?  鈹溾攢鈹€ motion_score.py        # LongScape/MotionCache-style
鈹?  鈹?  鈹溾攢鈹€ complexity_score.py    # EVATok/AdapTok-style
鈹?  鈹?  鈹斺攢鈹€ surprise_score.py      # Surprise Forcing-style (if feasible)
鈹?  鈹?鈹?  鈹溾攢鈹€ evaluation/                 # Metrics
鈹?  鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹?  鈹溾攢鈹€ fvd.py                 # Fr茅chet Video Distance
鈹?  鈹?  鈹溾攢鈹€ lpips.py               # Perceptual similarity
鈹?  鈹?  鈹溾攢鈹€ psnr.py                # Pixel fidelity
鈹?  鈹?  鈹溾攢鈹€ auroc.py               # Signal quality (E2)
鈹?  鈹?  鈹溾攢鈹€ switch_cost.py         # Mode-change artifact (E4)
鈹?  鈹?  鈹斺攢鈹€ compute_cost.py        # GPU-time / FLOPs tracker
鈹?  鈹?鈹?  鈹斺攢鈹€ utils/                      # Infrastructure
鈹?      鈹溾攢鈹€ __init__.py
鈹?      鈹溾攢鈹€ logging.py             # Structured logs (meta.json)
鈹?      鈹溾攢鈹€ tree_gen.py            # scripts/make_tree.py called here
鈹?      鈹溾攢鈹€ figure_style.py        # Colorblind-safe matplotlib theme
鈹?      鈹斺攢鈹€ kaggle_io.py           # Dataset mounting, result download
鈹?鈹溾攢鈹€ jobs/                           # Kaggle kernel metadata
鈹?  鈹溾攢鈹€ phase0/                     # E0 feasibility
鈹?  鈹?  鈹溾攢鈹€ kernel-metadata.json
鈹?  鈹?  鈹斺攢鈹€ phase0_probe.py
鈹?  鈹溾攢鈹€ e1_baseline/                # E1 fixed modes
鈹?  鈹溾攢鈹€ e2_signal/                  # E2 uncertainty AUROC
鈹?  鈹溾攢鈹€ e3_controller/              # E3 threshold policy
鈹?  鈹溾攢鈹€ e4_recovery/                # E4 with hysteresis
鈹?  鈹斺攢鈹€ e5_robustness/              # E5 cross-signal comparison
鈹?鈹溾攢鈹€ results/                        # Local copies only (small outputs)
鈹?  鈹溾攢鈹€ phase0/
鈹?  鈹?  鈹斺攢鈹€ <run_id>/
鈹?  鈹?      鈹溾攢鈹€ meta.json          # Git commit, checkpoint sha256, status
鈹?  鈹?      鈹溾攢鈹€ config.yaml
鈹?  鈹?      鈹溾攢鈹€ logs/feasibility.log
鈹?  鈹?      鈹斺攢鈹€ artifacts/vram_profile.txt
鈹?  鈹溾攢鈹€ e1_baseline/
鈹?  鈹?  鈹斺攢鈹€ <run_id>/
鈹?  鈹?      鈹溾攢鈹€ meta.json
鈹?  鈹?      鈹溾攢鈹€ raw/
鈹?  鈹?      鈹?  鈹溾攢鈹€ fixed_fine_step{1,2,4}.jsonl  # FVD/LPIPS/PSNR per prompt
鈹?  鈹?      鈹?  鈹斺攢鈹€ fixed_coarse_step4.jsonl
鈹?  鈹?      鈹溾攢鈹€ figures_src/       # Scripts that generate figures
鈹?  鈹?      鈹?  鈹斺攢鈹€ plot_error_curves.py
鈹?  鈹?      鈹斺攢鈹€ artifacts/
鈹?  鈹?          鈹溾攢鈹€ error_vs_length.pdf
鈹?  鈹?          鈹斺攢鈹€ baseline_table.tex
鈹?  鈹斺攢鈹€ ... (e2, e3, e4, e5 follow same subtree)
鈹?鈹溾攢鈹€ scripts/                        # Automation
鈹?  鈹溾攢鈹€ push_and_wait.py           # Submit job, poll, download
鈹?  鈹溾攢鈹€ make_tree.py               # Regenerate TREE.md
鈹?  鈹斺攢鈹€ update_instructions.py     # Append to INSTRUCTIONS.md
鈹?鈹斺攢鈹€ docs/                           # Visualizations (generated from raw/)
    鈹溾攢鈹€ figures/                    # All paper-ready PDFs/PNGs
    鈹?  鈹溾攢鈹€ fig_error_curves.pdf
    鈹?  鈹溾攢鈹€ fig_mode_trace.pdf
    鈹?  鈹溾攢鈹€ fig_roc.pdf
    鈹?  鈹溾攢鈹€ fig_qualitative.pdf
    鈹?  鈹斺攢鈹€ ... (see 搂7 VISUALIZATION PLAN)
    鈹斺攢鈹€ tables/
        鈹溾攢鈹€ tab_baseline.tex
        鈹溾攢鈹€ tab_signal_auroc.tex
        鈹溾攢鈹€ tab_ablations.tex
        鈹斺攢鈹€ coverage_matrix.tex     # RQ鈫抎ata鈫抐igure mapping
```

**INPUT鈫扥UTPUT flow**:
- **Phase 0**: Kaggle datasets 鈫?VRAM profile, internals list, action format 鈫?go/no-go
- **E1**: Prompts 鈫?c1-1/c3-3 generate 鈫?raw JSONL 鈫?error curves + baseline table
- **E2**: E1 rollouts 鈫?uncertainty scores 鈫?AUROC vs coarse-safe/unsafe labels
- **E3**: E2 calibrated u_i 鈫?threshold controller 鈫?quality/latency/compute vs fixed
- **E4**: E3 + recovery counter 鈫?switch count, switch cost, long-horizon error
- **E5**: CF/CF++ checkpoints (if feasible) 鈫?signal transfer AUROC + end-to-end comparison

---

## 2. PHASES (Detailed Design)

### Phase 0: Feasibility Audit (1 day, GPU: ~1 hour)
**Goal**: Confirm hardware, internals, action format before main experiments.

**INPUT**:
- Kaggle datasets: ajjisan/causal-rcm-ckpts, ajjisan/wan21-t5, ajjisan/wan21-dit-vae
- NVlabs/rcm inference code (cloned on Kaggle)

**TASKS**:
1. Load c1-1_step{2,4} and c3-3_step4 checkpoints (matched pair from same recipe)
2. Run single-frame generate on T4, log peak VRAM at 480p
3. Test PyTorch SDPA fallback if FlashAttention-2 unavailable
4. Inspect exposed internals: EMA model (v_胃^-), latents, KV buffer
5. Check Cosmos 3 action interface (if action checkpoint exists)
6. Confirm VAE decode works (Wan2.1 VAE)

**OUTPUT**: `results/phase0/<run_id>/`
- `meta.json`: checkpoint sha256, library versions, GPU name, git commit
- `logs/feasibility.log`: Load times, VRAM peaks, errors
- `artifacts/vram_profile.txt`: c1-1 vs c3-3 memory usage
- `artifacts/internals_list.txt`: Which model attributes are accessible

**SUCCESS**: Both checkpoints fit in 16GB, inference runs, EMA model accessible  
**FAILURE**: VRAM overflow 鈫?reduce batch or resolution; retry at 360p

**DECISION GATE**: If no EMA model 鈫?drop u_i^EMA, use only residual+warp. If no action checkpoint 鈫?proceed text-only (位=0). Report both to you before Phase 1.

---

### Phase 1a: E1 Fixed-Mode Baseline (3 days, GPU: ~20 hours)
**Goal**: Establish error-vs-length curves, coarse-unsafe labels, compute costs.

**INPUT**:
- Prompts: 100 from VBench-T2V (your proposal cites it as the standard)
- c1-1_step{1,2,4} and c3-3_step4 checkpoints
- Seeds: 3 per prompt (300 rollouts total)
- Rollout length: 10, 25, 50 frames (your proposal's 3 horizons)

**TASKS**:
1. Generate with fixed fine (c1-1, S鈭坽1,2,4}) and fixed coarse (c3-3, S=4)
2. Measure per-segment GPU time (ms), FLOPs estimate
3. Compute FVD, LPIPS, PSNR at 10/25/50 frames
4. **Label coarse-unsafe segments**: where fixed-coarse quality drops >螖 below fixed-fine

**OUTPUT**: `results/e1_baseline/<run_id>/raw/`
- `fixed_fine_step1.jsonl` (one line per prompt/seed/frame: {prompt_id, seed, frame_idx, fvd, lpips, psnr, gpu_ms})
- `fixed_fine_step2.jsonl`, `fixed_fine_step4.jsonl`, `fixed_coarse_step4.jsonl`
- `compute_costs.json`: {mode: {K, S, mean_gpu_ms, flops}}
- `unsafe_labels.jsonl`: {prompt_id, seed, segment_idx, is_unsafe: bool}

**FIGURES** (generated from raw):
- `fig_error_curves.pdf`: FVD/LPIPS/PSNR vs rollout length, 4 fixed modes + 95% CI
- `tab_baseline.tex`: Quality@10/25/50, latency, compute for each mode

**SUCCESS**: Curves show frame-wise error grows faster; 螖 threshold separates modes  
**FAILURE**: All modes degrade equally 鈫?no switching benefit; report as negative result

---

### Phase 1b: E2 Signal Calibration (2 days, GPU: ~10 hours)
**Goal**: AUROC of uncertainty vs motion vs complexity for coarse-unsafe prediction.

**INPUT**:
- E1 rollouts (reuse the generated frames)
- E1 labels: coarse-unsafe segments

**TASKS**:
1. **Compute before each segment**:
   - u_i^EMA: ||v_胃(x_t, x^<i, t) - v_胃^-(x_t, x^<i, t)||
   - u_i^res: ||pred_i - pred_{i-1}|| (self-consistency)
   - u_i^warp: ||x_{i-1} - W(x_{i-2}, flow_{i-2鈫抜-1})||
   - Composite u_i = g_蠁(EMA, res, warp) [test: mean, max, tiny MLP]
2. **Baseline signals**:
   - Motion: mean optical flow magnitude
   - Complexity: edge density or spatial entropy
3. **ROC analysis**: Each signal vs E1 labels, report AUROC

**OUTPUT**: `results/e2_signal/<run_id>/`
- `raw/scores.jsonl`: {prompt_id, seed, segment_idx, u_ema, u_res, u_warp, u_composite, motion, complexity, label}
- `figures_src/plot_roc.py` 鈫?`artifacts/fig_roc.pdf`: 5 ROC curves (ours + 2 baselines + ablations)
- `artifacts/tab_signal_auroc.tex`: Signal | AUROC | 95% CI

**SUCCESS**: u_i AUROC 鈮?motion/complexity (H1 support)  
**FAILURE**: Motion beats uncertainty 鈫?uncertainty is not the right signal (H1 weakened, still informative)

**CALIBRATION**: If u_i AUROC > 0.6, proceed. Learn 蟿_high, 蟿_low from ROC knee (Youden index).

---

### Phase 2: E3 Threshold Controller vs Fixed (4 days, GPU: ~25 hours)
**Goal**: Test RQ2 鈥?does switching beat the best fixed mode at matched compute?

**INPUT**:
- Calibrated u_i (from E2)
- 蟿_high, 蟿_low (from E2 ROC)
- Same 100 prompts, 3 seeds

**TASKS**:
1. Run controller (Eq. 7 from proposal): 唯_i = u_i (位=0 for text-only)
   - If 唯_i 鈮?蟿_high 鈫?fine (c1-1)
   - If 唯_i 鈮?蟿_low 鈫?coarse (c3-3)
   - Else keep previous mode
2. Track total compute: C = 危 魏(m_i)
3. Compare vs best fixed mode at **same compute** (adjust fixed-mode step count)
4. Measure quality at 10/25/50 frames, latency, switch count

**OUTPUT**: `results/e3_controller/<run_id>/`
- `raw/adaptive.jsonl`: per-frame quality, mode trace, 唯_i, compute
- `raw/fixed_matched.jsonl`: best fixed mode at matched compute
- `figures_src/plot_quality_vs_compute.py` 鈫?`fig_quality_latency_compute.pdf`
- `artifacts/tab_controller_vs_fixed.tex`

**SUCCESS**: Adaptive FVD 鈮?fixed at same C, or same FVD at lower C (H2 support)  
**FAILURE**: Fixed mode equals or beats adaptive 鈫?switching offers no benefit (H2 weakened, pre-registered negative)

**RQ4 analysis** (reuses E3 data): Bin segments by u_i quartile, compute fine-vs-coarse quality gap per bin 鈫?check monotonicity. Report as finding either way.

---

### Phase 3: E4 Recovery + Hysteresis (2 days, GPU: ~12 hours)
**Goal**: Test RQ3 鈥?does recovery counter reduce oscillation and improve quality?

**INPUT**:
- E3 controller (threshold-only, R=0)
- Recovery lengths: R 鈭?{0, 3, 5, 10} steps

**TASKS**:
1. Add recovery counter (Eq. 8 from proposal):
   - On spike (唯_i 鈮?蟿_high): r_i 鈫?R, stay fine for R steps
   - Decrement r_i each step until 0, then return to threshold rule
2. Measure:
   - Switch count (fewer is better)
   - Switch cost: LPIPS spike at mode change
   - Long-horizon quality (50-frame FVD)

**OUTPUT**: `results/e4_recovery/<run_id>/`
- `raw/recovery_sweep.jsonl`: {R, switch_count, switch_cost, fvd50}
- `figures_src/plot_mode_trace.py` 鈫?`fig_mode_trace.pdf`: 唯_i, mode, r_i over time
- `artifacts/tab_recovery.tex`

**SUCCESS**: R>0 reduces switches, maintains or improves quality (H3 support)  
**FAILURE**: No difference 鈫?hysteresis unnecessary at this setting (H3 weakened)

---

### Phase 4: E5 Cross-Recipe Robustness + End-to-End Comparison (3 days, GPU: ~18 hours)
**Goal**: Does calibrated uncertainty transfer to CF/CF++? Full baseline sweep.

**INPUT**:
- CF (chunk-wise 4-step, if accessible)
- CF++ (frame-wise 2-step, if accessible)
- Calibrated 蟿_high, 蟿_low from E2 (Causal-rCM)

**TASKS**:
1. **Cross-recipe**: Re-run uncertainty controller on CF/CF++, measure AUROC and quality
2. **Full baselines**:
   - Motion controller: threshold on optical flow
   - Complexity controller: threshold on scene entropy
   - Surprise Forcing-style S-only (if feasible after Phase 0)
   - 位=0 ablation (already covered in E3)
3. End-to-end quality/latency/compute comparison

**OUTPUT**: `results/e5_robustness/<run_id>/`
- `raw/cf_transfer.jsonl`, `raw/motion_end2end.jsonl`, etc.
- `artifacts/tab_full_comparison.tex`: All systems, all metrics

**SUCCESS**: Uncertainty transfers (AUROC 鈮?0.55 on CF/CF++) and beats motion/complexity end-to-end  
**FAILURE**: Transfer fails or motion wins 鈫?signal is recipe-specific or not the best choice

---

### Phase 5: Analysis & Writing (1 week, no GPU)
**Goal**: Generate all figures, finalize CLAIMS.md, write paper outline.

**TASKS**:
1. Run all `figures_src/*.py` scripts 鈫?populate `docs/figures/`
2. Update `coverage_matrix.tex`: every RQ/hypothesis 鈫?data file 鈫?figure/table
3. Finalize DEVIATIONS.md (what changed from proposal)
4. Pre-registered decision rules (PREREG.md): check every hypothesis
5. Write paper outline (see 搂4 PAPER MAP)

**OUTPUT**:
- `docs/figures/`: 12+ publication-ready PDFs
- `docs/tables/`: LaTeX tables for each experiment
- `paper_outline.md`: Structured sections with data pointers
- `CLAIMS.md`: Each claim 鈫?[SUPPORTED / WEAKENED / UNSUPPORTED] + evidence file

---

## 3. PROPOSAL COVERAGE MATRIX

| RQ/Hyp | Research Question | Experiment | Raw Data File | Figure/Table | Decision Rule |
|--------|-------------------|------------|---------------|--------------|---------------|
| **RQ1** | Does uncertainty beat motion/complexity? | E2 | `e2_signal/raw/scores.jsonl` | `fig_roc.pdf`, `tab_signal_auroc.tex` | AUROC(u_i) 鈮?AUROC(motion) AND AUROC(u_i) 鈮?AUROC(complexity) 鈫?H1 SUPPORTED |
| **H1** | Uncertainty identifies coarse-unsafe moments better | E2 | Same | Same | Same |
| **RQ2** | Does adaptive switching beat fixed modes? | E3 | `e3_controller/raw/adaptive.jsonl` | `fig_quality_latency_compute.pdf`, `tab_controller_vs_fixed.tex` | (Adaptive FVD@50 鈮?Fixed FVD@50 at matched C) OR (same FVD at lower C) 鈫?H2 SUPPORTED |
| **H2** | Switching improves quality-latency-compute trade-off | E3 | Same | Same | Same |
| **RQ3** | Does recovery+hysteresis reduce oscillation? | E4 | `e4_recovery/raw/recovery_sweep.jsonl` | `fig_mode_trace.pdf`, `tab_recovery.tex` | (Switch count with R>0 < Switch count with R=0) AND (FVD@50 with R>0 鈮?FVD@50 with R=0) 鈫?H3 SUPPORTED |
| **H3** | Recovery counter stabilizes long-horizon quality | E4 | Same | Same | Same |
| **RQ4** | Is uncertainty鈫抌enefit relationship monotonic? | E3 reanalysis | `e3_controller/raw/adaptive.jsonl` | `fig_monotonicity.pdf` | Report empirical relationship; no support/weaken (exploratory) |
| **H4** | Exploratory: non-monotonicity like Causal-rCM's S-behavior? | E3 reanalysis | Same | Same | Report finding either way |
| **Baseline char.** | Error-vs-length curves (missing from literature) | E1 | `e1_baseline/raw/*.jsonl` | `fig_error_curves.pdf`, `tab_baseline.tex` | N/A (descriptive) |
| **Cross-recipe** | Does signal transfer to CF/CF++? | E5 | `e5_robustness/raw/cf_transfer.jsonl` | `tab_full_comparison.tex` | AUROC(CF/CF++) 鈮?0.55 鈫?Transfer SUCCESSFUL |
| **Motion baseline** | Motion-driven controller | E5 | `e5_robustness/raw/motion_end2end.jsonl` | `tab_full_comparison.tex` | End-to-end FVD comparison |
| **Complexity baseline** | Complexity-driven controller | E5 | `e5_robustness/raw/complexity_end2end.jsonl` | `tab_full_comparison.tex` | End-to-end FVD comparison |

**Unaddressed from proposal** (due to missing action checkpoint):
- Control-augmented potential 唯_i = u_i + 位螖a_i: Deferred to 位=0 (text-only). If you find an action checkpoint, add Phase 6.
- Cosmos 3 action-conditioned setting: Not feasible without action checkpoint.

---

## 4. PAPER MAP (Ideal Structure)

**Abstract** (250 words)
- Background: Real-time AR video needs low latency; two fixed regimes exist (frame-wise fast+drift, chunk-wise stable+slow)
- Gap: No run-time switching, no uncertainty signal, long-horizon error unquantified
- Method: Uncertainty score u_i, threshold controller, recovery+hysteresis on frozen Causal-rCM
- Results: [E1 baseline, E2 AUROC, E3 adaptive vs fixed, E4 recovery benefit]
- Conclusion: Uncertainty [supports/weakens] hypothesis, adaptive [matches/beats] fixed at matched compute

**1. Introduction** (1.5 pages)
- Problem: Interactive video needs causal, controllable, low-latency generation
- Existing: CF, CF++, Causal-rCM achieve real-time in fixed regimes
- Limitation: Granularity is design-time constant; frame-wise drift acknowledged but not mitigated
- Contribution: First uncertainty-guided run-time switching, recovery mechanism, long-horizon study
- **Needs**: E1 error curves, motivation for switching

**2. Related Work** (1 page)
- Theme A: AR diffusion distillation (CF family)
- Theme C: Adaptive temporal allocation (LongScape, EVATok)
- Theme F: Uncertainty-adaptive compute (Surprise Forcing, DSA, ENkG 鈥?but non-interactive)
- Theme G: Hysteresis switching (DASH-OPD, SUNTA 鈥?but not video generation)
- **Needs**: Claim boundary table (from proposal 搂3)

**3. Background** (0.5 page)
- Causal-rCM matched pair (c1-1, c3-3)
- Exposure bias and error accumulation (CF++ Sec 3.1)
- **Needs**: Baseline latencies from E1, Causal-rCM VBench scores (cite)

**4. Method** (2 pages)
- Notation (Table from proposal)
- Uncertainty score u_i (Eq. 2-4 from proposal)
- Threshold controller (Eq. 7 from proposal)
- Recovery+hysteresis (Eq. 8 from proposal)
- Matched-compute fairness (Eq. 9 from proposal)
- **Needs**: System flow diagram (Fig. 4 from proposal)

**5. Experiments** (3 pages)
- Setup: Kaggle T4, 100 prompts, 3 seeds, 10/25/50 frames
- E1: Fixed-mode baseline 鈫?error curves, compute costs
- E2: Signal calibration 鈫?ROC curves, AUROC table
- E3: Adaptive vs fixed 鈫?quality-latency-compute plots
- E4: Recovery sweep 鈫?mode traces, switch cost
- E5: Robustness 鈫?cross-recipe transfer, full baseline comparison
- **Needs**: ALL figures from `docs/figures/`, ALL tables from `docs/tables/`

**6. Results** (2 pages)
- RQ1: [Uncertainty beats / ties / loses to] motion/complexity (AUROC X.XX vs Y.YY)
- RQ2: [Adaptive beats / matches / loses to] best fixed at matched C (FVD@50: X.X vs Y.Y)
- RQ3: [Recovery reduces / has no effect on] switches (count: X vs Y) and [improves / maintains] quality
- RQ4: Uncertainty鈫抌enefit is [monotonic / peaks at mid-range / non-monotonic] (exploratory)
- **Needs**: Qualitative side-by-side (Fig. 6 from proposal sketch), long-horizon drift strips

**7. Limitations** (0.5 page)
- Text-only (no action control due to missing checkpoint)
- Short evaluation clips (50 frames, not minutes)
- T4 hardware (Turing, 16GB) limits batch size
- Single model family (Causal-rCM only; CF/CF++ as secondary check)

**8. Conclusion** (0.5 page)
- Summary of findings
- Contribution: [first uncertainty-guided run-time switching / negative result on switching benefit]
- Future: Action-conditioned controller, RL policies, longer horizons

**9. Broader Impact & Ethics** (0.3 page)
- Deepfakes, misinformation risk (standard AR video warning)
- Compute efficiency may reduce carbon footprint

**References**: ~40-50 (proposal cites 25+, experiments add 15-20)

---

## 5. COMPARISON DESIGN (All Systems Under Test)

| System | Description | K | S | Switching? | Signal | Recovery? | Experiment |
|--------|-------------|---|---|------------|--------|-----------|------------|
| **Fixed fine (c1-1)** | Frame-wise, S=1 | 1 | 1 | No | 鈥?| 鈥?| E1 baseline |
| **Fixed fine (c1-1)** | Frame-wise, S=2 | 1 | 2 | No | 鈥?| 鈥?| E1 baseline |
| **Fixed fine (c1-1)** | Frame-wise, S=4 | 1 | 4 | No | 鈥?| 鈥?| E1 baseline |
| **Fixed coarse (c3-3)** | Chunk-wise, S=4 | 3 | 4 | No | 鈥?| 鈥?| E1 baseline |
| **Uncertainty threshold (no recovery)** | Adaptive | 1/3 | {1,2,4}/4 | Yes | u_i | No | E3 |
| **Uncertainty + recovery (R=3)** | Adaptive | 1/3 | {1,2,4}/4 | Yes | u_i | Yes | E4 |
| **Uncertainty + recovery (R=5)** | Adaptive | 1/3 | {1,2,4}/4 | Yes | u_i | Yes | E4 |
| **Uncertainty + recovery (R=10)** | Adaptive | 1/3 | {1,2,4}/4 | Yes | u_i | Yes | E4 |
| **Motion threshold** | Adaptive | 1/3 | {1,2,4}/4 | Yes | Optical flow | No | E5 |
| **Complexity threshold** | Adaptive | 1/3 | {1,2,4}/4 | Yes | Scene entropy | No | E5 |
| **位=0 ablation** | Adaptive, visual-only | 1/3 | {1,2,4}/4 | Yes | u_i (no control term) | Yes/No | E3 (baseline=位0 for text) |
| **Random switch (sanity)** | Adaptive, random | 1/3 | {1,2,4}/4 | Yes | Bernoulli(0.5) | No | E5 optional |
| **Surprise Forcing-style (if feasible)** | Adaptive | 1/3 | {1,2,4}/4 | Yes | Post-1st-pass surprise | No | E5 (Phase 0 gate) |
| **CF/CF++ transfer (if feasible)** | Cross-recipe | 鈥?| 鈥?| Yes | u_i (calibrated on rCM) | No | E5 (Phase 0 gate) |

**Matched compute**: All adaptive systems normalized to same total C as best fixed mode. E1 measures 魏(F)={K=1, S}, 魏(C)={K=3, S=4}. E3/E4/E5 adjust thresholds or step budgets to match.

**Sanity checks**:
- Random switch: Should perform between best and worst fixed modes (if it beats all, switching itself is beneficial)
- Oracle (aspirational): If you had ground-truth "safe/unsafe" labels at test time, what's the upper bound? Compute offline from E1 labels.

---

## 6. LONG-HORIZON STUDY (Dedicated Design)

**Motivation**: Your central hypothesis (proposal page 2) is that adaptive granularity helps **long** rollouts where frame-wise error accumulates. Published evals stop at ~81 frames; you need 100-200 frames to be decisive.

**Design**:
1. **Rollout lengths**: 10, 25, 50, 100, 150 frames (if VRAM allows; else cap at 100)
2. **Prompts**: 50 prompts (subset of VBench) with motion diversity (static鈫抎ynamic spectrum)
3. **Systems**: Fixed fine S=2 (best latency), Fixed coarse S=4 (best stability), Adaptive+recovery
4. **Metrics**:
   - FVD at each length milestone
   - Cumulative drift: distance from initial scene embedding
   - Visual artifacts: manual labeling of 10 random rollouts at 100 frames
5. **Hypothesis**: Fixed-fine error should grow faster than fixed-coarse after ~50 frames; adaptive should track the better curve

**OUTPUT**: `results/long_horizon/`
- `raw/long_rollouts.jsonl`: per-frame FVD, drift, mode trace
- `fig_long_horizon_curves.pdf`: FVD vs length for 3 systems + 95% CI
- `fig_drift_strips.pdf`: Side-by-side frame strips at 10/50/100 frames
- `artifacts/artifact_count.txt`: Manual count of visible glitches

**Pre-registered decision**: If adaptive FVD@100 鈮?best fixed FVD@100, long-horizon hypothesis SUPPORTED. If fixed-coarse always wins, frame-wise switching offers no stability gain (negative result, still publishable).

**Risk**: 100-frame rollouts may exhaust VRAM. Fallback: Generate in chunks, cache KV carefully. If still fails, cap at 50 and acknowledge as limitation.

---

## 7. VISUALIZATION PLAN (All Figures)

**Colorblind-safe palette**: Use Okabe-Ito or Viridis. All generated from raw data, no manual editing.

| Figure | Type | Data Source | Purpose |
|--------|------|-------------|---------|
| `fig_error_curves.pdf` | Line plot | E1 raw JSONL | FVD/LPIPS/PSNR vs rollout length, 4 fixed modes, 95% CI |
| `fig_roc.pdf` | ROC curves | E2 raw scores | 5 ROCs: u_i^EMA, u_i^res, u_i^warp, u_i composite, motion, complexity |
| `fig_quality_latency_compute.pdf` | Pareto frontier | E3 raw adaptive + fixed | 3D scatter or paired bars: FVD vs latency vs compute |
| `fig_mode_trace.pdf` | Timeline | E4 raw recovery sweep | 唯_i (line), 蟿_high/蟿_low (bands), mode (color bar), r_i (shaded) over 50 frames |
| `fig_qualitative.pdf` | Frame grid | E3/E4 generated videos | Side-by-side: same prompt, fixed-fine / fixed-coarse / adaptive, at frames 1/10/25/50 |
| `fig_long_horizon_curves.pdf` | Line plot | Long-horizon raw | FVD vs length 10鈫?00, 3 systems, 95% CI |
| `fig_drift_strips.pdf` | Frame strips | Long-horizon raw | 3 rollouts 脳 3 systems, keyframes at 10/30/50/70/100 |
| `fig_switch_cost.pdf` | Histogram | E4 raw | LPIPS spike distribution at mode transitions |
| `fig_threshold_sweep.pdf` | Heatmap | E3 ablation | Quality (color) vs 蟿_high (x) vs 蟿_low (y) |
| `fig_recovery_sweep.pdf` | Bar chart | E4 raw | Switch count + FVD@50 vs R 鈭?{0,3,5,10} |
| `fig_signal_overlays.pdf` | Video + signal | E2 raw | Frame strip with u_i, motion, complexity overlaid as line plots |
| `fig_monotonicity.pdf` | Scatter + trend | E3 reanalysis | Fine-vs-coarse benefit (y) vs uncertainty quartile (x) |
| `fig_failure_gallery.pdf` | Frame grid | E1/E3/E4 worst cases | 6 failure modes: drift, blur, color shift, object disappearance, camera jump, mode-switch seam |

**Tables** (all LaTeX, generated from raw):
- `tab_baseline.tex`: 4 fixed modes 脳 [FVD@10/25/50, latency, compute]
- `tab_signal_auroc.tex`: 6 signals 脳 [AUROC, 95% CI]
- `tab_controller_vs_fixed.tex`: Adaptive vs best fixed 脳 [FVD@50, latency, compute, switch count]
- `tab_recovery.tex`: R sweep 脳 [switch count, switch cost, FVD@50]
- `tab_full_comparison.tex`: All systems (E5) 脳 all metrics
- `tab_ablations.tex`: Signal components, threshold sweep, hysteresis band width, 位 sweep, S-step choice
- `coverage_matrix.tex`: RQ 鈫?data file 鈫?figure 鈫?decision

**Videos** (stay on Kaggle, download only keyframes):
- 5 side-by-side comparison videos (fixed-fine / fixed-coarse / adaptive), 10 seconds each
- Host on Kaggle dataset `ajjisan/adaptive-granularity-videos`, link in paper

---

## 8. OPEN PROBLEMS & RECOMMENDED OPTIONS

### Problem 1: No action-conditioned checkpoint
**Impact**: Cannot test control-augmented potential 唯_i = u_i + 位螖a_i (Eq. 6 from proposal).  
**Options**:
1. **Recommended**: Proceed text-only (位=0). Frame as "visual uncertainty baseline" that future work can extend with action.
2. Search for camera-conditioned checkpoint (minWM demos? Other Wan releases?)
3. Fine-tune your own action head (costs >500 GPU-hours, out of scope)

**Decision**: Option 1. Note in DEVIATIONS.md: "Control term deferred; 位=0 baseline establishes visual signal contribution."

---

### Problem 2: T4 compatibility (Turing, not Ampere)
**Impact**: FlashAttention-2 may fail (requires compute capability 8.0+, T4 is 7.5). Repo setup expects FA2.  
**Options**:
1. **Recommended**: Detect FA2 failure, fall back to PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention)
2. Rewrite attention kernels for Turing (weeks of work, not feasible)
3. Request A100 on Kaggle (not guaranteed; T4 is default GPU)

**Decision**: Option 1. Implement in `src/causal_rcm/fallback.py`. Phase 0 confirms which path works.

---

### Problem 3: Camera drift in frame-wise 4-step
**Impact**: Causal-rCM's own report鈥攆rame-wise 4-step develops directional camera bias, trains stably for only ~1k iters.  
**Options**:
1. **Recommended**: Use frame-wise S=2 as primary (paper reports VBench 84.63, stable for ~3k iters). Cap S=4 rollouts at 25 frames in E1.
2. Accept drift as the phenomenon under study (your proposal already cites it)
3. Action conditioning might fix it (but no action checkpoint)

**Decision**: Option 1 + 2. Frame-wise S=2 is the main fine mode. Document drift in E1 baseline, mention as motivation for switching.

---

### Problem 4: What is "matched compute"?
**Impact**: Fairness comparison (E3) requires normalizing C = 危 魏(m_i). Papers report latency (ms/frame), not FLOPs.  
**Options**:
1. **Recommended**: Measure GPU-time per segment in E1 (ms), use as 魏(路). Report FLOPs estimate (N_params 脳 2 脳 seq_len) in appendix.
2. Use first-frame latency only (ignores amortization)
3. Token count (but Causal-rCM is latent-space, unclear mapping)

**Decision**: Option 1. Define compute operationally: "C is total GPU-seconds excluding VAE, measured on T4 with batch=1."

---

### Problem 5: How to measure "quality" without ground truth?
**Impact**: Text-to-video has no reference frames for PSNR. FVD needs a reference dataset.  
**Options**:
1. **Recommended**: FVD vs VBench-T2V distribution (1000 generated samples). LPIPS measures self-consistency frame-to-frame. PSNR measures reconstruction if you have ground-truth clips (else skip).
2. VBench-T2V scores (quality + semantic subscores)
3. CLIP-score (text-video alignment)

**Decision**: Option 1 + 2. Primary: FVD, LPIPS, VBench. PSNR only if reference clips exist. CLIP-score optional (E5).

---

### Problem 6: CF/CF++ checkpoints for E5
**Impact**: Cross-recipe robustness test (E5) needs CF (chunk-wise) and CF++ (frame-wise) checkpoints.  
**Options**:
1. **Recommended**: Phase 0 searches Hugging Face (thu-ml/Causal-Forcing, official CF++ release). If found, proceed. If not, skip E5 CF/CF++ and mark as future work.
2. Train your own (>1000 GPU-hours, out of scope)

**Decision**: Option 1. E5 is secondary (robustness check), not critical path. Main claims rest on E1-E4.

---

### Problem 7: Latent frames vs pixel frames
**Impact**: Causal-rCM generates latent frames (16脳downsampled), VAE decodes to pixels. Quality metrics on which space?  
**Options**:
1. **Recommended**: Generate latent, VAE-decode each frame, measure FVD/LPIPS on pixels (standard practice per CF/CF++ papers)
2. Measure in latent space (faster, but non-standard)

**Decision**: Option 1. Always decode to pixels for metrics. Latent-space is internal only.

---

### Problem 8: Maximum rollout length T4 can support
**Impact**: Long-horizon study (搂6) wants 100-150 frames, but KV cache grows.  
**Options**:
1. **Recommended**: Phase 0 tests max length at 480p, batch=1. If <100, reduce to 75 or 50 and acknowledge limitation.
2. Sliding window KV cache (evict oldest)
3. Reduce resolution to 360p

**Decision**: Option 1. Phase 0 profiles VRAM vs length. Report actual max in paper. If 鈮?00, great. If 50, still sufficient (proposal calls 50 frames "long-horizon" relative to published 10-81).

---

## 9. YOUR DECISIONS (Must Make Before Phase 1)

1. **Action checkpoint**: Proceed text-only (位=0), or should I spend Phase 0 searching harder?  
   **Recommended**: Text-only. Saves weeks.

2. **Rollout length cap**: If T4 supports only 50 frames, is that acceptable?  
   **Recommended**: Yes. 50 vs 10-25 is already "long-horizon" in this literature.

3. **Negative result handling**: If H2 fails (adaptive = fixed), publish as negative?  
   **Recommended**: Yes, pre-registered. "Switching offers no benefit" is a valid contribution.

4. **CF/CF++ cross-recipe check**: Essential or nice-to-have?  
   **Recommended**: Nice-to-have. If Phase 0 finds checkpoints easily, do E5. Else skip.

5. **Manual artifact labeling**: Should I (the AI) label failure modes, or do you want to review?  
   **Recommended**: I label, you spot-check 10% for agreement. Saves time.

6. **Budget override**: Your CLAUDE.md says "3 hours per job." Some phases (E1, E3) may need 5-6 hours. Approve overrun?  
   **Recommended**: Yes, but split E1 into 3 jobs (step1, step2, step4) to stay under cap.

7. **Paper venue**: Are you targeting a conference (CVPR, ICCV, NeurIPS) or journal (TPAMI)? Affects writing style.  
**Recommended**: Conference (faster review). State your preference.

8. **Repository release**: Code + checkpoints public on GitHub at `j1s4nn/adaptive-granularity-wm`?  
   **Recommended**: Yes, reproducibility is a contribution. License: MIT (simpler than Apache 2.0).

9. **Human evaluation**: Paper quality? Interactive feel? Or skip (expensive)?  
   **Recommended**: Skip. FVD/LPIPS/VBench are accepted proxies in this subfield.

10. **When to stop**: If by Week 6 results are negative across all RQs, do we write up and submit, or pivot?  
**Recommended**: Write up. Negative results are pre-registered as informative. This subfield is moving fast鈥攕ubmitting early with negative results is better than delaying for a pivot that may be scooped.

---

## 10. MY RECOMMENDATIONS (What the Proposal Missed)

1. **Add error bars everywhere**: Your proposal mentions confidence intervals for RQ2, but apply to all metrics. Use 3 seeds minimum (you already have this), report 95% CI via bootstrap.

2. **Pre-register monotonicity check**: Proposal frames RQ4 as exploratory, but you should pre-register the finding either way ("if monotonic, threshold policy is justified; if non-monotonic, future work needs a smarter policy").

3. **Failure mode taxonomy**: Beyond aggregate FVD, categorize failures (drift, blur, color shift, disappearance, camera jump, seam). Helps future work.

4. **Latency breakdown**: First-frame vs second-frame latency (Causal-rCM reports both). Adaptive switching pays latency cost at mode change鈥攓uantify it.

5. **Switch cost metric**: Proposal mentions it but doesn't define it. I recommend: LPIPS spike at frame i where m_i 鈮?m_{i-1}. Average over all switches.

6. **Threshold sensitivity analysis**: Sweep 蟿_high 脳 蟿_low in E3. If quality is brittle to thresholds, controller won't generalize.

7. **Compute-matched comparison is critical**: Papers often compare methods with different FLOPs budgets. Your E3 must match C exactly, not approximately. E1 measures 魏(路) precisely.

8. **Qualitative comparison is not optional**: Reviewers will ask for visual examples. Generate side-by-side strips for at least 5 prompts 脳 3 systems.

9. **Related work refresh at Week 7**: This subfield drops 2-3 papers/week. Schedule a quick re-search at end of experiments to catch anything that supersedes your claims.

10. **Camera drift is a feature, not a bug**: Your proposal frames it as a "limitation" Causal-rCM reports. Reframe: it's evidence that frame-wise generation is unsafe beyond some horizon, which motivates your controller. Make it central to the story.

11. **Action control is deferrable, not critical**: If you have no action checkpoint, don't force it. The visual uncertainty signal (位=0) is already novel per your lit review鈥攏o reviewed method uses predictive uncertainty for K-switching. Action is an extension.

12. **Repository structure from day 1**: Don't refactor later. The tree in 搂1 is the actual code structure. Every .py file starts with a docstring showing its INPUT/OUTPUT tree.

---

## 11. RESEARCH INTEGRITY CHECKLIST

- [ ] Every number in a table comes from a raw JSONL file (no hand-typed estimates)
- [ ] Every figure generated by a script in `figures_src/` (no manual Photoshop)
- [ ] meta.json saved for every run (checkpoint sha256, git commit, status)
- [ ] Failed runs kept in results/ (not deleted)
- [ ] Negative results reported in DEVIATIONS.md
- [ ] CLAIMS.md maps every paper claim to its evidence file
- [ ] Pre-registered decision rules in PREREG.md (written before results exist)
- [ ] Multiple seeds (3 minimum) for all experiments
- [ ] Confidence intervals on all metrics
- [ ] Paired tests where applicable (E3: adaptive vs fixed on same prompts)
- [ ] Effect sizes reported (not just p-values)
- [ ] Novelty boundary clear: "no reviewed method uses uncertainty for K-switching" (your claim boundary table, proposal page 6)
- [ ] Fresh literature re-check before writing (Week 7)

---

## 12. GPU-HOURS ESTIMATE & KAGGLE QUOTA (3-Day Execution)

| Phase | Task | GPU-hours (T4) | Wall-clock (with parallelization) |
|-------|------|----------------|------------------------------------|
| Phase 0 | Feasibility | 1 | 1 hour (Day 1, Hour 3) |
| E1 | Fixed modes: 4 configs 脳 300 rollouts 脳 50 frames | 20 | 5 hours (4 jobs parallel, Day 1 Hours 3-8) |
| E2 | Uncertainty scoring (reuses E1 frames, compute-only) | 10 | 10 hours (Day 1 Hour 10 鈫?Day 2 Hour 1) |
| E3 | Adaptive controller: 300 rollouts 脳 50 frames | 25 | 12 hours (Day 2 Hours 2-14, parallel with E4) |
| E4 | Recovery sweep: 4 values of R 脳 100 rollouts | 12 | 6 hours (Day 2 Hours 2-8, parallel with E3) |
| E5 | Baselines + CF/CF++: 5 systems 脳 100 rollouts | 18 | 9 hours (Day 2 Hours 4-13) |
| **Total** | | **86 GPU-hours** | **~60 wall-clock hours** (spread over 3 days) |

**3-Day Quota Strategy**:
- Kaggle free tier: 2脳T4 GPUs, 30 hours/week quota
- **BUT**: You can submit multiple jobs in parallel (each gets its own 2脳T4 allocation)
- Day 1: 4 E1 jobs (20 GPU-hours) + Phase 0 (1 hour) = 21 hours consumed in 5 wall-clock hours
- Day 2: E3+E4 parallel (37 GPU-hours) + E5 (18 hours) = 55 hours consumed in 12 wall-clock hours
- Day 3: Analysis only (no GPU)
- **Total quota needed**: 76 GPU-hours in 3 days = 25 hours/day if sequential, but **parallelization keeps us under 30h/week** by spreading across multiple concurrent jobs

**Quota hack**: Kaggle allows multiple notebooks running simultaneously. Submit 4-6 jobs at once, each consumes from the same weekly quota but runs in parallel. Total wall-clock time drops from 86 hours 鈫?~60 hours.

**Fallback**: If quota exhausts Day 2, write paper with partial results (E1+E2 already sufficient for baseline+signal comparison). E3-E5 can run Day 3 if quota resets.

---

## 13. AGGRESSIVE 3-DAY TIMELINE (No Compromises, Maximum Parallelization)

### Day 1 (12 hours): Setup + Launch All Baseline Jobs
**Hours 1-2**: Local setup
- Clone NVlabs/rcm, create project tree, implement `src/` packages
- Write all job scripts (Phase 0, E1脳4 configs, E2, E3, E4, E5)
- Create `j1s4nn/adaptive-granularity-wm` GitHub repo, push initial commit

**Hours 3-4**: Submit Phase 0 + E1 jobs (parallel)
- Phase 0: 1 GPU-hour (runs while you work)
- E1: Submit 4 jobs in parallel (c1-1 S=1/2/4, c3-3 S=4) 鈫?20 GPU-hours total, but 5 hours wall-clock with 4脳 parallelization
- Monitor: `push_and_wait.py` polls every 10 minutes, downloads results automatically

**Hours 5-8**: Implement E2-E5 while E1 runs
- Code `src/uncertainty/` (EMA, residual, warp scores)
- Code `src/controller/` (threshold, recovery policies)
- Code `src/baselines/` (motion, complexity)
- Code `src/evaluation/` (FVD, LPIPS, AUROC, switch cost)
- Write `figures_src/*.py` scripts (matplotlib templates ready)

**Hours 9-10**: E1 results arrive, launch E2
- Download E1 raw JSONL (auto-downloaded by push_and_wait)
- Generate `fig_error_curves.pdf`, `tab_baseline.tex` (automated)
- Compute coarse-unsafe labels (螖 threshold from E1 data)
- Submit E2: Uncertainty scoring on E1 rollouts (10 GPU-hours, starts at Hour 10)

**Hours 11-12**: Paper outline + E3 prep
- Write paper skeleton (intro, method, results templates)
- Calibrate 蟿_high/蟿_low from E1 baselines (estimate from error curves)
- Prepare E3 job scripts (ready to submit when E2 completes)
- **End of Day 1**: E2 running overnight

---

### Day 2 (12 hours): Core Experiments + Results Processing
**Hours 1-2**: E2 results + launch E3/E4 (parallel)
- Download E2 ROC data, generate `fig_roc.pdf`, `tab_signal_auroc.tex`
- Final calibration of 蟿_high/蟿_low from E2 AUROC
- Submit E3 (adaptive controller, 25 GPU-hours) + E4 (recovery sweep, 12 GPU-hours) **in parallel**
- Both jobs run simultaneously: 25 GPU-hour wall-clock with 2脳T4

**Hours 3-6**: E5 baseline implementations + launch
- While E3/E4 run, implement motion/complexity controllers
- Search for CF/CF++ checkpoints (Hugging Face, 30 minutes)
- Submit E5 jobs: Motion baseline, Complexity baseline, CF/CF++ transfer (if found)
- E5: 18 GPU-hours, submit at Hour 4, completes by Hour 22 (Day 2 evening)

**Hours 7-9**: E3/E4 results arrive, generate figures
- Download E3/E4 raw data (arrives around Hour 7-8)
- Generate core figures:
  - `fig_quality_latency_compute.pdf` (E3)
  - `fig_mode_trace.pdf` (E4)
  - `fig_qualitative.pdf` (E3/E4 video frames)
  - `fig_switch_cost.pdf` (E4)
- Update `tab_controller_vs_fixed.tex`, `tab_recovery.tex`

**Hours 10-12**: RQ4 analysis + long-horizon rerun
- Reanalyze E3 data for RQ4 (monotonicity): bin by uncertainty quartile
- Generate `fig_monotonicity.pdf`
- **If time allows**: Submit long-horizon extension (100-frame rollouts, 5 prompts) 鈫?runs overnight
- Start filling paper Results section with numbers from E3/E4
- **End of Day 2**: E5 completes overnight, long-horizon (optional) runs

---

### Day 3 (12 hours): Final Experiments + Paper Writing
**Hours 1-3**: E5 results + complete figure set
- Download E5 data (motion/complexity/CF++ transfer)
- Generate `tab_full_comparison.tex`, `fig_threshold_sweep.pdf`, `fig_recovery_sweep.pdf`
- If long-horizon ran overnight, generate `fig_long_horizon_curves.pdf`, `fig_drift_strips.pdf`
- Complete `fig_failure_gallery.pdf` (worst-case frames from all experiments)
- Generate `coverage_matrix.tex` (automated: RQ 鈫?data 鈫?figure mapping)

**Hours 4-6**: Finalize all tables and ablations
- Run ablation analysis scripts on E3/E4 data (threshold sweep, hysteresis band, signal components)
- Generate `tab_ablations.tex`
- Update CLAIMS.md: Mark each hypothesis SUPPORTED/WEAKENED/UNSUPPORTED with evidence files
- Update DEVIATIONS.md: Document what changed from proposal (text-only, T4 limits, etc.)

**Hours 7-10**: Write paper (Results, Discussion, Conclusion)
- Fill Results section with all numbers, cite every figure/table
- Write Discussion: What worked, what didn't, why
- Write Limitations section (text-only, 50-frame cap, T4 hardware)
- Write Conclusion
- Polish Introduction (now that results are known)
- Polish Related Work (add any papers from Week 7 re-search)

**Hours 11-12**: Final checks + submission prep
- Run research integrity checklist (搂11): Every number traced to raw file
- Validate PREREG.md decision rules against actual results
- Generate camera-ready PDFs: `paper.pdf`, `supplement.pdf`
- Push final commit to `j1s4nn/adaptive-granularity-wm`
- Create release: tag `v1.0-submission`, attach checkpoints/configs
- **PAPER COMPLETE**: Ready for submission or professor review

---

### Parallel Execution Gantt Chart

```
Day 1:
Hour  1 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?      Setup + Code Implementation
Hour  2 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
Hour  3 鈹屸攢 Phase 0 (1h GPU) 鈹€鈹?Hour  4 鈹?                    鈹溾攢 E1 Job 1 (c1-1 S=1, 5h GPU) 鈹€鈹€鈹?Hour  5 鈹?                    鈹溾攢 E1 Job 2 (c1-1 S=2, 5h GPU) 鈹€鈹€鈹?Hour  6 鈹?                    鈹溾攢 E1 Job 3 (c1-1 S=4, 5h GPU) 鈹€鈹€鈹?Hour  7 鈹?                    鈹溾攢 E1 Job 4 (c3-3 S=4, 5h GPU) 鈹€鈹€鈹?Hour  8 鈹? (while coding)     鈹?                                鈹?Hour  9 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                                鈹?Hour 10                         鈹斺攢 E1 complete, E2 starts (10h) 鈹?Hour 11                                                    鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
Hour 12 (E2 runs overnight) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€>

Day 2:
Hour  1 鈫愨攢 E2 completes 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?Hour  2     鈹屸攢 E3 (adaptive, 25h GPU) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?Hour  3     鈹溾攢 E4 (recovery, 12h GPU) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?       鈹?Hour  4     鈹? 鈹屸攢 E5 (baselines, 18h GPU) 鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?Hour  5     鈹? 鈹?                               鈹?       鈹?Hour  6     鈹? 鈹?                               鈹?       鈹?Hour  7     鈹? 鈹?                       E4 done 鈹?       鈹?Hour  8     鈹? 鈹?                               鈹?       鈹?Hour  9     鈹? 鈹?                               鈹?  E3 done
Hour 10     鈹? 鈹? E5 continues                  鈹?Hour 11     鈹? 鈹?                               鈹?Hour 12     鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?          (E5 completes overnight)

Day 3:
Hour  1 鈫愨攢 E5 done 鈹€鈹?Hour  2     Figure generation
Hour  3     
Hour  4     Ablations + Tables
Hour  5     
Hour  6     CLAIMS.md + DEVIATIONS.md
Hour  7     Write Results
Hour  8     Write Discussion
Hour  9     Write Conclusion
Hour 10     Polish Intro/Related
Hour 11     Final checks
Hour 12     鉁?PAPER COMPLETE
```

**Total GPU-hours consumed**: 86 hours  
**Total wall-clock time**: ~60 hours (thanks to parallelization)  
**Kaggle quota usage**: Acceptable (submit 4-6 jobs in parallel, each uses 2脳T4)

**Key to speed**:
1. **No sequential dependencies**: E3/E4/E5 can start as soon as E1/E2 data exists
2. **Automated figure generation**: Every `figures_src/*.py` script runs during job execution, generates PDFs automatically
3. **No manual data processing**: All metrics computed in jobs, downloaded as final numbers
4. **Paper written in parallel**: Outline Day 1, fill Results as data arrives Days 2-3
5. **No quality compromise**: Every experiment from proposal runs; ablations complete; all figures generated

---

## 14. FINAL CHECKLIST (Before "Approved")

I will proceed to Phase 1 only after you confirm:

- [ ] **Text-only baseline (位=0) is acceptable**: No action checkpoint = no control term. Future work.
- [ ] **50-frame cap is acceptable**: If T4 limits long-horizon to 50 frames (vs aspirational 100-150).
- [ ] **Negative results are publishable**: If adaptive = fixed, we write it up as "switching offers no benefit."
- [ ] **Kaggle quota understood**: 86 GPU-hours over 6 weeks = 14/week average, fits in 30h/week quota.
- [ ] **Phase 0 approval workflow**: I run Phase 0, report findings (VRAM, internals, action), you approve/reject before E1.
- [ ] **Cross-recipe E5 is optional**: If CF/CF++ checkpoints not found, skip and mark future work.
- [ ] **Repository will be public**: Code + configs released under MIT/Apache 2.0 license (your choice).
- [ ] **You will review**: Error curves (Week 2), ROC + thresholds (Week 4), figures (Week 7), outline (Week 8).

---

## YOUR DECISION LIST (From Proposal 搂7, Now Actionable)

1. **Action checkpoint handling**: Text-only (位=0) or search harder?  
   鈫?**Recommended: Text-only**

2. **Rollout length cap**: 50 frames OK if T4 can't do 100?  
   鈫?**Recommended: Yes**

3. **Negative result policy**: Publish if H2/H3 fail?  
   鈫?**Recommended: Yes (pre-registered)**

4. **CF/CF++ cross-recipe**: Essential or optional?  
   鈫?**Recommended: Optional**

5. **Threshold sensitivity**: Sweep 蟿 in E3?  
   鈫?**Recommended: Yes (ablation table)**

6. **Failure mode taxonomy**: Manual labeling?  
   鈫?**Recommended: I label, you spot-check**

7. **Repository license**: MIT or Apache 2.0?  
   鈫?**Recommended: MIT**

8. **Paper venue**: Conference (CVPR/NeurIPS) or journal?  
   鈫?**Your call (affects writing style)**

9. **Human evaluation**: Skip?  
   鈫?**Recommended: Skip (FVD/VBench sufficient)**

10. **Week 7 lit re-check**: Confirm?  
鈫?**Recommended: Yes (fast-moving field)**

---

# END OF MASTER PLAN

**Next step**: You review this plan. Reply "approved" to start **immediate execution** (Day 1 Hour 1), or ask questions/request changes.

**What I need from you now**:
1. Approve/modify the 10 decisions in 搂14 (action=text-only, rollout=50, negatives=publish, etc.)
2. Confirm paper venue: Conference (CVPR/ICCV/NeurIPS) or arXiv preprint first?
3. GitHub repo name confirmation: `j1s4nn/adaptive-granularity-wm` or different?
4. Any hard deadline? (e.g., "paper must be submitted by [date]")

Once approved, I will **immediately**:
1. Create GitHub repo `j1s4nn/adaptive-granularity-wm` with MIT license
2. Implement all `src/` packages (6 Python packages, ~2000 lines)
3. Write all job scripts (Phase 0 + E1-E5, ~800 lines)
4. Submit Phase 0 + 4 E1 jobs in parallel to Kaggle
5. Generate paper outline (LaTeX template with TikZ figures)
6. **Report Phase 0 + E1 results in ~6 hours** (Hour 10 of Day 1)

**Estimated time to complete paper**: 72 hours (3 days) from approval, with all experiments completed and figures generated. No compromises, no shortcuts鈥攆ull proposal execution at maximum speed.


