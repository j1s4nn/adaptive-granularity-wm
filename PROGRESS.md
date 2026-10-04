## Day 1 Progress Report (Hour 1.5)

**✓ Completed:**
1. Git repo initialized with j1s4nn credentials
2. Project structure created (6 Python packages)
3. Core packages implemented:
   - `src/causal_rcm/`: Checkpoint loading, inference wrapper, T4 fallback
   - `src/uncertainty/`: EMA, residual, warp scores + composite aggregator
4. Phase 0 probe script written (feasibility check)
5. Phase 0 **SUBMITTED TO KAGGLE** and running
   - URL: https://www.kaggle.com/code/ajjisan/adaptive-granularity-phase-0-feasibility
   - Status: Running
   - Expected: ~1 hour GPU time

**Next (while Phase 0 runs):**
- Implement `src/controller/` (threshold policy + recovery)
- Implement `src/baselines/` (motion, complexity controllers)
- Implement `src/evaluation/` (FVD, LPIPS, AUROC metrics)
- Write E1 job scripts (4 parallel jobs for all fixed modes)
- Prepare E2-E5 job scripts

**Phase 0 will report back:**
- VRAM usage (c1-1 vs c3-3)
- Max rollout length on T4
- FlashAttention-2 availability (expected: NO, fallback to SDPA)
- Action checkpoint presence (expected: NO, proceed λ=0)

**Timeline on track:** Day 1, Hour 2 of 36 total hours available.
