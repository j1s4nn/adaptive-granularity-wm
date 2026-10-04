# STATUS: Day 1 (real state, corrected — previous STATUS.md was over-optimistic)

**Last updated**: 2026-10-04 16:58 local

## Where the workflow actually is

**Phase 0b (checkpoint + VRAM probe) — SUBMITTED AND RUNNING on Kaggle.**
- Kernel: `ajjisan/phase-0b-checkpoint-vram-probe-t4` (v1), T4, 30-min timeout requested
- Pushed 16:15, still `KernelWorkerStatus.RUNNING` at 16:55 (40+ min)
- Likely cause of slowness: ~24 GB dataset copy to session disk + 3x 2.8 GB checkpoint torch.load
- No results downloaded yet (logs unavailable until run ends)

## What failed before (fixed today)
1. Both old phase0 kernels (`adaptive-granularity-phase-0-feasibility`,
   `phase-0-feasibility-probe-causal-rcm-on-t4`) ERRORED in ~7s:
   `ModuleNotFoundError: No module named 'src'` — src/ was not uploaded with the kernel.
   Fix: bundle `src/` into the job folder; `kaggle kernels push -p <folder>` uploads it.
2. `load_checkpoints.py` assumed files at dataset root; real datasets nest files in
   subfolders (`rcm_ckpts/...`, `wan21_t5/...`, `wan21_dit_vae/...`).
   Fix: recursive search (`find_file_recursive`), plus `weights_only` fallback in torch.load.
3. `push_and_wait.py` compared status against CLI 1.x strings; CLI 2.2.4 returns
   `KernelWorkerStatus.X`. Fix: substring matching + parse real slug from push URL.

## Environment facts (verified today)
- Kaggle CLI 2.2.4 installed via pip into Python 3.14 (user Scripts dir: `C:\Users\jisan\AppData\Roaming\Python\Python314\Scripts`)
- Auth: `~/.kaggle/access_token` (KGAT token). CLI works: list/status/output/push verified against account **ajjisan**
- `~/.kaggle/credentials.json.j1s4nn` exists but MUST NOT be used (user wants ajjisan only)
- Git: 5 commits locally, **NO remote configured yet** (needs `git remote add origin https://github.com/j1s4nn/adaptive-granularity-wm`)

## Dataset contents (verified via `kaggle datasets files`)
- `ajjisan/causal-rcm-ckpts`: c1-1_step2.pt, c1-1_step4.pt (2.84 GB each) + c3-3_step2.pt,
  c3-3_step2_noisy_ctx.pt, c3-3_step4.pt, TF-sCM c3-3_step4.pt, umT5_wan_negative_emb.pt
  **NOTE: no c1-1_step1 → E1 must use c1-1 step2/step4 + c3-3 step4**
- `ajjisan/wan21-t5`: models_t5_umt5-xxl-enc-bf16.pth (11.36 GB) + google/umt5-xxl tokenizer files
- `ajjisan/wan21-dit-vae`: Wan2.1_VAE.pth (507 MB), diffusion_pytorch_model.safetensors (5.67 GB), config.json

## Next steps
1. Phase 0b finishes → download meta.json → report VRAM/EMA/verdict
2. **Phase 0c**: clone NVlabs/rcm on Kaggle, real 10-frame generation, SDPA fallback,
   fp16 (NOT bf16 on T4) — this is the true go/no-go for E1. `src/causal_rcm/inference.py`
   is currently ALL PLACEHOLDERS and must be implemented against the rcm repo.
3. E1 baseline: 4 parallel jobs → c1-1 step2, c1-1 step4, c3-3 step4 (+c3-3 step2 for
   matched-compute). Prompts: 100 from VBench-T2V, 3 seeds, 10/25/50 frames.
4. E2 → E3/E4 → E5 per MASTER_PLAN.

## Hard constraints (CLAUDE.md)
- Kaggle account ajjisan; never read/print/commit credential file contents
- No HF downloads inside jobs; no HF downloads on this PC
- Jobs ≤3h unless split; resumable (append results.jsonl, skip finished)
- All three datasets must be listed in every kernel-metadata.json
- T4 only, fp16 only
