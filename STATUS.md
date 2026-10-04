# STATUS: Day 1 (real state, corrected — previous STATUS.md was over-optimistic)

**Last updated**: 2026-10-04 17:32 local

## Phase 0b: COMPLETE — VERDICT: PASS

Kernel `ajjisan/phase-0b-minimal-probe-checkpoints-only` v3 (single-dataset, self-contained script).
Results: `results/phase0/phase0b_minimal_v3/results/phase0/20261004_092703_meta.json`

Findings:
- **Env**: torch 2.11.0+cu128, Python 3.13.15, Tesla T4 (14.56 GB), FA2 unavailable (expected),
  PyTorch SDPA available → use SDPA fallback, fp16 (no bf16 on T4)
- **c3-3_step4**: loads OK, 72.8 s, 2706.8 MB, peak VRAM 2.64 GB
- **c1-1_step2**: loads OK, 71.8 s, 2706.8 MB, peak VRAM 2.64 GB
- **Checkpoint structure**: plain Wan2.1 DiT state_dict (keys `net.patch_embedding`, `net.blocks.0-29.*`,
  self_attn + cross_attn + ffn + modulations, `net.head.*`). Matches Wan2.1-1.3B T2V.
- **NO EMA weights in either checkpoint** → u_i^EMA DROPPED per pre-registered plan;
  uncertainty = residual + warp scores only (update E2 accordingly)
- Dataset mount path pattern: `/kaggle/input/datasets/ajjisan/<dataset>/<subfolder>/...`
  → ALWAYS search files recursively, never assume a fixed mount path

## Lessons learned (Kaggle plumbing)
1. Kaggle script kernels upload ONLY the code file — bundling src/ does NOT work.
   All job scripts must be SELF-CONTAINED single files.
2. `kaggle kernels push -p <folder>`: id/title must slugify consistently or you get
   409 Conflict. Match id to the title-derived slug exactly for version updates.
3. Attaching the 11 GB T5 + 5.7 GB DiT datasets made a kernel hang 70+ min;
   single-dataset (ckpts only, ~6 GB) runs in ~4 min. Attach only what a job needs,
   or budget for long mounts.
4. `kaggle kernels delete <slug>` (confirm with `echo yes |`) kills a stuck session.
5. CLI 2.x status strings: `KernelWorkerStatus.RUNNING/COMPLETE/ERROR` — push_and_wait.py fixed.

## VRAM arithmetic for Phase 0c (REAL RISK)
T5-XXL encoder bf16 = 11.36 GB; model 2.64 GB; VAE 0.5 GB → 14.5 GB > 14.56 total.
Mitigation options: encode text on CPU then unload T5, or int8 T5, or batch=1 fp16
with VAE offloaded. Phase 0c must measure this exactly.

## Next: Phase 0c — real inference test (rcm repo integration)
- Clone https://github.com/NVlabs/rcm ON KAGGLE (code only; no HF model downloads — checkpoints
  come from our datasets)
- Repo expects FA2 + transformer-engine + Python 3.12 → on Kaggle we get torch 2.11/3.13:
  patch attention to SDPA, disable TE, fp16
- Target: one 10-frame rollout per checkpoint, measure VRAM + latency, save 3 keyframes
- `src/causal_rcm/inference.py` placeholders must be rewritten against the actual repo
  after we see how generation works

## What failed before (fixed today)
1. Both old phase0 kernels errored in ~7s: `ModuleNotFoundError: No module named 'src'`
   (src/ not uploaded — script kernels take only the code file)
2. First relaunch hung 70+ min (3 datasets attached → huge mount copy), killed via delete
3. v2 probe: mount-path guessing failed (files nested under `datasets/ajjisan/...`),
   plus a verdict f-string bug on empty lists → both fixed in v3

## Environment facts (verified today)
- Kaggle CLI 2.2.4 (pip, Python 3.14, `C:\Users\jisan\AppData\Roaming\Python\Python314\Scripts`)
- Auth: `~/.kaggle/access_token` (KGAT). Works against account **ajjisan**.
- `~/.kaggle/credentials.json.j1s4nn` — DO NOT USE (user wants ajjisan only)
- Git: no remote configured yet (repo exists at github.com/j1s4nn/adaptive-granularity-wm)

## Dataset contents (verified)
- `ajjisan/causal-rcm-ckpts`: c1-1_step2/step4, c3-3_step2, c3-3_step2_noisy_ctx,
  c3-3_step4, TF-sCM c3-3_step4, umT5_wan_negative_emb.pt (no c1-1_step1)
- `ajjisan/wan21-t5`: models_t5_umt5-xxl-enc-bf16.pth (11.36 GB) + google/umt5-xxl tokenizer
- `ajjisan/wan21-dit-vae`: Wan2.1_VAE.pth (507 MB), diffusion_pytorch_model.safetensors (5.67 GB), config.json

## Hard constraints (CLAUDE.md)
- Kaggle account ajjisan; never read/print/commit credential file contents
- No HF model downloads inside jobs; no HF downloads on this PC
- Jobs ≤3h unless split; resumable (append results.jsonl, skip finished)
- All three datasets must be listed in every kernel-metadata.json (if attached)
- T4 only, fp16 only
