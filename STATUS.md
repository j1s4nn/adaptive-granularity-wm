# STATUS: Phase 1a/1b complete 鈥?operational plan v2 locked

**Last updated**: 2026-10-05

## Current execution state

The original three-day/86-GPU-hour plan is superseded by the Operational Master Plan v2 at the top of `MASTER_PLAN.md`. The remaining paper will focus on the proposal's core contribution: uncertainty-guided switching between matched Causal-rCM c1-1 and c3-3 modes, with recovery and hysteresis. The verified setup is text-to-video, so `lambda_action=0` and action conditioning is a documented future extension.

Phase 0 (planning and feasibility) is complete. **Phase 1a/1b is now complete; the next phase is the fixed-mode baseline.**

## Phase 1a/1b audit - PASS with stated limitation

| Criterion | Result | Evidence |
|---|---|---|
| Real Phase 0c-based runner and nested dataset discovery | PASS | `jobs/phase1_switch/phase1_switch.py`; v2 metadata |
| c1-1 -> c3-3 -> c1-1 real continuation | PASS | v2 metadata and log; all stages complete |
| Boundary latent transfer | PASS | both boundary diagnostics are 0.0 |
| Keyframes and combined decode | PASS | v2 run `frames/`; decode 53.030 s |
| KV-state persistence | UNAVAILABLE | API exposes no KV cache; checkpoint reload is measured |
| Quality superiority/seam quality | NOT CLAIMED | feasibility smoke test only |

 **Do not repeat Phase 0.** Phase 1a/1b is complete; the next phase is the fixed-mode baseline.

## Phase 0 audit

| Item | Result | Evidence |
|---|---|---|
| c1-1/c3-3 checkpoint loading | PASS | `results/phase0/phase0b_minimal_v3/.../20261004_092703_meta.json` |
| Real c1-1 inference on T4 | PASS | `results/phase0/phase0c_v7/.../frames/` |
| Real c3-3 inference on T4 | PASS | `results/phase0/phase0c_v7/.../frames/` |
| fp16 + SDPA fallback | PASS | Phase 0c logs |
| EMA weights | UNAVAILABLE | Phase 0b metadata; remove `u_ema` |
| Action-conditioned interface | UNVERIFIED/UNAVAILABLE | No action checkpoint in verified datasets |
| Runtime c1-1鈫攃3-3 switching | VERIFIED | Phase 1b smoke test; quality remains for later phases |
| Real E1 metrics | NOT STARTED | Current E1 script contains placeholders |

## Proposal coverage after audit

| Proposal component | Status |
|---|---|
| RQ1: uncertainty vs motion/complexity | Planned with residual/warp only |
| RQ2: adaptive fine/coarse switching | Core; blocked only by switch smoke test |
| RQ3: recovery + hysteresis | Planned |
| RQ4: uncertainty鈥揵enefit relationship | Exploratory reanalysis planned |
| Action-augmented potential | Deferred; `lambda=0` |
| Camera-controlled world model | Deferred; verified backbone is text-to-video |
| E5 CF/CF++ transfer | Optional and cannot block the paper |

## Phase 1 acceptance criteria

1. The runner is based on the proven Phase 0c recipe, not the dummy `src` wrapper.
2. A real switch sequence either succeeds with aligned context or produces a reproducible failure report.
3. No random frames or placeholder metrics remain in the submitted E1 path.
4. Kaggle outputs are resumable, small, and linked here.

After these criteria are met, stop and audit before Phase 2.

# Historical status below

**Last updated**: 2026-10-04 19:55 local

## Phase 0c: COMPLETE 鈥?VERDICT: PASS (real inference works on T4)

Kernel `ajjisan/phase-0c-real-causal-rcm-inference-test-on-t4` v7.
Meta: `results/phase0/phase0c_v7/results/phase0c/20261004_114113/20261004_114113_meta.json`
Keyframes: `results/phase0/phase0c_v7/results/phase0c/20261004_114113/frames/*.png`

Measured on T4 (fp16, SDPA, 480p 16:9):
- **c1-1 frame-wise 1-step** (c1-1_step2 ckpt, steps_per_chunk "4 1"): DiT 4.5 s,
  peak VRAM 4.05 GB, VAE decode 25.5 s (13 frames) 鈥?OK
- **c3-3 chunk-wise 4-step** (c3-3_step4 ckpt, chunks [3,3]): DiT 14.3 s,
  peak VRAM 4.73 GB, VAE decode 46.6 s (21 frames) 鈥?OK
- Env: torch 2.11.0+cu128, Python 3.13.15, Tesla T4, SDPA (no FA2), fp16 everywhere

## WORKING RECIPE (the exact procedure that works 鈥?reuse for E1-E5)
1. Script kernels are SINGLE-FILE only (no src/ bundling). All job scripts self-contained.
2. pip deps: einops transformers sentencepiece ftfy regex imageio imageio-ffmpeg tqdm
   torchvision fvcore omegaconf attrs pyyaml iopath termcolor pynvml pandas loguru
   safetensors cloudpickle dill matplotlib packaging
3. `git clone --depth 1 https://github.com/NVlabs/rcm.git /kaggle/working/rcm`
4. Patch rcm/inference/wan2pt1_t2v_causal_infer.py: TENSOR_KWARGS dtype bf16 -> fp16,
   `.to(dtype=torch.bfloat16)` -> fp16 (2 places).
5. Patch rcm/utils/umt5.py: default dtype bf16 -> fp16; tokenizer_path -> LOCAL dataset dir
   `/kaggle/input/datasets/ajjisan/wan21-t5/wan21_t5/google/umt5-xxl`.
6. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` BEFORE torch import
   (fixes VAE-decode OOM from fragmentation) and `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`.
7. DiT: `init_weights_on_device()` -> instantiate 1.3B WanModel -> `load_dit_weights`
   (strips net. prefix, assign=True) -> `.to(cuda, fp16).cpu()` until sampling.
8. T5 (the tricky part 鈥?umT5-XXL bf16 is 11.4 GB):
   - build `umt5_xxl(encoder_only=True)` normally on CPU (fp32 transient ~22 GB RAM is OK),
     then `.to(fp16)` (11.2 GB RAM)
   - `easy_io.load(t5_path, file_format="pt", map_location="cuda", weights_only=False)`
   - cast every tensor to fp16 IN PLACE on GPU, `load_state_dict(strict=False, assign=True)`
     (params become GPU tensors with no copy)
   - encode prompt, then `del model + empty_cache`
9. Sampling: build t_steps (sigma_max=1600, mid_t 15/16 5/6 5/8), per-chunk schedules,
   `causal_rollout_sampling(net, noise, t_steps, t_steps_per_chunk, steps_per_chunk,
   condition, None, 1.0, first_chunk_t, chunk_t, ode=False, generator=...)`.
10. VAE decode AFTER `del net + empty_cache`.
11. Configs verified:
    - c1-1 1-step: ckpt c1-1_step2, steps_per_chunk=[4,1], mid_t_schedules "15/16,5/6,5/8;"
    - c1-1 2-step: ckpt c1-1_step2, steps_per_chunk=[4,2], "15/16,5/6,5/8;5/6"
    - c1-1 4-step: ckpt c1-1_step4, num_steps=4, mid_t "15/16 5/6 5/8"
    - c3-3 4-step: ckpt c3-3_step4, first_chunk_t=3, chunk_t=3, num_steps=4, mid_t "15/16 5/6 5/8"
12. Latent frames T = 1 + (num_pixels-1)//4; for c3-3 need (T-3) % 3 == 0
    (e.g. num_frames 21 -> T=6; for 50 frames T=13 -> (13-3)%3=1 INVALID!
    c3-3 needs num_frames s.t. T-3 divisible by 3: T in {6,9,12,15,...} ->
    num_frames in {21,33,45,57,...}; for E1 use 10/25/50 c1-1 and matched ~21/33/45 c3-3,
    or trim frames post-hoc for metric alignment)

## Phase 0b (earlier today): PASS
c1-1_step2 + c3-3_step4 load (2.71 GB each, 72 s), NO EMA weights (u_i^EMA dropped),
Wan2.1 DiT state_dict structure confirmed. `results/phase0/phase0b_minimal_v3/...`

## Phase 0c failure history (all fixed, for the record)
v1 fvcore missing -> v2 loguru -> v3 GPU OOM (T5 built fp32 on GPU) -> v4 RAM OOM
("Killed": 11.4 GB ckpt + 11.2 GB model on CPU) -> v5 meta-device .to() clash ->
v6: generation WORKED; VAE decode OOM (fragmentation) + frame-shape bug -> v7 PASS
(expandable_segments + batch-squeeze fix)

## Next: E1 fixed-mode baseline (needs user approval before submitting)
4 parallel T4 jobs: c1-1 S=1, c1-1 S=2, c1-1 S=4, c3-3 S=4; 100 VBench prompts x 3 seeds x
10/25/50 frames (c3-3 frame counts adjusted for chunk divisibility).
Prompts file: rcm repo already ships `evaluation/vbench_text2video/prompts.json` (140 prompts).

## Kaggle plumbing (verified today)
- CLI 2.2.4, account ajjisan, KGAT access_token auth
- Script kernels upload ONLY the code file; id must slugify to title (else 409)
- `kaggle kernels delete <slug>` (echo yes |) kills stuck sessions
- Status strings: KernelWorkerStatus.{RUNNING,COMPLETE,ERROR}
- Mount path: /kaggle/input/datasets/ajjisan/<dataset>/... (search recursively)
- Single-dataset mounts ~2 min; 3-dataset mounts can be slow (10+ min)




