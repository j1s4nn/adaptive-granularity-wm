# Project: uncertainty-guided adaptive temporal granularity (Causal-rCM, Wan2.1-1.3B)

## Rules
- Kaggle account: ajjisan. Kaggle runs are disposable; local files are the source of truth.
- Private datasets: ajjisan/causal-rcm-ckpts, ajjisan/wan21-t5, ajjisan/wan21-dit-vae.
  Every Kaggle job must list all three under dataset_sources in kernel-metadata.json.
- Find real mount paths with find or glob under /kaggle/input; never hardcode them. Print them in the logs.
- Never download model files from Hugging Face inside experiment jobs, and never download them to this PC.
- Never read, print, or commit Kaggle credentials (~/.kaggle).
- Kaggle scripts must be resumable: append results to /kaggle/working/results.jsonl, skip finished items.
- Wrap main work in try/except; on error save partial results and exit cleanly.
- Keep each job under 3 hours. Print detailed logs (versions, file lists, VRAM, errors).
- Wait for jobs with scripts/push_and_wait.py (no manual polling). Download only small outputs to results/<job>/, then git commit.
- Check kaggle kernels push --help for the GPU accelerator option and pick T4.

## Known facts and open risks
- Checkpoints: NVlabs/rcm repo, worstcoder/rcm-Wan files. Matched pair: c1-1_step4 and c3-3_step4.
- Hardware is Kaggle T4 (Turing, 16 GB). Repo setup expects FlashAttention-2, Transformer Engine, Python 3.12.
  These may not work on T4. Look for a fallback to PyTorch SDPA. Use fp16, not bf16.
- No action-conditioned or camera checkpoint was found for Wan2.1-1.3B. The delta-a control term is unresolved.
  Do not invent a workaround without asking me.

## Current phase
Phase 0a only (environment and code probe, no model load). Do not start Phase 0b or any experiment until I approve.
