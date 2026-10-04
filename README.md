# Uncertainty-Guided Adaptive Temporal Granularity for Video World Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Research by**: Hossen Md Jisan (NUIST, 4th year AI)  
**Paper**: [Preprint coming soon]  
**Code**: Full reproducibility with frozen Causal-rCM checkpoints

---

## Overview

Interactive video world models must generate frames causally at low latency. The Causal-rCM family achieves real-time generation in two fixed regimes:
- **Frame-wise (K=1)**: Fast (0.27s first-frame), but prone to camera drift in long rollouts
- **Chunk-wise (K=3)**: Stable, but slower (0.60s first-frame) and coarser response

**This work asks**: Can we switch between fine/coarse generation *at run time* based on the model's own predictive uncertainty, and does a recovery+hysteresis mechanism control long-horizon error?

### Key Contributions

1. **First uncertainty-guided run-time granularity controller** for AR video diffusion (no prior work uses predictive uncertainty for K-switching)
2. **Recovery+hysteresis mechanism** that reacts to uncertainty spikes and re-stabilizes
3. **Long-horizon characterization** (10/25/50-frame rollouts) — error-vs-length curves missing from literature
4. **Complete experimental protocol** with matched-compute comparisons, AUROC signal evaluation, and ablations

---

## Project Structure

```
04_Experiment/
├── src/                    # Python packages (uncertainty, controller, baselines, evaluation)
├── jobs/                   # Kaggle kernel scripts (Phase 0, E1-E5)
├── results/                # Local experiment outputs (small files only)
├── docs/figures/           # Publication-ready PDFs (generated from raw data)
├── config/                 # Hyperparameters, thresholds
├── scripts/                # Automation (push_and_wait.py, make_tree.py)
├── MASTER_PLAN.md          # Full experimental design (19k words)
└── CLAUDE.md               # Project instructions
```

See [MASTER_PLAN.md](MASTER_PLAN.md) for complete design.

---

## Quick Start

### Prerequisites

- Kaggle account with GPU quota (2×T4, 30h/week)
- Kaggle CLI configured (`~/.kaggle/kaggle.json`)
- Python 3.11+, PyTorch 2.0+

### Setup

```bash
# Clone this repo
git clone https://github.com/j1s4nn/adaptive-granularity-wm.git
cd adaptive-granularity-wm

# Install dependencies
pip install -r requirements.txt

# Verify Kaggle datasets are accessible
kaggle datasets files ajjisan/causal-rcm-ckpts --page-size 5
kaggle datasets files ajjisan/wan21-t5 --page-size 5
kaggle datasets files ajjisan/wan21-dit-vae --page-size 5
```

### Run Experiments

```bash
# Phase 0: Feasibility check (1 GPU-hour)
python scripts/push_and_wait.py --job jobs/phase0 --timeout 7200

# E1: Fixed-mode baseline (20 GPU-hours, 4 parallel jobs)
python scripts/push_and_wait.py --job jobs/e1_baseline --parallel 4

# E2-E5: See MASTER_PLAN.md for full sequence
```

---

## Method (Brief)

### Uncertainty Score
Pre-generation score computed from:
- **EMA divergence**: ||v_θ - v_θ^-|| (student vs EMA copy)
- **Residual**: ||pred_i - pred_{i-1}|| (self-consistency)
- **Warp mismatch**: ||x_{i-1} - W(x_{i-2}, flow)|| (optical flow prediction error)

Combined: `u_i = g_φ(u_EMA, u_res, u_warp)`

### Threshold Controller
```
If Ψ_i ≥ τ_high → fine mode (K=1, c1-1 checkpoint)
If Ψ_i ≤ τ_low  → coarse mode (K=3, c3-3 checkpoint)
Else            → keep previous mode
```

Where `Ψ_i = u_i + λΔa_i` (text-only: λ=0)

### Recovery + Hysteresis
After uncertainty spike (Ψ_i ≥ τ_high):
1. Force fine mode for R steps (recovery counter)
2. Decrement counter each step
3. Return to threshold rule when r_i = 0

Prevents rapid oscillation at threshold boundary.

---

## Results (Summary)

**RQ1 (Signal)**: Uncertainty AUROC = [pending] vs motion = [pending]  
**RQ2 (Switching)**: Adaptive FVD@50 = [pending] vs best fixed = [pending]  
**RQ3 (Recovery)**: Switch count with R>0: [pending] vs R=0: [pending]  
**RQ4 (Monotonicity)**: [pending — exploratory finding]

See paper for full results.

---

## Citation

```bibtex
@article{jisan2026uncertainty,
  title={Uncertainty-Guided Adaptive Temporal Granularity for Interactive Video World Models},
  author={Jisan, Hossen Md},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Causal-rCM backbone: [NVlabs/rcm](https://github.com/NVlabs/rcm)
- Wan2.1 base model: [Alibaba Research](https://github.com/alibaba/Wan)
- Kaggle compute: T4 GPUs for all experiments
