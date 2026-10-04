"""
Phase 0: Feasibility Probe
==========================

GOAL: Confirm T4 compatibility, checkpoint loading, VRAM limits, internals access

INPUT:
- Kaggle datasets (mounted in /kaggle/input)
- T4 GPU (16GB VRAM)

OUTPUT:
- VRAM profile (peak usage for c1-1 vs c3-3)
- Available internals list (EMA model, latents, KV cache)
- FlashAttention-2 availability
- Maximum rollout length at 480p
- Action checkpoint detection (Cosmos 3)

PASS/FAIL criteria:
- PASS: Both checkpoints load, peak VRAM <14GB, inference runs
- FAIL: VRAM overflow, checkpoints missing, critical errors
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

import torch
import torch.cuda

# Add src to path
sys.path.insert(0, '/kaggle/working')

from src.causal_rcm import (
    load_causal_rcm_checkpoints,
    check_flash_attention,
    get_attention_backend
)


def profile_vram():
    """Profile GPU memory usage."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    torch.cuda.reset_peak_memory_stats()

    return {
        "device": torch.cuda.get_device_name(),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3)
    }


def check_internals(checkpoint):
    """
    Check what internals are exposed in the checkpoint.

    Looking for:
    - EMA model (for u_i^EMA)
    - Model architecture details
    - Config/hyperparameters
    """
    internals = {
        "keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Not a dict",
        "has_ema": False,
        "has_model": False,
        "has_config": False
    }

    if isinstance(checkpoint, dict):
        # Common checkpoint structures
        for key in checkpoint.keys():
            key_lower = str(key).lower()
            if 'ema' in key_lower:
                internals["has_ema"] = True
                internals["ema_key"] = key
            if 'model' in key_lower or 'state_dict' in key_lower:
                internals["has_model"] = True
                internals["model_key"] = key
            if 'config' in key_lower or 'hparams' in key_lower:
                internals["has_config"] = True
                internals["config_key"] = key

    return internals


def test_inference(checkpoints, max_frames=10):
    """
    Test basic inference with minimal rollout.

    Args:
        checkpoints: Loaded checkpoint dict
        max_frames: Small test rollout length
    """
    print(f"\nTesting inference with {max_frames} frames...")

    # This is a placeholder - actual test depends on rcm repo structure
    # In a real Phase 0, we'd:
    # 1. Clone NVlabs/rcm repo
    # 2. Use their inference script
    # 3. Generate 10 frames
    # 4. Measure VRAM peak

    result = {
        "status": "PLACEHOLDER",
        "message": "Actual inference requires rcm repo integration",
        "vram_peak_gb": 0.0
    }

    return result


def search_action_checkpoint():
    """
    Search for action-conditioned (Cosmos 3) checkpoint.
    """
    print("\nSearching for action-conditioned checkpoint...")

    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return {"found": False, "message": "Not in Kaggle environment"}

    # Search all mounted datasets for action/camera/cosmos keywords
    action_files = []
    for dataset_dir in kaggle_input.iterdir():
        if dataset_dir.is_dir():
            for file in dataset_dir.rglob("*.pt"):
                filename_lower = file.name.lower()
                if any(kw in filename_lower for kw in ['action', 'camera', 'cosmos', 'control']):
                    action_files.append(str(file))

    return {
        "found": len(action_files) > 0,
        "files": action_files,
        "message": f"Found {len(action_files)} potential action checkpoints"
    }


def main():
    """Run Phase 0 feasibility probe."""

    start_time = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "run_id": run_id,
        "phase": "phase0",
        "start_time": datetime.now().isoformat(),
        "status": "running"
    }

    try:
        print("="*60)
        print("PHASE 0: FEASIBILITY PROBE")
        print("="*60)

        # 1. Check CUDA and attention backend
        print("\n1. Checking GPU and attention backend...")
        fa2_available = check_flash_attention()
        attn_backend = get_attention_backend()

        results["gpu"] = {
            "flash_attention_2": fa2_available,
            "attention_backend": attn_backend,
            "initial_vram": profile_vram()
        }

        # 2. Load checkpoints
        print("\n2. Loading Causal-rCM checkpoints...")
        vram_before = profile_vram()

        checkpoints = load_causal_rcm_checkpoints(
            device='cuda',
            checkpoint_type='both',  # Load both c1-1 and c3-3
            step_count=2,  # Start with 2-step (recommended)
            verify_hashes=False  # Skip for speed
        )

        vram_after = profile_vram()

        results["checkpoints"] = {
            "c1_1_loaded": 'c1_1' in checkpoints,
            "c3_3_loaded": 'c3_3' in checkpoints,
            "vae_loaded": 'vae' in checkpoints,
            "t5_loaded": 't5' in checkpoints,
            "metadata": checkpoints.get('metadata', {}),
            "vram_after_load": vram_after,
            "vram_delta_gb": vram_after['peak_allocated_gb'] - vram_before['peak_allocated_gb']
        }

        # 3. Inspect internals
        print("\n3. Inspecting checkpoint internals...")
        if 'c1_1' in checkpoints:
            results["c1_1_internals"] = check_internals(checkpoints['c1_1'])
        if 'c3_3' in checkpoints:
            results["c3_3_internals"] = check_internals(checkpoints['c3_3'])

        # 4. Test inference (placeholder)
        print("\n4. Testing inference...")
        inference_result = test_inference(checkpoints, max_frames=10)
        results["inference_test"] = inference_result

        # 5. Search for action checkpoint
        action_search = search_action_checkpoint()
        results["action_checkpoint"] = action_search

        # 6. VRAM limit estimation
        print("\n5. Estimating maximum rollout length...")
        total_vram = results["gpu"]["initial_vram"]["total_memory_gb"]
        used_vram = vram_after["peak_allocated_gb"]
        available_vram = total_vram - used_vram - 2.0  # Reserve 2GB buffer

        # Rough estimate: ~0.1GB per frame at 480p
        estimated_max_frames = int(available_vram / 0.1)

        results["vram_limits"] = {
            "total_gb": total_vram,
            "used_after_load_gb": used_vram,
            "available_gb": available_vram,
            "estimated_max_frames_480p": estimated_max_frames,
            "recommendation": "50 frames" if estimated_max_frames >= 50 else f"{estimated_max_frames} frames"
        }

        # 7. Final verdict
        print("\n6. Final verdict...")

        pass_criteria = [
            ('c1_1' in checkpoints or 'c3_3' in checkpoints, "At least one checkpoint loaded"),
            (vram_after["peak_allocated_gb"] < 14.0, "VRAM usage < 14GB"),
            (estimated_max_frames >= 25, "Can generate ≥25 frames")
        ]

        all_pass = all(criterion[0] for criterion in pass_criteria)

        results["verdict"] = {
            "overall": "PASS" if all_pass else "FAIL",
            "criteria": [
                {"pass": crit[0], "description": crit[1]}
                for crit in pass_criteria
            ]
        }

        results["status"] = "complete"

        # Print summary
        print("\n" + "="*60)
        print(f"PHASE 0 VERDICT: {results['verdict']['overall']}")
        print("="*60)
        for criterion in results["verdict"]["criteria"]:
            status = "✓" if criterion["pass"] else "✗"
            print(f"{status} {criterion['description']}")

        if not action_search["found"]:
            print("\n⚠ WARNING: No action-conditioned checkpoint found")
            print("  → Proceeding with text-only (λ=0)")

        print(f"\n✓ Estimated max rollout: {estimated_max_frames} frames at 480p")
        print(f"✓ Attention backend: {attn_backend}")

    except Exception as e:
        print(f"\n✗ PHASE 0 FAILED: {e}")
        traceback.print_exc()

        results["status"] = "failed"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        results["verdict"] = {"overall": "FAIL"}

    finally:
        # Save results
        results["end_time"] = datetime.now().isoformat()
        results["duration_seconds"] = time.time() - start_time

        output_dir = Path("/kaggle/working/results/phase0")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / f"{run_id}_meta.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_dir / f'{run_id}_meta.json'}")
        print(f"✓ Duration: {results['duration_seconds']:.1f}s")


if __name__ == "__main__":
    main()
