"""Compatibility preflight checks for HF Space training image.

Run modes:
- build-time:  python space/preflight.py --build-check
- runtime:     python space/preflight.py
"""
from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import sys
from typing import List


def _check_import(mod: str, errors: List[str]) -> object | None:
    try:
        return importlib.import_module(mod)
    except Exception as exc:
        errors.append(f"import `{mod}` failed: {exc!r}")
        return None


def _check_trl(errors: List[str]) -> None:
    # Support top-level, known nested paths, and discovery by scanning TRL modules.
    # Unsloth may patch TRL to add GRPO classes, so try that first.
    try:
        from unsloth import FastLanguageModel, PatchFastRL
        PatchFastRL("GRPO", FastLanguageModel)
    except Exception:
        # Keep scanning/diagnostics below; missing patch should surface naturally.
        pass

    try:
        import trl
    except Exception as exc:
        errors.append(f"import `trl` failed: {exc!r}")
        return

    if hasattr(trl, "GRPOConfig") and hasattr(trl, "GRPOTrainer"):
        print(f"[preflight] trl={getattr(trl, '__version__', 'unknown')} (top-level GRPO exports)")
        return

    candidates = [
        "trl.trainer.grpo_config",
        "trl.trainer.grpo_trainer",
        "trl.trainer.grpo",
        "trl.trainer",
    ]
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, "GRPOConfig") and hasattr(mod, "GRPOTrainer"):
            print(
                f"[preflight] trl={getattr(trl, '__version__', 'unknown')} "
                f"(GRPO from {mod_name})"
            )
            return

    try:
        for m in pkgutil.walk_packages(trl.__path__, prefix="trl."):
            if "grpo" not in m.name.lower():
                continue
            try:
                mod = importlib.import_module(m.name)
            except Exception:
                continue
            if hasattr(mod, "GRPOConfig") and hasattr(mod, "GRPOTrainer"):
                print(
                    f"[preflight] trl={getattr(trl, '__version__', 'unknown')} "
                    f"(GRPO discovered in {m.name})"
                )
                return
    except Exception:
        pass

    errors.append(
        "GRPO symbols unavailable in `trl` after scanning known and discovered paths"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-check", action="store_true")
    args = parser.parse_args()

    errors: List[str] = []

    torch = _check_import("torch", errors)
    tv = _check_import("torchvision", errors)
    _check_import("xformers", errors)
    _check_import("transformers", errors)
    _check_import("datasets", errors)
    _check_import("peft", errors)
    _check_import("accelerate", errors)
    _check_import("bitsandbytes", errors)
    _check_import("trl", errors)
    # Build containers in HF Space run without GPU driver. Unsloth import can
    # touch CUDA immediately, so gate that to runtime only.
    if args.build_check:
        print("[preflight] build-check: skipping unsloth + GRPO symbol checks")
    else:
        _check_import("unsloth", errors)
        _check_import("unsloth_zoo", errors)
        _check_trl(errors)

    if torch is not None and tv is not None:
        tver = getattr(torch, "__version__", "unknown")
        tvver = getattr(tv, "__version__", "unknown")
        print(f"[preflight] torch={tver} torchvision={tvver}")
        # Common crash source: mismatched builds can break torchvision::nms.
        try:
            has_nms = hasattr(tv.ops, "nms")
            if not has_nms:
                errors.append("torchvision.ops.nms missing (torch/vision ABI mismatch)")
        except Exception as exc:
            errors.append(f"access torchvision.ops.nms failed: {exc!r}")

    if torch is not None:
        cuda_ok = bool(torch.cuda.is_available())
        print(f"[preflight] cuda_available={cuda_ok}")
        if not args.build_check and not cuda_ok:
            errors.append("CUDA GPU not available at runtime")

    if errors:
        print("[preflight] FAILED")
        for err in errors:
            print(f"[preflight] - {err}")
        return 1

    print("[preflight] OK")
    if args.build_check:
        # Build check intentionally does not require CUDA.
        print("[preflight] build-check mode (CUDA requirement skipped)")
    else:
        print(f"[preflight] runtime mode on host={os.uname().nodename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
