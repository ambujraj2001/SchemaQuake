"""Compatibility preflight checks for HF Space training image.

Run modes:
- build-time:  python space/preflight.py --build-check
- runtime:     python space/preflight.py
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import List


def _check_import(mod: str, errors: List[str]) -> object | None:
    try:
        return importlib.import_module(mod)
    except Exception as exc:
        errors.append(f"import `{mod}` failed: {exc!r}")
        return None


def _check_trl(errors: List[str]) -> None:
    # Support both top-level and nested symbols depending on TRL build.
    try:
        from trl import GRPOConfig, GRPOTrainer  # noqa: F401
        return
    except Exception:
        pass
    try:
        from trl.trainer.grpo_config import GRPOConfig  # noqa: F401
        from trl.trainer.grpo_trainer import GRPOTrainer  # noqa: F401
        return
    except Exception as exc:
        errors.append(f"GRPO symbols unavailable in `trl`: {exc!r}")


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
    _check_import("unsloth", errors)
    _check_import("unsloth_zoo", errors)
    _check_import("trl", errors)
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
