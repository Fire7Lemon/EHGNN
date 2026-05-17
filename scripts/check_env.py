#!/usr/bin/env python3
"""Read-only environment check for EHGNN reproduction (torch/DGL/OGB stack).

Does not modify any files. Exit code 1 if any checked dependency fails to import.
"""
from __future__ import annotations

import sys


def line(title: str, value: object = "") -> None:
    sep = "=" * 60
    print(sep)
    print(f" {title}")
    print(sep)
    if value != "":
        print(value)


def main() -> int:
    line("EHGNN environment check")
    print(f"Python executable: {sys.executable}")
    print(f"Python version:    {sys.version.replace(chr(10), ' ')}")
    print(f"sys.prefix:        {sys.prefix}")
    print()

    failures: list[str] = []

    # --- torch + CUDA ---
    try:
        import torch

        print(f"[OK] torch: {torch.__version__}")
        print(f"     torch.version.cuda: {torch.version.cuda}")
        print(f"     torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                print(f"     GPU name: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"     GPU name: [FAIL] {type(e).__name__}: {e}")
                failures.append("torch.gpu_name")
        else:
            print("     GPU name: (none — CUDA not available)")
    except Exception as e:
        print(f"[FAIL] torch: {type(e).__name__}: {e}")
        failures.append("torch")
        torch = None  # noqa: F841

    print()

    # --- dgl ---
    try:
        import dgl

        ver = getattr(dgl, "__version__", "no __version__")
        print(f"[OK] dgl: {ver}")
    except Exception as e:
        print(f"[FAIL] dgl: {type(e).__name__}: {e}")
        failures.append("dgl")

    print()

    # --- torch_scatter ---
    try:
        import torch_scatter

        ver = getattr(torch_scatter, "__version__", "no __version__")
        print(f"[OK] torch_scatter: {ver}")
    except Exception as e:
        print(f"[FAIL] torch_scatter: {type(e).__name__}: {e}")
        failures.append("torch_scatter")

    print()

    # --- scientific stack ---
    for mod_name, pkg_label in [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sklearn", "sklearn"),
        ("pandas", "pandas"),
    ]:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "no __version__")
            print(f"[OK] {pkg_label}: {ver}")
        except Exception as e:
            print(f"[FAIL] {pkg_label}: {type(e).__name__}: {e}")
            failures.append(pkg_label)

    print()

    # --- tqdm (module name tqdm) ---
    try:
        import tqdm

        ver = getattr(tqdm, "__version__", "no __version__")
        print(f"[OK] tqdm: {ver}")
    except Exception as e:
        print(f"[FAIL] tqdm: {type(e).__name__}: {e}")
        failures.append("tqdm")

    print()

    # --- ogb ---
    try:
        import ogb

        ver = getattr(ogb, "__version__", "no __version__")
        print(f"[OK] ogb: {ver}")
    except Exception as e:
        print(f"[FAIL] ogb: {type(e).__name__}: {e}")
        failures.append("ogb")

    print()

    # --- optional parallel deps ---
    for mod_name, label in [("joblib", "joblib"), ("dask", "dask")]:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "no __version__")
            print(f"[OK] {label} (optional): {ver}")
        except Exception as e:
            print(f"[WARN] {label} (optional): {type(e).__name__}: {e}")

    print()
    line("Summary")
    if failures:
        print(f"FAILED imports ({len(failures)}): {', '.join(failures)}")
        print("Exit code: 1")
        return 1
    print("All required imports succeeded.")
    print("Exit code: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
