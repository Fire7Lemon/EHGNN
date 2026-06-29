#!/usr/bin/env python3
"""Import-level smoke checks (no training). Called from smoke_test_method_exploration.sh."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = EXP_ROOT / "common"
PROJECT_ROOT = EXP_ROOT.parents[1]

P4_MAIN = EXP_ROOT / "04_distill_mlp_pubmed_nc/code/main_pubmed_nc_distill.py"

P4_DRY_RUN_CMD = [
    sys.executable,
    str(P4_MAIN),
    "--dataset",
    "PubMed",
    "--seed",
    "42",
    "--teacher_epochs",
    "1",
    "--student_epochs",
    "1",
    "--teacher_mode",
    "train",
    "--student_input",
    "raw",
    "--dry_run_runtime_check",
    "--root_out",
    "experiments/opt_20260629_method_exploration",
]

ENTRY_HELP = [
    EXP_ROOT / "01_sehgnn_lite_pubmed_nc/code/run_pubmed_nc_sehgnn_lite.py",
    EXP_ROOT / "02_lp_pair_decoder_pubmed_lp/code/main_pubmed_lp_pair_decoder.py",
    EXP_ROOT / "03_sampled_lp_training_pubmed_lp/code/main_pubmed_lp_sampled_training.py",
    EXP_ROOT / "04_distill_mlp_pubmed_nc/code/main_pubmed_nc_distill.py",
    EXP_ROOT / "05_ppr_topk_lite_design/code/demo_pubmed_ppr_topk.py",
]

PARSERS = [
    EXP_ROOT / "00_existing_results/code/p0_scan_and_parse.py",
    EXP_ROOT / "01_sehgnn_lite_pubmed_nc/code/parse_sehgnn_lite_results.py",
    EXP_ROOT / "02_lp_pair_decoder_pubmed_lp/code/parse_pair_decoder_results.py",
    EXP_ROOT / "03_sampled_lp_training_pubmed_lp/code/parse_sampled_lp_results.py",
    EXP_ROOT / "04_distill_mlp_pubmed_nc/code/parse_distill_results.py",
    EXP_ROOT / "05_ppr_topk_lite_design/code/parse_ppr_topk_results.py",
]

PLOTS = [
    EXP_ROOT / "01_sehgnn_lite_pubmed_nc/code/plot_sehgnn_lite_results.py",
    EXP_ROOT / "02_lp_pair_decoder_pubmed_lp/code/plot_pair_decoder_results.py",
    EXP_ROOT / "03_sampled_lp_training_pubmed_lp/code/plot_sampled_lp_results.py",
    EXP_ROOT / "04_distill_mlp_pubmed_nc/code/plot_distill_results.py",
    EXP_ROOT / "05_ppr_topk_lite_design/code/plot_ppr_topk_results.py",
]


def _bootstrap_common_from_file(py_file: Path) -> Path:
    exp_root = py_file.resolve().parents[2]
    common = exp_root / "common"
    if str(common) not in sys.path:
        sys.path.insert(0, str(common))
    return exp_root


def check_path_utils_for_file(py_file: Path) -> tuple[bool, str]:
    """Verify file's EXP_ROOT/common bootstrap can import path_utils."""
    try:
        _bootstrap_common_from_file(py_file)
        import path_utils  # noqa: F401

        root = path_utils.get_project_root(py_file)
        dp = path_utils.resolve_project_data_path(root)
        if not dp.endswith("/"):
            return False, "data path missing trailing slash: {}".format(dp)
        if "data/" not in dp.replace("\\", "/"):
            return False, "unexpected data path: {}".format(dp)
        return True, "data_path={}".format(dp)
    except Exception as e:
        return False, str(e)


def check_plot_utils_for_file(py_file: Path) -> tuple[bool, str]:
    try:
        _bootstrap_common_from_file(py_file)
        import plot_utils  # noqa: F401
        return True, "plot_utils ok"
    except Exception as e:
        return False, str(e)


def check_parser_import(py_file: Path) -> tuple[bool, str]:
    try:
        spec = importlib.util.spec_from_file_location(
            "parser_mod_{}".format(py_file.stem), py_file
        )
        if spec is None or spec.loader is None:
            return False, "spec failed"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return True, "parser module loaded"
    except Exception as e:
        return False, str(e)


def check_entry_help(py_file: Path, timeout: int = 60) -> tuple[str, str]:
    """Return status: pass | fail | skip, message."""
    try:
        r = subprocess.run(
            [sys.executable, str(py_file), "--help"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return "fail", "timeout"
    except Exception as e:
        return "fail", str(e)

    if r.returncode == 0:
        return "pass", "exit 0"
    err = (r.stderr or r.stdout or "")[-500:]
    optional_deps = (
        "No module named 'dgl'",
        "No module named 'torch'",
        "No module named 'torch_scatter'",
        "No module named 'torch_geometric'",
    )
    if any(m in err for m in optional_deps):
        return "skip", "missing optional dep in env: " + err.strip()[:200]
    if "No module named 'path_utils'" in err or "No module named 'plot_utils'" in err:
        return "fail", err.strip()[:300]
    if "No module named 'utils'" in err or "No module named 'models'" in err:
        return "fail", "NC/LP path bootstrap missing: " + err.strip()[:300]
    return "fail", "exit {}: {}".format(r.returncode, err.strip()[:300])


def check_p4_dry_runtime(timeout: int = 600) -> tuple[str, str]:
    """P4 tiny runtime check (RW + one teacher/student forward)."""
    try:
        r = subprocess.run(
            P4_DRY_RUN_CMD,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return "fail", "timeout after {}s".format(timeout)
    except Exception as e:
        return "fail", str(e)

    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 0 and "[P4-DRY-RUN] PASS" in out:
        return "pass", "dry_run_runtime_check OK"
    err = out[-800:]
    optional_deps = (
        "No module named 'dgl'",
        "No module named 'torch'",
        "No module named 'torch_scatter'",
        "No module named 'torch_geometric'",
    )
    if any(m in err for m in optional_deps):
        return "skip", "missing optional dep: " + err.strip()[:200]
    if "'NoneType' object has no attribute 'append'" in err:
        return "fail", "t_typess None bug: " + err.strip()[:300]
    return "fail", "exit {}: {}".format(r.returncode, err.strip()[:400])


def main() -> int:
    failures = 0
    print("=== import_smoke_checks ===")
    print("EXP_ROOT:", EXP_ROOT)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print()

    for py_file in ENTRY_HELP + PARSERS:
        ok, msg = check_path_utils_for_file(py_file)
        tag = "PASS" if ok else "FAIL"
        print("[{}] path_utils {}".format(tag, py_file.relative_to(EXP_ROOT)))
        print("       ", msg)
        if not ok:
            failures += 1

    for py_file in PLOTS:
        ok, msg = check_plot_utils_for_file(py_file)
        tag = "PASS" if ok else "FAIL"
        print("[{}] plot_utils {}".format(tag, py_file.relative_to(EXP_ROOT)))
        print("       ", msg)
        if not ok:
            failures += 1

    for py_file in PARSERS:
        ok, msg = check_parser_import(py_file)
        tag = "PASS" if ok else "FAIL"
        print("[{}] import parser {}".format(tag, py_file.relative_to(EXP_ROOT)))
        print("       ", msg)
        if not ok:
            failures += 1

    print()
    print("--- entry --help ---")
    for py_file in ENTRY_HELP:
        status, msg = check_entry_help(py_file)
        print("[{}] --help {}".format(status.upper(), py_file.relative_to(EXP_ROOT)))
        print("       ", msg)
        if status == "fail":
            failures += 1

    print()
    print("--- P4 dry_run_runtime_check ---")
    status, msg = check_p4_dry_runtime()
    print("[{}] P4 dry_run {}".format(status.upper(), P4_MAIN.relative_to(EXP_ROOT)))
    print("       ", msg)
    if status == "fail":
        failures += 1

    print()
    if failures:
        print("FAILURES:", failures)
        return 1
    print("ALL IMPORT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
