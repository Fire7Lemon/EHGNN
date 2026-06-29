"""Path and sys.path helpers for method-exploration experiment scripts."""
from __future__ import annotations

import sys
from pathlib import Path

EXP_SUBDIR = "experiments/opt_20260629_method_exploration"


def get_project_root(current_file, levels: int = 4) -> Path:
    """Return EHGNN repo root (default: code/ -> ... -> repo root = parents[4])."""
    return Path(current_file).resolve().parents[levels]


def get_exp_root(project_root: Path) -> Path:
    return Path(project_root).resolve() / EXP_SUBDIR


def ensure_trailing_slash(path) -> str:
    """utils.load_PubMed concatenates ``data_path + dataset + '/'`` — path must end with ``/``."""
    s = str(path)
    return s if s.endswith("/") else s + "/"


def resolve_project_data_path(project_root: Path, path_arg: str | None = None) -> str:
    """
    Return ``data_path`` for ``load_PubMed`` / LP ``load_PubMed``.

    Default: ``PROJECT_ROOT/data/`` (absolute, trailing slash).
    Only an **absolute** ``path_arg`` overrides the default.
    """
    root = Path(project_root).resolve()
    if path_arg is not None and Path(path_arg).is_absolute():
        return ensure_trailing_slash(path_arg)
    return ensure_trailing_slash(root / "data")


def safe_relpath(path, root) -> str:
    """Relative path from ``root``; fallback to original string if not under root."""
    path = Path(path)
    root = Path(root).resolve()
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def resolve_under_root(path, root: Path) -> Path:
    """Resolve ``path``; if relative, join with resolved project ``root``."""
    p = Path(path)
    root = Path(root).resolve()
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def add_source_dir(project_root: Path, task: str) -> Path:
    """
    Add NC or LP source tree to ``sys.path`` (never both in one process).

    task='nc' -> PROJECT_ROOT/'Node Classification'
    task='lp' -> PROJECT_ROOT/'Link Prediction'
    """
    task = task.lower().strip()
    root = Path(project_root).resolve()
    if task == "nc":
        src = root / "Node Classification"
    elif task == "lp":
        src = root / "Link Prediction"
    else:
        raise ValueError("task must be 'nc' or 'lp', got {!r}".format(task))
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
    return src


def bootstrap_paths(
    current_file,
    task: str | None = None,
    extra_code_dirs: list[Path] | None = None,
    levels: int = 4,
) -> tuple[Path, Path, Path]:
    """
    Insert common/, script code dir, optional NC/LP, optional extra code dirs.

    Returns (project_root, exp_root, code_dir).
    """
    code_dir = Path(current_file).resolve().parent
    project_root = get_project_root(current_file, levels=levels)
    exp_root = get_exp_root(project_root)
    common = exp_root / "common"

    for p in (common, code_dir):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    if task:
        add_source_dir(project_root, task)

    if extra_code_dirs:
        for d in extra_code_dirs:
            d = Path(d)
            if d.is_dir():
                s = str(d.resolve())
                if s not in sys.path:
                    sys.path.insert(0, s)

    return project_root, exp_root, code_dir


def add_code_dir(code_dir: Path) -> None:
    """Append an extra experiment code directory to sys.path if it exists."""
    d = Path(code_dir).resolve()
    if d.is_dir():
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)
