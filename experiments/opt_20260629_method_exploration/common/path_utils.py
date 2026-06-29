"""Path helpers for experiment scripts calling EHGNN utils load_* functions."""
from __future__ import annotations

from pathlib import Path


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
    if path_arg is not None and Path(path_arg).is_absolute():
        return ensure_trailing_slash(path_arg)
    return ensure_trailing_slash(project_root / "data")
