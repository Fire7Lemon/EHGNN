"""Optional matplotlib helpers — plot scripts exit 0 when matplotlib is missing."""
from __future__ import annotations


def matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def require_matplotlib_or_skip(script_name: str = "plot"):
    """
    Return ``matplotlib.pyplot`` module, or exit 0 with a clear message if missing.
    """
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print(
            "[SKIP] {}: matplotlib not installed; skipping plots (exit 0)".format(
                script_name
            )
        )
        raise SystemExit(0)
