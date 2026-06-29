# EHGNN Method Exploration Audit Fix Report

## Audit Time

2026-06-03 — systematic engineering audit (post server P5/P2/P3 runs)

## Scanned Files

Glob under `experiments/opt_20260629_method_exploration/`:

| Area | Count / paths |
|------|----------------|
| Entry scripts | P1–P5 `run_*.py`, `main_*.py`, `demo_*.py` |
| Parsers | P1–P5 `parse_*.py`, P0 `p0_scan_and_parse.py` |
| Plotters | P1–P5 `plot_*.py` |
| Common | `common/path_utils.py`, `common/plot_utils.py` |
| Server | `server_scripts/*.sh`, `README_RUN_ALL.md` |
| Registry | `RUN_REGISTRY.csv`, `SERVER_RUN_QUEUE.md`, `COMMANDS.md` |

### Grep scan summary (2026-06-03)

| Pattern | Finding |
|---------|---------|
| `from utils import` | 5 entry scripts (P1–P5); all use `bootstrap_paths(..., task='nc'|'lp')` |
| `relative_to(ROOT` | **Cleared** in parsers; only inside `safe_relpath()` in `path_utils.py` |
| `matplotlib` | 5 plot scripts + P0; plot scripts use `plot_utils.require_matplotlib_or_skip` |
| `load_PubMed` | 5 entry scripts; all use `resolve_project_data_path(PROJECT_ROOT)` |
| `adj(scipy_fmt` | Only in `ppr_topk_lite.dgl_graph_to_scipy_csr` fallback chain |
| `server_results/` writes | **None** in experiment runners; P0/config JSON **read** baseline refs only |
| `Node Classification/results` writes | **None** in experiment code |
| `Link Prediction/results` writes | **None** in experiment code |

## Detected Issues (historical)

1. **utils import** — `PROJECT_ROOT` off-by-one → `ModuleNotFoundError: utils`
2. **data_path** — missing trailing `/` → `dataPubMed/node.dat`
3. **DGL adj API** — `DGLGraph.adj(scipy_fmt=...)` unsupported on server DGL
4. **matplotlib** — server env missing matplotlib → plot hard-fail
5. **parser paths** — `log_path.relative_to(ROOT)` with relative log + absolute ROOT
6. **stage status** — parse/plot failure masked successful training (P2/P3)
7. **common import path** — P4 failed with `ModuleNotFoundError: No module named 'path_utils'`; entry scripts used `CODE_DIR.parents[4]` to locate `common/` (points above repo root). `py_compile` did not catch it because imports are not executed.
8. **P4 teacher preprocessing** — P4 passed import smoke but failed at teacher RW loop: `t_typess` initialized as `None` instead of `[]` → `AttributeError: 'NoneType' object has no attribute 'append'`. `--help` smoke could not catch this.

## Fixed Issues

| Fix | Files |
|-----|-------|
| Unified `bootstrap_paths` / `add_source_dir(nc\|lp)` | `common/path_utils.py`, P1–P5 entry scripts |
| **Common `sys.path` bootstrap (`EXP_ROOT = parents[2]`)** | P0–P5 entry/parse/plot scripts (16 files); `bootstrap_common_path()` in `path_utils.py` |
| **Import-level smoke test** | `common/import_smoke_checks.py`, `server_scripts/smoke_test_method_exploration.sh` |
| **P4 teacher flow alignment + dry run** | `build_rw_similarity_matrices()`, `--dry_run_runtime_check`, smoke test P4 runtime block |
| `resolve_project_data_path` → `{ROOT}/data/` | All entry scripts |
| `safe_relpath` + `resolve_under_root` | P1–P5 parsers, P0 scanner |
| DGL CSR compat layer | `ppr_topk_lite.dgl_graph_to_scipy_csr` |
| matplotlib optional (exit 0) | `common/plot_utils.py`, all P1–P5 plot scripts |
| `PARSE_ONLY=1` + skip-existing CSV | All `run_p1`–`run_p5` server scripts |
| parse/plot `\|\| echo [WARN]` | P1–P5 server scripts |
| Removed shell `PYTHONPATH` NC/LP mix | All server scripts (Python handles sys.path) |
| Smoke test harness | `server_scripts/smoke_test_method_exploration.sh` |
| Smoke test: real import / `--help` | `common/import_smoke_checks.py` — checks `path_utils`, `plot_utils`, parsers, entry `--help`; fails on missing common bootstrap |
| Smoke test: P4 `dry_run_runtime_check` | Loads PubMed, builds RW/`t_typess`, one teacher+student forward; catches `t_typess None` and similar runtime bugs |

## Remaining Risks

| Risk | Mitigation |
|------|------------|
| P5 PPR-lite **459s** vs RW **0.2s** — not production-ready | Document as design/demo only; low Jaccard ~0.08 |
| P4 teacher retrain cost | Use `TEACHER_MODE=load` when logits exist |
| DGL/torch version drift | Run `check_server_before_opt_runs.sh` before batch |
| matplotlib on server | Plot skip OK; install `matplotlib` in `ehgnn` for figures |
| P3 ratio025 may still need full training | Script skips only when main CSV exists |
| `run_all` still marks stage failed if sub-script exits non-zero | Sub-scripts now tolerate parse/plot WARN |

## Validation Commands

```bash
cd /home/mayq/ehgnn/EHGNN

# 1) Smoke test (no training)
bash experiments/opt_20260629_method_exploration/server_scripts/smoke_test_method_exploration.sh

# 2) Environment check
bash experiments/opt_20260629_method_exploration/server_scripts/check_server_before_opt_runs.sh

# 3) Targeted parse-only (examples)
PARSE_ONLY=1 bash experiments/opt_20260629_method_exploration/server_scripts/run_p5_ppr_topk_demo.sh
PARSE_ONLY=1 bash experiments/opt_20260629_method_exploration/server_scripts/run_p3_sampled_lp_pubmed.sh
```

Local (no DGL):

```powershell
python experiments/opt_20260629_method_exploration/common/import_smoke_checks.py
python -m py_compile experiments/opt_20260629_method_exploration/common/path_utils.py
python -m py_compile experiments/opt_20260629_method_exploration/common/plot_utils.py
# entry --help on server (ehgnn env): see import_smoke_checks ENTRY_HELP list
```

## Server Re-run Plan

| Stage | Status | Next action |
|-------|--------|-------------|
| **P5** | Demo **done** (CSV/JSON on server) | `PARSE_ONLY=1` for parse/plot; **do not** re-run demo |
| **P2** | Training likely done | `PARSE_ONLY=1` if only parse/plot missing |
| **P3 ratio050** | **done** | Skip training; parse + ratio025 |
| **P3 ratio025** | Pending | Normal run (skip050 auto) |
| **P1** | Not run | Full `run_p1_sehgnn_lite_pubmed_nc.sh` |
| **P4** | `t_typess None` — **fixed** | smoke + `dry_run_runtime_check` PASS → `run_p4_distill_pubmed_nc.sh` |

### P5 server results (recorded)

| Metric | Value |
|--------|-------|
| PPR runtime | 459.5319 s |
| RW runtime | 0.2193 s |
| mean Jaccard | 0.0844 |
| mean overlap@20 | 0.1096 |

**Conclusion:** Naive PPR-lite **not** a drop-in RW Top-K replacement; design/demo evidence only.

---

**Formal reproduction code (`Node Classification/`, `Link Prediction/`) was not modified.**
