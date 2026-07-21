"""E-010 phase 1 — checkpoint-matched divergence curves for the bonus
order-only pairs P2 = (pythia-160m, data-seed1) and P3 = (pythia-160m,
data-seed2).

Pre-registered in docs/experiment_log.md E-010 (2026-07-22, D-016). Runs ONLY
if the phase-0 main-run continuity gate G-vii passed (src/e009_audit_checkpoints
--set=main).

Comparability (D-012): the measured-number code is REUSED FROM E-009 UNCHANGED —
this module imports `run_cell` (which calls `load_model`/`d_f`/`d_theta_rel`),
`build_eval_tokens`, `GRID`, `N_BLOCKS`, `BLOCK` from `e009_divergence`. So the
metric, the 12-point grid, and the frozen eval shard are byte-identical to the
F-015 P1/J1 run, and P2/P3 cells sit directly next to those numbers. The ONLY
E-010-specific logic here is the pair list, the output filename, the
reproducibility cell, and the DIRECTION of the F-014 tripwire.

Floor / ceiling / J1 anchors are NOT re-run: they are pair-independent, already
A-001-certified at this instrument version (F-015), and are read from the F-015
raw file at gate-evaluation time (H-d/H-e), not here.

Mirrored F-014 tripwire: an order-only pair SHARES its init, so d_theta must be
SMALL at early t (P1 measured 2.4e-5 @4, 0.025 @256). Any cell with
4 <= t <= 1000 and d_theta > 0.5 betrays a main-run continuity defect the
phase-0 audit (<= step256) would miss at later t -> voids that pair. (E-009's J1
tripwire was the opposite condition: d_theta < 0.5 = family collapse of an
INDEPENDENT-init pair.)

Incremental/resumable: results JSON rewritten after every cell; existing cells
skipped on rerun. Forward evals only, no gradients.
"""

import datetime
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from e009_divergence import (  # noqa: E402  measured-number code, reused as-is
    RAW, GRID, N_BLOCKS, BLOCK, build_eval_tokens, run_cell,
)

DATE = datetime.date.today().isoformat()

PAIRS = {
    "P2": ("EleutherAI/pythia-160m", "EleutherAI/pythia-160m-data-seed1"),
    "P3": ("EleutherAI/pythia-160m", "EleutherAI/pythia-160m-data-seed2"),
}
# mirror of E-009's J1 tripwire: shared-init pairs must stay LOW in d_theta.
TRIPWIRE = {"t_min": 4, "t_max": 1000, "d_theta_max": 0.5}
# determinism cert for the new arm (P2@16000 evaluated twice)
REPRO = ("repro:P2@16000", "EleutherAI/pythia-160m", 16000,
         "EleutherAI/pythia-160m-data-seed1", 16000)


def resolve_out_path():
    """E-010's own raw file (kept separate from the E-009 file for provenance).
    Append to today's if present, else the newest existing e010 file, else fresh."""
    today = RAW / f"e010_divergence_{DATE}.json"
    if today.exists():
        return today
    existing = sorted(RAW.glob("e010_divergence_*.json"))
    return existing[-1] if existing else today


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch.use_deterministic_algorithms(True)
        det = True
    except Exception:
        det = False

    tokens = build_eval_tokens()

    OUT = resolve_out_path()
    print(f"{'resuming from' if OUT.exists() else 'creating'} {OUT}", flush=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {
        "experiment": "E-010 phase 1 (bonus order-only divergence curves)",
        "date": DATE, "device": device, "deterministic_algorithms": det,
        "torch": torch.__version__, "grid": GRID,
        "n_blocks": N_BLOCKS, "block_len": BLOCK, "cells": {},
    }
    cells = results["cells"]

    def save():
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    jobs = []
    for name, (ra, rb) in PAIRS.items():
        for t in GRID:
            jobs.append((f"{name}@{t}", ra, t, rb, t))
    jobs.append(REPRO)

    for key, ra, ta, rb, tb in jobs:
        if key in cells:
            continue
        print(f"[{key}] {ra}@step{ta} vs {rb}@step{tb} ...", flush=True)
        try:
            cells[key] = run_cell(ra, ta, rb, tb, tokens, device)
            dtr = cells[key]["d_theta_rel"]
            dtr_s = f"{dtr:.6g}" if dtr is not None else \
                f"n/a ({cells[key].get('d_theta_reason')})"
            print(f"  sym_kl={cells[key]['sym_kl']:.6g}  d_theta={dtr_s}  "
                  f"disagree={cells[key]['disagree_rate']:.4f}", flush=True)
        except Exception as e:
            cells[key] = {"void": True, "error": repr(e)}
            print(f"  VOID: {e!r}", flush=True)
        save()

    # mirrored tripwire: an order-only (shared-init) pair must stay near ~0 in
    # d_theta at early t; d_theta > 0.5 there means main is NOT continuing from
    # its own init at that step (F-014 signature) -> void that pair.
    tw = TRIPWIRE
    tripped = {
        k: c["d_theta_rel"] for k, c in cells.items()
        if "@" in k and not k.startswith("repro") and not c.get("void")
        and k.split("@")[0] in PAIRS
        and tw["t_min"] <= int(k.split("@")[1]) <= tw["t_max"]
        and c.get("d_theta_rel") is not None
        and c["d_theta_rel"] > tw["d_theta_max"]
    }
    results["tripwire"] = {
        "config": tw, "tripped_cells": tripped,
        "voided_pairs": sorted({k.split("@")[0] for k in tripped}),
    }

    # pre-declared reproducibility verdict (P2@16000 twice)
    if "P2@16000" in cells and "repro:P2@16000" in cells and \
            not cells["P2@16000"].get("void") and \
            not cells["repro:P2@16000"].get("void"):
        a, b = cells["P2@16000"]["sym_kl"], cells["repro:P2@16000"]["sym_kl"]
        results["repro_rel_err"] = abs(a - b) / max(abs(a), 1e-300)
        results["repro_pass"] = results["repro_rel_err"] <= 1e-6
    save()
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
