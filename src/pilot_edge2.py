"""M3.1 E-008: pre-registered test of the F-012 edge law rho = c*K*eta.

Law and prefactor FROZEN from E-007's committed raw JSON before any new cell
runs (c fit by least squares through the origin of rho on K*eta over the 8
prefix/full-family cells; nulls fit to the same cells by the same procedure):

  (i)  T = 64 point predictions at three cells OFF the E-007 fitting grid:
       (K=350, eta=0.025), (K=650, eta=0.025), (K=982, eta=0.00625).
       Gate: each within +-25% of c*K*eta AND median |rel err| beats both
       null-A (rho = a*K) and null-B (rho = b*eta).
  (ii) T = 128 form test in a fresh run (rng 43, full word seed 103):
       prefix cells K in {300, 600, 900, 1200} at eta = 0.05.
       Gate: rho linear in K through the origin (centered R^2 >= 0.98) and
       strictly increasing. c(128) is a measurement, not a prediction.

Transport machinery bit-identical to E-005/E-007 (batched forward-JVP chain,
exact HVP, per-sample audit gates). Hypothesis pre-registered in
docs/experiment_log.md (E-008) BEFORE this run; E-000 gates BOTH arms.
"""

import json
import os
import subprocess
import sys

import numpy as np

from pilot_defects import D_IN, e000, init_params, make_data
from pilot_edge import prefix_word, transport_cell
from pilot_transport import inversion_pairs

N_EVAL = 1024
C_FROZEN = 0.0329183360612305   # E-007 8-cell fit, procedure above
NULL_A = 0.00121399172419113    # rho = a*K   (ignores eta)
NULL_B = 19.566349647885417     # rho = b*eta (ignores K)
CELLS_T64 = [(350, 0.025), (650, 0.025), (982, 0.00625)]
ETA_REF = 0.05
T128_GRID = [300, 600, 900, 1200]
REL_TOL = 0.25
R2_MIN = 0.98


def arm_t64(theta0, batches, T=64):
    """Clause (i): frozen-law point predictions at out-of-grid cells."""
    X_eval = np.random.default_rng(7).normal(size=(N_EVAL, D_IN))
    word_full = [int(t) for t in np.random.default_rng(102).permutation(T)]
    K_full = len(inversion_pairs(word_full))
    cells = {}
    for s, eta in CELLS_T64:
        word = word_full if s == K_full else prefix_word(word_full, s, T)
        K = len(inversion_pairs(word))
        if K != s:
            return {"ABORT": f"cell ({s},{eta}): word has {K} inversions"}
        cell = transport_cell(theta0, batches, eta, T, word, X_eval)
        if not cell["gap_f_above_noise"]:
            cell["VOID"] = "gap at noise floor"
        if cell["param_residual"] < 1 and cell["audit_median_rel_err"] > 0.02:
            cell["VOID"] = "audit gate failed at non-overshoot cell"
        rho, pred = cell["param_residual"], C_FROZEN * K * eta
        cell.update({
            "eta": eta, "pred_law": pred,
            "pred_nullA": NULL_A * K, "pred_nullB": NULL_B * eta,
            "rel_err_law": abs(rho - pred) / pred,
            "rel_err_nullA": abs(rho - NULL_A * K) / (NULL_A * K),
            "rel_err_nullB": abs(rho - NULL_B * eta) / (NULL_B * eta),
        })
        cells[f"K{s}_eta{eta}"] = cell
    live = [c for c in cells.values() if "VOID" not in c]
    voids = sum(1 for c in cells.values() if "VOID" in c)  # count void CELLS once
    out = {"cells": cells, "n_void": voids}
    if voids > 1:
        out["VERDICT_i"] = "INCONCLUSIVE (>1 void cell)"
        return out
    med = float(np.median([c["rel_err_law"] for c in live]))
    med_a = float(np.median([c["rel_err_nullA"] for c in live]))
    med_b = float(np.median([c["rel_err_nullB"] for c in live]))
    within = all(c["rel_err_law"] <= REL_TOL for c in live)
    beats = med < med_a and med < med_b
    out.update({"median_rel_err_law": med, "median_rel_err_nullA": med_a,
                "median_rel_err_nullB": med_b, "all_within_25pct": within,
                "beats_both_nulls": beats,
                "PASS_i": bool(within and beats)})
    return out


def arm_t128(T=128):
    """Clause (ii): form test rho = c(T)*K*eta at T = 128, fresh run."""
    rng = np.random.default_rng(43)
    batches = make_data(rng, T)
    theta0 = init_params(rng)
    gate = e000(theta0, batches, ETA_REF)
    if not gate["PASS"]:
        return {"E000_gate": gate, "ABORT": "E-000 gate failed for T=128 arm"}
    X_eval = np.random.default_rng(7).normal(size=(N_EVAL, D_IN))
    word_full = [int(t) for t in np.random.default_rng(103).permutation(T)]
    K_full = len(inversion_pairs(word_full))
    if K_full < max(T128_GRID):
        return {"ABORT": f"full word has only {K_full} inversions"}
    cells = {}
    for s in T128_GRID:
        word = prefix_word(word_full, s, T)
        K = len(inversion_pairs(word))
        if K != s:
            return {"ABORT": f"prefix {s}: word has {K} inversions"}
        cell = transport_cell(theta0, batches, ETA_REF, T, word, X_eval)
        if cell["param_residual"] < 1 and cell["audit_median_rel_err"] > 0.02:
            cell["VOID"] = "audit gate failed at non-overshoot cell"
        cells[s] = cell
    live_K = np.array([s for s in T128_GRID if "VOID" not in cells[s]])
    rho = np.array([cells[s]["param_residual"] for s in live_K])
    m = float((rho * live_K).sum() / (live_K * live_K).sum())
    ss_res = float(((rho - m * live_K) ** 2).sum())
    ss_tot = float(((rho - rho.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    mono = bool(np.all(np.diff(rho) > 0))
    c128 = m / ETA_REF
    return {"E000_gate": gate, "K_full": K_full, "cells": cells,
            "slope_m": m, "R2_centered": r2, "monotone": mono,
            "c_T128": c128, "c_ratio_128_over_64": c128 / C_FROZEN,
            "PASS_ii": bool(r2 >= R2_MIN and mono and len(live_K) == len(T128_GRID))}


def main():
    T = 64
    rng = np.random.default_rng(42)
    batches = make_data(rng, T)
    theta0 = init_params(rng)

    r000 = e000(theta0, batches, ETA_REF)
    print("E-000 gate (T=64 arm):", json.dumps(r000, indent=2))
    if not r000["PASS"]:
        print("E-000 FAILED — aborting before subject measurements.")
        sys.exit(1)

    res_i = arm_t64(theta0, batches, T)
    print("E-008 arm (i):", json.dumps(res_i, indent=2))
    res_ii = arm_t128()
    print("E-008 arm (ii):", json.dumps(res_ii, indent=2))
    if "ABORT" in res_i or "ABORT" in res_ii:
        sys.exit(1)

    r_all = [c["r"] for c in res_i["cells"].values()] + \
            [res_ii["cells"][s]["r"] for s in T128_GRID]
    secondary_r = {"max_r": float(max(r_all)),
                   "all_r_below_1": bool(max(r_all) < 1)}
    print("E-008 secondary (iii):", json.dumps(secondary_r, indent=2))

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    out = {"commit": commit, "c_frozen": C_FROZEN, "null_a": NULL_A,
           "null_b": NULL_B, "seed_data_t64": 42, "seed_data_t128": 43,
           "seed_perm_t64": 102, "seed_perm_t128": 103, "seed_eval": 7,
           "E000_gate_t64": r000, "arm_i": res_i, "arm_ii": res_ii,
           "secondary_iii": secondary_r}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "results", "raw"), exist_ok=True)
    path = os.path.join(root, "results", "raw", "pilot_edge2_2026-07-20.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("written:", path)


if __name__ == "__main__":
    main()
