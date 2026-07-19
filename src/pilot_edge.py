"""M3.1 E-007: K-3.1 validity-edge probe.

Two pre-declared sub-sweeps, one variable each:
  (a) inversion count K via bubble-sort PREFIXES of the reduced word from
      identity toward the seed-102 full permutation (a prefix of s adjacent
      swaps has exactly K = s inversions, all a subset of the full
      permutation's) at fixed eta = 0.05 -- where does the parameter-space
      attribution overshoot rho = ||T(theta1)-theta2|| / ||theta1-theta2||
      cross 1?  s = 218 doubles as a structure probe against w16 (same K,
      different inversion set).
  (b) eta in {0.0125, 0.025} at the fixed full permutation -- is the
      overshoot eta-suppressed (K-3.1 asymptotically exact) or a K-wall?

Transport mechanism, pair-defect computation, and audit machinery are
bit-identical to E-005 (batched forward-JVP chain, exact HVP).

Hypothesis pre-registered in docs/experiment_log.md (E-007) BEFORE this run.
E-000 self-checks gate the run (instrument before subject).
"""

import json
import os
import subprocess
import sys

import numpy as np

from pilot_defects import (D_IN, bubble_schedule, e000, init_params,
                           make_data, run_word)
from pilot_transport import (checkpoints, d_f, inversion_pairs)
from pilot_transport_batched import (jvp_chain, pair_defects,
                                     replay_transport_one)

N_EVAL = 1024
N_AUDIT = 10
PREFIX_GRID = [218, 350, 500, 650, 800]
ETA_GRID = [0.0125, 0.025]
ETA_REF = 0.05


def prefix_word(word_full, s, T):
    """Word after the first s adjacent swaps of the reduced bubble word
    identity -> word_full. Has exactly s inversions (checked by caller)."""
    swaps = bubble_schedule(list(range(T)), word_full)
    cur = list(range(T))
    for p in swaps[:s]:
        cur[p], cur[p + 1] = cur[p + 1], cur[p]
    return cur


def transport_cell(theta0, batches, eta, T, word2, X_eval, audit_seed=300):
    """One (word, eta) cell: batched-JVP transport + per-sample audit."""
    chk = checkpoints(theta0, batches, eta, T)
    theta1 = chk[-1]
    theta2 = run_word(theta0, word2, batches, eta)
    gap_f = d_f(theta1, theta2, X_eval)
    gap_p = float(np.linalg.norm(theta1 - theta2))

    defects, pair_counts = pair_defects(chk, word2, batches, eta)
    K = len(defects)
    corr, hvp_steps = jvp_chain(chk, batches, eta, T,
                                [(a + 2, dbar) for a, b, dbar in defects])
    t_op = theta1 + corr

    audit_rng = np.random.default_rng(audit_seed)
    audit_idx = sorted(audit_rng.choice(K, min(N_AUDIT, K), replace=False))
    audit_errs = []
    for i in audit_idx:
        a, b, dbar = defects[i]
        d_jvp, _ = jvp_chain(chk, batches, eta, T, [(a + 2, dbar)])
        d_rep = replay_transport_one(chk, batches, eta, T, a, dbar)
        audit_errs.append(float(np.linalg.norm(d_jvp - d_rep)
                                / max(np.linalg.norm(d_rep), 1e-300)))

    return {
        "K_inversions": K,
        "gap_f_do_nothing": gap_f,
        "gap_f_above_noise": bool(gap_f > 1e-8),
        "gap_param_do_nothing": gap_p,
        "r": float(d_f(t_op, theta2, X_eval) / gap_f),
        "param_residual": float(np.linalg.norm(t_op - theta2) / gap_p),
        "corr_norm": float(np.linalg.norm(corr)),
        "audit_median_rel_err": float(np.median(audit_errs)),
        "audit_max_rel_err": float(max(audit_errs)),
        "budget": {"rule_b": pair_counts["rule_b"] + 2 * hvp_steps,
                   "rule_c": pair_counts["rule_c"] + 2 * hvp_steps},
    }


def e007(theta0, batches, T):
    eval_rng = np.random.default_rng(7)
    X_eval = eval_rng.normal(size=(N_EVAL, D_IN))
    word_full = [int(t) for t in np.random.default_rng(102).permutation(T)]
    results = {"T": T, "eta_ref": ETA_REF, "sweep_K": {}, "sweep_eta": {}}

    # (a) K-interpolation via bubble prefixes at eta = 0.05
    for s in PREFIX_GRID:
        word_s = prefix_word(word_full, s, T)
        K_check = len(inversion_pairs(word_s))
        if K_check != s:
            results["sweep_K"][s] = {
                "ABORT": f"prefix word has {K_check} inversions, expected {s}"}
            return results
        cell = transport_cell(theta0, batches, ETA_REF, T, word_s, X_eval)
        # audit gate at cells without overshoot (E-005 convention)
        if cell["param_residual"] < 1 and cell["audit_median_rel_err"] > 0.02:
            cell["ABORT"] = "per-sample audit gate failed at non-overshoot cell"
            results["sweep_K"][s] = cell
            return results
        results["sweep_K"][s] = cell

    # reference anchor: full permutation at eta = 0.05 must reproduce E-005
    anchor = transport_cell(theta0, batches, ETA_REF, T, word_full, X_eval)
    results["anchor_full_eta0.05"] = anchor
    if abs(anchor["r"] - 0.6081) > 1e-3 or abs(anchor["param_residual"] - 1.631) > 2e-3:
        results["ABORT"] = "reference anchor does not reproduce E-005"
        return results

    # (b) eta-sweep at the fixed full permutation
    for eta in ETA_GRID:
        cell = transport_cell(theta0, batches, eta, T, word_full, X_eval)
        if not cell["gap_f_above_noise"]:
            cell["VOID"] = "gap at noise floor (pre-declared void condition)"
        results["sweep_eta"][eta] = cell
    return results


def main():
    T = 64
    rng = np.random.default_rng(42)
    batches = make_data(rng, T)
    theta0 = init_params(rng)

    r000 = e000(theta0, batches, ETA_REF)
    print("E-000 gate:", json.dumps(r000, indent=2))
    if not r000["PASS"]:
        print("E-000 FAILED — aborting before subject measurements.")
        sys.exit(1)

    r007 = e007(theta0, batches, T)
    print("E-007 validity edge:", json.dumps(r007, indent=2))
    aborted = ("ABORT" in r007
               or any("ABORT" in c for c in r007["sweep_K"].values())
               or any("ABORT" in c for c in r007["sweep_eta"].values()))
    if aborted:
        sys.exit(1)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    out = {"commit": commit, "seed_data": 42, "seed_perm_full": 102,
           "seed_eval": 7, "seed_audit": 300, "E000_gate": r000, "E007": r007}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "results", "raw"), exist_ok=True)
    path = os.path.join(root, "results", "raw", "pilot_edge_2026-07-19.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("written:", path)


if __name__ == "__main__":
    main()
