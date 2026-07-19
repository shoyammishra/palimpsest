"""M3.1 E-005: budget compression I -- batched forward-mode JVP transport.

Replaces E-003's per-inversion tail replay (O(KT) grad evals) with the single
batched forward-JVP chain declared in design.md 2.2: every reference defect
dbar_ij is injected into one carried vector at its position a+2, and the
vector is pushed through the trajectory with one FD-HVP per step
(Du_t(x) w = w - eta H_t(x) w). The chain is exactly linear, so the batched
correction equals the sum of the individual JVP transports; it differs from
E-003's replay correction only by the per-defect linearization error bounded
by Lemma K-2 and measured by E-001.

Instrument checks (2.2 validation gate, pre-registered in E-005):
  (i)  per-sample audit: individual JVP vs individual branched replay on
       10 sampled inversions per severity (gate at w4/w16, report at full);
  (ii) batched-vs-summed exactness on the audited subset (float precision --
       any discrepancy is a code bug, abort);
  (iii) whole-correction comparison vs the replay correction.

Budget: dual-reported per D-012 -- rule (b) raw uncached counts (E-003's
rule) and rule (c) grad-eval-equivalents (trajectory grads free from
checkpoints, caching counted once, FD-HVP = 2 evals). Audit evals are
instrument cost, never operator budget.

Hypothesis pre-registered in docs/experiment_log.md (E-005) BEFORE this run.
E-000 self-checks gate the run (instrument before subject).
"""

import json
import os
import subprocess
import sys

import numpy as np

from pilot_defects import (D_IN, e000, grad, init_params, make_data, pack,
                           run_word, unpack)
from pilot_transport import (block_shuffle_word, checkpoints, d_f,
                             inversion_pairs, k31_transport)

HVP_EPS = 1e-5
N_EVAL = 1024
N_AUDIT = 10


def hvp_exact(theta, X, y, v):
    """Analytic Hessian-vector product: forward-over-reverse through the
    manual backprop of pilot_defects.grad. Exactly linear in v — this is what
    makes the batched-vs-summed check a true code-bug detector (E-005
    pre-run amendment)."""
    W1, b1, w2, b2 = unpack(theta)
    V1, c1, v2, c2 = unpack(v)
    n = X.shape[0]
    z = X @ W1.T + b1
    a = np.tanh(z)
    r = a @ w2 + b2 - y
    dpred = r / n
    za = 1 - a ** 2
    zd = X @ V1.T + c1
    ad = za * zd
    rd = ad @ w2 + a @ v2 + c2
    dpred_d = rd / n
    dw2_d = ad.T @ dpred + a.T @ dpred_d
    db2_d = np.array([dpred_d.sum()])
    dz_d = ((np.outer(dpred_d, w2) + np.outer(dpred, v2)) * za
            + np.outer(dpred, w2) * (-2 * a * ad))
    dW1_d = dz_d.T @ X
    db1_d = dz_d.sum(axis=0)
    return pack(dW1_d, db1_d, dw2_d, db2_d)


def hvp_fd(theta, batch, w, eps=HVP_EPS):
    """FD Hessian-vector product with normalized direction (e000-validated).
    Instrument cross-check of hvp_exact only — not used in the chain."""
    nw = np.linalg.norm(w)
    if nw == 0:
        return np.zeros_like(w)
    u = w / nw
    return nw * (grad(theta + eps * u, *batch) - grad(theta - eps * u, *batch)) / (2 * eps)


def pair_defects(chk, word2, batches, eta):
    """dbar for every inversion, bit-identical to E-003's computation, but
    with D-012 rule-(c) bookkeeping: the first step of the ab order is the
    trajectory step (free, = chk[a+1]); g_b(chk[a]) and g_b(chk[a+1]) are
    cacheable; g_a at the ba-order midpoint is pair-specific.

    Returns (defects, counts): defects = list of (a, b, dbar); counts holds
    rule-(b) and rule-(c) eval counts for the pair-defect stage.
    """
    inv = inversion_pairs(word2)
    cache = {}
    cache_misses = 0

    def g_cached(b, pos):
        nonlocal cache_misses
        key = (b, pos)
        if key not in cache:
            cache[key] = grad(chk[pos], *batches[b])
            cache_misses += 1
        return cache[key]

    defects = []
    pair_specific = 0
    for a, b in inv:
        # order ab: chk[a] -> chk[a+1] (trajectory step, free) -> y_ab
        y_ab = chk[a + 1] - eta * g_cached(b, a + 1)
        # order ba: chk[a] -> mid -> y_ba (second eval is pair-specific)
        mid = chk[a] - eta * g_cached(b, a)
        y_ba = mid - eta * grad(mid, *batches[a])
        pair_specific += 1
        defects.append((a, b, y_ba - y_ab))
    counts = {"rule_b": 4 * len(inv),
              "rule_c": cache_misses + pair_specific,
              "cache_misses": cache_misses, "pair_specific": pair_specific}
    return defects, counts


def jvp_chain(chk, batches, eta, T, injections):
    """Push all injected defects through one shared forward-JVP chain.

    injections: list of (position, vector) with position in [0, T]; vectors
    are added to the carried w when the chain reaches their position, then
    propagated by w <- w - eta * H_t(chk[t]) w for each remaining step.
    Returns (correction, hvp_steps).
    """
    by_pos = {}
    for pos, vec in injections:
        by_pos.setdefault(pos, []).append(vec)
    start = min(by_pos)
    w = np.zeros_like(chk[0])
    hvp_steps = 0
    for t in range(start, T):
        for vec in by_pos.get(t, ()):
            w = w + vec
        w = w - eta * hvp_exact(chk[t], batches[t][0], batches[t][1], w)
        hvp_steps += 1
    for vec in by_pos.get(T, ()):
        w = w + vec
    return w, hvp_steps


def replay_transport_one(chk, batches, eta, T, a, dbar):
    """E-003's per-inversion transported defect (ground truth for the audit)."""
    tail = list(range(a + 2, T))
    return run_word(chk[a + 2] + dbar, tail, batches, eta) - chk[-1]


def e005(theta0, batches, eta, T, e003_reference):
    eval_rng = np.random.default_rng(7)
    X_eval = eval_rng.normal(size=(N_EVAL, D_IN))
    chk = checkpoints(theta0, batches, eta, T)
    theta1 = chk[-1]
    results = {"eta": eta, "T": T, "hvp_eps": HVP_EPS, "severities": {}}

    severities = {
        "w4": block_shuffle_word(T, 4, np.random.default_rng(100)),
        "w16": block_shuffle_word(T, 16, np.random.default_rng(101)),
        "full": list(np.random.default_rng(102).permutation(T)),
    }
    for name, word2 in severities.items():
        word2 = [int(t) for t in word2]
        theta2 = run_word(theta0, word2, batches, eta)
        gap_f = d_f(theta1, theta2, X_eval)

        defects, pair_counts = pair_defects(chk, word2, batches, eta)
        injections = [(a + 2, dbar) for a, b, dbar in defects]

        # inline instrument check: analytic HVP vs the e000-validated FD-HVP,
        # once per severity, on the first injected defect at its checkpoint
        a0, _, dbar0 = defects[0]
        t0 = min(a0 + 2, T - 1)
        h_ex = hvp_exact(chk[t0], batches[t0][0], batches[t0][1], dbar0)
        h_fd = hvp_fd(chk[t0], batches[t0], dbar0)
        hvp_xcheck = float(np.linalg.norm(h_ex - h_fd)
                           / max(np.linalg.norm(h_fd), 1e-300))
        if hvp_xcheck > 1e-4:
            results["severities"][name] = {
                "ABORT": f"analytic-vs-FD HVP mismatch: {hvp_xcheck}"}
            return results

        corr, hvp_steps = jvp_chain(chk, batches, eta, T, injections)
        t_op = theta1 + corr

        # replay operator re-run for the correction-vector comparison (iii)
        t_replay, replay_info = k31_transport(chk, word2, batches, eta, T)
        corr_replay = t_replay - theta1

        # (i) per-sample audit: individual JVP vs individual replay
        K = len(defects)
        audit_rng = np.random.default_rng(300)
        audit_idx = sorted(audit_rng.choice(K, min(N_AUDIT, K), replace=False))
        audit_errs = []
        summed_jvp = np.zeros_like(theta1)
        audit_subset = []
        for i in audit_idx:
            a, b, dbar = defects[i]
            d_jvp, _ = jvp_chain(chk, batches, eta, T, [(a + 2, dbar)])
            d_rep = replay_transport_one(chk, batches, eta, T, a, dbar)
            audit_errs.append(float(np.linalg.norm(d_jvp - d_rep)
                                    / max(np.linalg.norm(d_rep), 1e-300)))
            summed_jvp += d_jvp
            audit_subset.append((a + 2, dbar))
        # (ii) batched-vs-summed exactness on the audited subset
        batched_subset, _ = jvp_chain(chk, batches, eta, T, audit_subset)
        linearity_resid = float(np.linalg.norm(batched_subset - summed_jvp)
                                / max(np.linalg.norm(summed_jvp), 1e-300))

        r_jvp = d_f(t_op, theta2, X_eval) / gap_f
        r_replay = d_f(t_replay, theta2, X_eval) / gap_f
        budget_b = pair_counts["rule_b"] + 2 * hvp_steps
        budget_c = pair_counts["rule_c"] + 2 * hvp_steps
        cos = float(corr @ corr_replay
                    / (np.linalg.norm(corr) * np.linalg.norm(corr_replay)))
        results["severities"][name] = {
            "K_inversions": K,
            "hvp_xcheck_rel_err": hvp_xcheck,
            "gap_f_do_nothing": gap_f,
            "r_jvp": r_jvp,
            "r_replay_rerun": r_replay,
            "r_replay_e003": e003_reference[name],
            "delta_r": r_jvp - r_replay,
            "param_residual_jvp": float(np.linalg.norm(t_op - theta2)
                                        / np.linalg.norm(theta1 - theta2)),
            "corr_cosine_vs_replay": cos,
            "corr_norm_ratio_vs_replay": float(np.linalg.norm(corr)
                                               / np.linalg.norm(corr_replay)),
            "audit_rel_errs": audit_errs,
            "audit_median_rel_err": float(np.median(audit_errs)),
            "audit_max_rel_err": float(max(audit_errs)),
            "linearity_residual": linearity_resid,
            "budget": {
                "rule_b_jvp": budget_b,
                "rule_c_jvp": budget_c,
                "rule_b_replay_e003": replay_info["grad_evals"],
                "hvp_steps": hvp_steps,
                "pair_counts": pair_counts,
                "vs_retrain_rule_b": budget_b / T,
                "vs_retrain_rule_c": budget_c / T,
                "compression_rule_b": replay_info["grad_evals"] / budget_b,
            },
        }
        if linearity_resid > 1e-10:
            results["severities"][name]["ABORT"] = (
                "batched chain != sum of individual JVPs — code bug")
            return results
    return results


def main():
    T = 64
    eta = 0.05
    rng = np.random.default_rng(42)
    batches = make_data(rng, T)
    theta0 = init_params(rng)

    r000 = e000(theta0, batches, eta)
    print("E-000 gate:", json.dumps(r000, indent=2))
    if not r000["PASS"]:
        print("E-000 FAILED — aborting before subject measurements.")
        sys.exit(1)

    e003_reference = {"w4": 0.145, "w16": 0.265, "full": 0.608}
    r005 = e005(theta0, batches, eta, T, e003_reference)
    print("E-005 batched JVP transport:", json.dumps(r005, indent=2))
    if any("ABORT" in s for s in r005["severities"].values()):
        sys.exit(1)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    out = {"commit": commit, "seed_data": 42, "seeds_perm": [100, 101, 102],
           "seed_eval": 7, "seed_audit": 300, "E000_gate": r000, "E005": r005}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "results", "raw"), exist_ok=True)
    path = os.path.join(root, "results", "raw",
                        "pilot_transport_batched_2026-07-19.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("written:", path)


if __name__ == "__main__":
    main()
