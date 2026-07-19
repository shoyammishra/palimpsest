"""M3.1 E-006: budget compression II -- importance-weighted inversion sampling.

With the chain fixed-cost (E-005), the only remaining budget lever is sampling
M < K inversions. The with-replacement importance estimator of the correction,

    Sigma_hat = (1/M) sum_m  Dbar_{k_m} / p_{k_m},

is unbiased for sum_k Dbar_k by exact chain linearity, and its RMS relative
error obeys the algebraic identity  relerr(M) = A_p / sqrt(M)  with
A_p^2 = (sum_k ||Dbar_k||^2 / p_k) / ||sum_k Dbar_k||^2 - 1.  E-003's
cancellation ratios (0.12-0.22) pre-derive M* >> K at every severity, so the
pre-registered headline is a NEGATIVE: sampling is dominated by enumeration
at pilot scale (clause 2), with the variance law as instrument check
(clause 1) and the severity trend of A/sqrt(K) as the extrapolation
deliverable (clause 3).

Per-inversion transported defects Dbar_k are computed once per severity via
single-injection JVP chains (instrument cost per D-012, not operator budget);
every estimator afterwards is a weighted subset-sum of that matrix.

Hypothesis pre-registered in docs/experiment_log.md (E-006) BEFORE this run.
E-000 self-checks gate the run (instrument before subject).
"""

import json
import os
import subprocess
import sys

import numpy as np

from pilot_defects import (D_IN, e000, init_params, make_data, run_word)
from pilot_transport import (block_shuffle_word, checkpoints, d_f,
                             inversion_pairs)
from pilot_transport_batched import jvp_chain, pair_defects

N_EVAL = 1024
N_SEEDS = 100
N_SEEDS_RECHECK = 1000
DELTA_R_GATE = 0.05
M_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512]
SEED_FAMILY = 500


def delta_matrix(chk, batches, eta, T, defects):
    """Per-inversion transported defects Dbar_k (K x dim), one JVP chain each."""
    rows = []
    for a, b, dbar in defects:
        d_k, _ = jvp_chain(chk, batches, eta, T, [(a + 2, dbar)])
        rows.append(d_k)
    return np.array(rows)


def scheme_weights(defects, Dbar, chk, eta):
    """Sampling weight vectors (unnormalized) for the four pre-declared schemes.

    Free proxy uses only trajectory gradients g_t = (chk[t]-chk[t+1])/eta,
    which are recoverable from the I_1 checkpoints (D-012 rule (c): zero cost).
    """
    K = len(defects)
    traj_g_norm = np.array([np.linalg.norm(chk[t] - chk[t + 1]) / eta
                            for t in range(len(chk) - 1)])
    return {
        "uniform": np.ones(K),
        "proxy_gagb": np.array([traj_g_norm[a] * traj_g_norm[b]
                                for a, b, _ in defects]),
        "pair_oracle": np.array([np.linalg.norm(dbar) for _, _, dbar in defects]),
        "oracle": np.linalg.norm(Dbar, axis=1),
    }


def predicted_A(p, norms, mu_norm):
    return float(np.sqrt(np.sum(norms ** 2 / p) / mu_norm ** 2 - 1))


def sampled_budget(defects, idx, T):
    """Deployed-operator budget for the sampled subset, both D-012 rules.

    Rule (b): 4 evals per unique sampled inversion (duplicates reuse dbar).
    Rule (c): unique cacheable grads (b at positions a and a+1) + one
    pair-specific eval per unique inversion. Chain runs from the earliest
    sampled injection; one HVP step = 2 evals under both rules.
    """
    uniq = sorted(set(int(i) for i in idx))
    cache_keys = set()
    for i in uniq:
        a, b, _ = defects[i]
        cache_keys.add((b, a))
        cache_keys.add((b, a + 1))
    chain_steps = T - min(defects[i][0] + 2 for i in uniq)
    return {"rule_b": 4 * len(uniq) + 2 * chain_steps,
            "rule_c": len(cache_keys) + len(uniq) + 2 * chain_steps}


def run_cell(Dbar, p, mu, M, n_seeds, seed_key, theta1, theta2, gap_f, X_eval,
             defects, T):
    """One (scheme, M) cell: n_seeds independent M-sample estimators."""
    K = Dbar.shape[0]
    relerrs, rs, budgets_b, budgets_c = [], [], [], []
    for s in range(n_seeds):
        rng = np.random.default_rng([SEED_FAMILY] + seed_key + [s])
        idx = rng.choice(K, M, p=p, replace=True)
        est = (Dbar[idx] / (M * p[idx][:, None])).sum(axis=0)
        relerrs.append(np.linalg.norm(est - mu) / np.linalg.norm(mu))
        rs.append(d_f(theta1 + est, theta2, X_eval) / gap_f)
        bud = sampled_budget(defects, idx, T)
        budgets_b.append(bud["rule_b"])
        budgets_c.append(bud["rule_c"])
    return (np.array(relerrs), np.array(rs),
            float(np.mean(budgets_b)), float(np.mean(budgets_c)))


def e006(theta0, batches, eta, T, e005_reference):
    eval_rng = np.random.default_rng(7)
    X_eval = eval_rng.normal(size=(N_EVAL, D_IN))
    chk = checkpoints(theta0, batches, eta, T)
    theta1 = chk[-1]
    results = {"eta": eta, "T": T, "n_seeds": N_SEEDS,
               "delta_r_gate": DELTA_R_GATE, "severities": {}}

    severities = {
        "w4": block_shuffle_word(T, 4, np.random.default_rng(100)),
        "w16": block_shuffle_word(T, 16, np.random.default_rng(101)),
        "full": list(np.random.default_rng(102).permutation(T)),
    }
    for si, (name, word2) in enumerate(severities.items()):
        word2 = [int(t) for t in word2]
        theta2 = run_word(theta0, word2, batches, eta)
        gap_f = d_f(theta1, theta2, X_eval)
        gap_p = float(np.linalg.norm(theta1 - theta2))

        defects, _ = pair_defects(chk, word2, batches, eta)
        K = len(defects)
        Dbar = delta_matrix(chk, batches, eta, T, defects)
        mu = Dbar.sum(axis=0)
        norms = np.linalg.norm(Dbar, axis=1)

        # (ii) reconstruction identity vs the E-005 batched chain (abort on fail)
        corr_batched, _ = jvp_chain(chk, batches, eta, T,
                                    [(a + 2, dbar) for a, b, dbar in defects])
        recon = float(np.linalg.norm(mu - corr_batched)
                      / np.linalg.norm(corr_batched))
        r_full = d_f(theta1 + mu, theta2, X_eval) / gap_f
        sev = {
            "K_inversions": K,
            "gap_f_do_nothing": gap_f,
            "reconstruction_rel_err": recon,
            "r_full_enumeration": r_full,
            "r_jvp_e005": e005_reference[name],
            "cancellation_ratio_jvp": float(np.linalg.norm(mu) / norms.sum()),
            "corr_over_gap_p": float(np.linalg.norm(mu) / gap_p),
            "schemes": {}, "truncation": {},
        }
        if recon > 1e-10 or abs(r_full - e005_reference[name]) > 1e-3:
            sev["ABORT"] = "Dbar matrix fails reconstruction vs E-005 chain"
            results["severities"][name] = sev
            return results

        weights = scheme_weights(defects, Dbar, chk, eta)
        # proxy-quality diagnostics (Spearman via rank correlation)
        def spearman(x, y):
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(y)).astype(float)
            return float(np.corrcoef(rx, ry)[0, 1])
        sev["proxy_spearman_vs_pair_norm"] = spearman(weights["proxy_gagb"],
                                                      weights["pair_oracle"])
        sev["proxy_spearman_vs_oracle"] = spearman(weights["proxy_gagb"], norms)
        sev["pair_norm_spearman_vs_oracle"] = spearman(weights["pair_oracle"],
                                                       norms)

        m_grid = [M for M in M_GRID if M < K]
        gamma_x, gamma_y = [], []
        for ci, (sname, w) in enumerate(weights.items()):
            p = w / w.sum()
            A = predicted_A(p, norms, float(np.linalg.norm(mu)))
            cells = {}
            for M in m_grid:
                relerrs, rs, bud_b, bud_c = run_cell(
                    Dbar, p, mu, M, N_SEEDS, [si, ci, M], theta1, theta2,
                    gap_f, X_eval, defects, T)
                rms = float(np.sqrt(np.mean(relerrs ** 2)))
                pred = A / np.sqrt(M)
                cell = {
                    "rms_relerr": rms, "pred_relerr": pred,
                    "law_ratio": rms / pred,
                    "r_median": float(np.median(rs)),
                    "r_mean_sq": float(np.mean(rs ** 2)),
                    "r_q10": float(np.quantile(rs, 0.1)),
                    "r_q90": float(np.quantile(rs, 0.9)),
                    "budget_rule_b_mean": bud_b, "budget_rule_c_mean": bud_c,
                }
                # (iii) variance-law gate; failing cells re-run at S=1000
                if abs(cell["law_ratio"] - 1) > 0.35:
                    relerrs2, rs2, _, _ = run_cell(
                        Dbar, p, mu, M, N_SEEDS_RECHECK, [si, ci, M, 9],
                        theta1, theta2, gap_f, X_eval, defects, T)
                    rms2 = float(np.sqrt(np.mean(relerrs2 ** 2)))
                    cell["law_ratio_recheck_s1000"] = rms2 / pred
                    if abs(rms2 / pred - 1) > 0.15:
                        cell["ABORT"] = "variance-law identity violated"
                        cells[M] = cell
                        sev["schemes"][sname] = {"A_pred": A, "cells": cells}
                        sev["ABORT"] = "sampler bug: variance law failed"
                        results["severities"][name] = sev
                        return results
                cells[M] = cell
                # functional-sensitivity fit points: E||n||^2 vs E[d_f^2] excess
                gamma_x.append((pred * np.linalg.norm(mu)) ** 2)
                gamma_y.append((cell["r_mean_sq"] - r_full ** 2) * gap_f ** 2)
            sev["schemes"][sname] = {"A_pred": A, "cells": cells}

        # gamma: d_f-per-param-norm sensitivity, least squares through origin
        gx, gy = np.array(gamma_x), np.array(gamma_y)
        gamma_sq = float(gx @ gy / (gx @ gx))
        sev["gamma_fit"] = float(np.sqrt(max(gamma_sq, 0.0)))
        # M* extrapolation per scheme from the fitted mapping:
        # median r <= r_full + gate  <=>  gamma^2 (A/sqrt(M))^2 ||mu||^2
        #                                 <= ((r_full+gate)^2 - r_full^2) gap_f^2
        r_gate = r_full + DELTA_R_GATE
        noise_budget_sq = (r_gate ** 2 - r_full ** 2) * gap_f ** 2
        for sname in weights:
            A = sev["schemes"][sname]["A_pred"]
            m_star = (gamma_sq * A ** 2 * float(np.linalg.norm(mu)) ** 2
                      / noise_budget_sq)
            sev["schemes"][sname]["M_star_extrapolated"] = float(m_star)
            sev["schemes"][sname]["M_star_over_K"] = float(m_star / K)

        # biased comparator: top-M truncation by ||Dbar|| (no reweighting)
        order = np.argsort(-norms)
        run_sum = np.cumsum(Dbar[order], axis=0)
        for M in m_grid:
            t_est = theta1 + run_sum[M - 1]
            sev["truncation"][M] = {
                "r": float(d_f(t_est, theta2, X_eval) / gap_f),
                "mass_fraction": float(norms[order][:M].sum() / norms.sum()),
            }
        sev["A_oracle_over_sqrtK"] = float(
            sev["schemes"]["oracle"]["A_pred"] / np.sqrt(K))
        results["severities"][name] = sev
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

    e005_reference = {"w4": 0.1457, "w16": 0.2655, "full": 0.6081}
    r006 = e006(theta0, batches, eta, T, e005_reference)
    print("E-006 inversion sampling:", json.dumps(r006, indent=2))
    if any("ABORT" in s for s in r006["severities"].values()):
        sys.exit(1)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    out = {"commit": commit, "seed_data": 42, "seeds_perm": [100, 101, 102],
           "seed_eval": 7, "seed_family_sampling": SEED_FAMILY,
           "E000_gate": r000, "E006": r006}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "results", "raw"), exist_ok=True)
    path = os.path.join(root, "results", "raw",
                        "pilot_sampling_2026-07-19.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("written:", path)


if __name__ == "__main__":
    main()
