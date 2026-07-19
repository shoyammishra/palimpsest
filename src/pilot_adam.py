"""M3.1 E-004: Adam-normalization eta-sweep (K-4d check).

Does the Theta(eta) order-defect law (K-4b, confirmed for linear EMA readout
in E-002) survive Adam's *nonlinear* state readout? Index-free smooth-
normalized Adam per K-4a/K-4d:

    m+ = b1 m + (1-b1) g;  v+ = b2 v + (1-b2) g^2;
    th+ = th - eta * m+ / sqrt(v+ + ehat^2)        (no bias correction)

Two pre-declared conditions sharing one warm state (theta_w, m_w, v_w):
  (i)  b1=0.9, b2=0.9  -- full Adam-class
  (ii) b1=0,   b2=0.9  -- normalization-only (RMSprop-like): under a linear
       readout b1=0 gives slope 2 (E-002), so slope 1 here can come only from
       the v-channel through the nonlinear readout -- K-4d's mechanism claim.

Instrument check (independent code path): frozen-theta state recursion
(theta never moves; m, v recur; readouts summed) predicts the leading-order
defect delta_pred = eta * (sum_ab N - sum_ba N); measured/predicted ratio
and cosine at the smallest eta must pass before slopes are interpretable.

Hypothesis pre-registered in docs/experiment_log.md (E-004) BEFORE this run.
E-000 self-checks gate the run (instrument before subject).
"""

import json
import os
import subprocess
import sys

import numpy as np

from pilot_defects import e000, grad, init_params, make_data, run_word

EHAT = 1e-8
ETAS = np.logspace(-5, -3, 7)
NOISE_FLOOR = 1e-10


def adam_pair(theta, m, v, first, second, batches, eta, b1, b2):
    """Two index-free smooth-normalized Adam steps (K-4a/K-4d variant)."""
    for t in (first, second):
        g = grad(theta, *batches[t])
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        theta = theta - eta * m / np.sqrt(v + EHAT ** 2)
    return theta, m, v


def frozen_readout_sum(theta, m, v, order, batches, b1, b2):
    """State recursion at frozen theta; returns the summed step readouts.

    Different code path from adam_pair: theta never moves, so the sum is the
    exact eta->0 limit of (theta - th_end)/eta for the given order.
    """
    s = np.zeros_like(theta)
    for t in order:
        g = grad(theta, *batches[t])
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        s += m / np.sqrt(v + EHAT ** 2)
    return s


def warm_state(theta0, batches, eta_burn=0.05, beta_m=0.9, beta_v=0.9):
    """E-002's warm start extended with a v-channel: 5 SGD steps for theta,
    EMA of g / g^2 along that same trajectory for m / v."""
    burn = [2, 3, 4, 5, 6]
    theta = run_word(theta0, burn, batches, eta_burn)
    m = np.zeros_like(theta0)
    v = np.zeros_like(theta0)
    tmp = theta0
    for t in burn:
        g = grad(tmp, *batches[t])
        m = beta_m * m + (1 - beta_m) * g
        v = beta_v * v + (1 - beta_v) * g * g
        tmp = tmp - eta_burn * m
    return theta, m, v


def loglog_slope(etas, norms):
    sl, ic = np.polyfit(np.log(etas), np.log(norms), 1)
    pred = np.polyval([sl, ic], np.log(etas))
    ss_res = np.sum((np.log(norms) - pred) ** 2)
    ss_tot = np.sum((np.log(norms) - np.mean(np.log(norms))) ** 2)
    return float(sl), float(1 - ss_res / ss_tot)


def e004(theta0, batches):
    theta, m, v = warm_state(theta0, batches)
    results = {"etas": list(map(float, ETAS)), "ehat": EHAT,
               "warm_v_min": float(v.min()), "warm_v_max": float(v.max()),
               "conditions": {}}
    for b1, b2 in ((0.9, 0.9), (0.0, 0.9)):
        dth, dm, dv = [], [], []
        for eta in ETAS:
            th_ab, m_ab, v_ab = adam_pair(theta, m, v, 0, 1, batches, eta, b1, b2)
            th_ba, m_ba, v_ba = adam_pair(theta, m, v, 1, 0, batches, eta, b1, b2)
            dth.append(np.linalg.norm(th_ba - th_ab))
            dm.append(np.linalg.norm(m_ba - m_ab))
            dv.append(np.linalg.norm(v_ba - v_ab))
        sl_th, r2_th = loglog_slope(ETAS, dth)
        sl_m, _ = loglog_slope(ETAS, dm)
        sl_v, _ = loglog_slope(ETAS, dv)
        # frozen-theta constant-and-direction check at the smallest eta
        s_ab = frozen_readout_sum(theta, m, v, (0, 1), batches, b1, b2)
        s_ba = frozen_readout_sum(theta, m, v, (1, 0), batches, b1, b2)
        pred = ETAS[0] * (s_ab - s_ba)
        th_ab, _, _ = adam_pair(theta, m, v, 0, 1, batches, ETAS[0], b1, b2)
        th_ba, _, _ = adam_pair(theta, m, v, 1, 0, batches, ETAS[0], b1, b2)
        meas = th_ba - th_ab
        ratio = float(np.linalg.norm(meas) / np.linalg.norm(pred))
        cosine = float(meas @ pred / (np.linalg.norm(meas) * np.linalg.norm(pred)))
        # pre-declared diagnostic: refit on the lowest 4 eta points
        sl_lo, r2_lo = loglog_slope(ETAS[:4], dth[:4])
        results["conditions"][f"b1={b1}_b2={b2}"] = {
            "slope_theta": sl_th, "r2_theta": r2_th,
            "slope_theta_low4": sl_lo, "r2_theta_low4": r2_lo,
            "slope_m": sl_m, "slope_v": sl_v,
            "delta_theta_norms": list(map(float, dth)),
            "delta_m_norms": list(map(float, dm)),
            "delta_v_norms": list(map(float, dv)),
            "min_delta_theta": float(min(dth)),
            "noise_floor_ok": bool(min(dth) > NOISE_FLOOR),
            "frozen_check_ratio": ratio,
            "frozen_check_cosine": cosine,
            "frozen_check_pass": bool(0.9 <= ratio <= 1.1 and cosine >= 0.999),
        }
    return results


def main():
    T = 64
    rng = np.random.default_rng(42)
    batches = make_data(rng, T)
    theta0 = init_params(rng)

    r000 = e000(theta0, batches, 0.05)
    print("E-000 gate:", json.dumps(r000, indent=2))
    if not r000["PASS"]:
        print("E-000 FAILED — aborting before subject measurements.")
        sys.exit(1)

    r004 = e004(theta0, batches)
    print("E-004 Adam eta-sweep:", json.dumps(r004, indent=2))

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    out = {"commit": commit, "seed_data": 42, "seed_harness": 0,
           "E000_gate": r000, "E004": r004}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "results", "raw"), exist_ok=True)
    path = os.path.join(root, "results", "raw", "pilot_adam_2026-07-19.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("written:", path)


if __name__ == "__main__":
    main()
