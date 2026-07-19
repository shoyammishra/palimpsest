"""M3.1 pilot harness: swap defects, branched-replay transport, scaling checks.

Runs, in order (E-000 gate first — abort if it fails):
  E-000  instrument self-checks: gradcheck, commuting sanity, K-1 telescoping
  E-001  linearity of transported defects R(h) across defect scale h
  E-002  order-defect eta-scaling, SGD (beta=0) vs EMA momentum (beta=0.9)

numpy-only, float64, fully seeded. Hypotheses pre-registered in
docs/experiment_log.md; results written to results/raw/ and printed.
"""

import json
import os
import subprocess
import sys

import numpy as np

RNG = np.random.default_rng(0)
D_IN, HID, N_BATCH = 8, 24, 16  # ~241 params


# ---------------- model: 1-hidden-layer tanh MLP, manual backprop ----------------

def init_params(rng):
    W1 = rng.normal(0, 1 / np.sqrt(D_IN), (HID, D_IN))
    b1 = np.zeros(HID)
    w2 = rng.normal(0, 1 / np.sqrt(HID), HID)
    b2 = np.zeros(1)
    return pack(W1, b1, w2, b2)


def pack(W1, b1, w2, b2):
    return np.concatenate([W1.ravel(), b1, w2, b2])


def unpack(theta):
    i = 0
    W1 = theta[i:i + HID * D_IN].reshape(HID, D_IN); i += HID * D_IN
    b1 = theta[i:i + HID]; i += HID
    w2 = theta[i:i + HID]; i += HID
    b2 = theta[i:i + 1]
    return W1, b1, w2, b2


def loss(theta, X, y):
    W1, b1, w2, b2 = unpack(theta)
    a = np.tanh(X @ W1.T + b1)
    r = a @ w2 + b2 - y
    return 0.5 * np.mean(r ** 2)


def grad(theta, X, y):
    W1, b1, w2, b2 = unpack(theta)
    n = X.shape[0]
    z = X @ W1.T + b1
    a = np.tanh(z)
    r = a @ w2 + b2 - y
    dpred = r / n
    dw2 = a.T @ dpred
    db2 = np.array([dpred.sum()])
    dz = np.outer(dpred, w2) * (1 - a ** 2)
    dW1 = dz.T @ X
    db1 = dz.sum(axis=0)
    return pack(dW1, db1, dw2, db2)


# ---------------- update maps and words ----------------

def sgd_step(theta, batch, eta):
    X, y = batch
    return theta - eta * grad(theta, X, y)


def run_word(theta, word, batches, eta):
    for t in word:
        theta = sgd_step(theta, batches[t], eta)
    return theta


def momentum_pair(theta, m, first, second, batches, eta, beta):
    """Two EMA-momentum steps (m+ = beta m + (1-beta) g; th+ = th - eta m+)."""
    for t in (first, second):
        X, y = batches[t]
        m = beta * m + (1 - beta) * grad(theta, X, y)
        theta = theta - eta * m
    return theta, m


def make_data(rng, n_steps, return_teacher=False):
    teacher = init_params(rng)  # labels from a random teacher net + noise
    batches = []
    for _ in range(n_steps):
        X = rng.normal(size=(N_BATCH, D_IN))
        W1, b1, w2, b2 = unpack(teacher)
        y = np.tanh(X @ W1.T + b1) @ w2 + b2 + 0.1 * rng.normal(size=N_BATCH)
        batches.append((X, y))
    if return_teacher:
        return batches, teacher
    return batches


# ---------------- E-000: instrument self-checks ----------------

def bubble_schedule(word_from, word_to):
    """Adjacent-swap positions transforming word_from into word_to (reduced)."""
    cur, swaps = list(word_from), []
    for i, target in enumerate(word_to):
        j = cur.index(target, i)
        while j > i:
            cur[j - 1], cur[j] = cur[j], cur[j - 1]
            swaps.append(j - 1)
            j -= 1
    assert cur == list(word_to)
    return swaps


def e000(theta0, batches, eta):
    out = {}
    # 1. gradcheck vs central differences
    X, y = batches[0]
    g = grad(theta0, X, y)
    idx = RNG.choice(theta0.size, 25, replace=False)
    # eps=1e-5 balances FD truncation vs float64 cancellation noise; 1e-6 was
    # diagnosed (2026-07-19) to inject ~1e-10 absolute FD noise, which breaches
    # the relative tolerance on small-magnitude components — a checker artifact,
    # not a gradient bug (cross-validated at eps=1e-4/1e-5 and by the HVP check).
    eps, errs = 1e-5, []
    for i in idx:
        e = np.zeros_like(theta0); e[i] = eps
        fd = (loss(theta0 + e, X, y) - loss(theta0 - e, X, y)) / (2 * eps)
        errs.append(abs(fd - g[i]) / max(abs(fd), abs(g[i]), 1e-12))
    out["gradcheck_max_rel_err"] = float(max(errs))
    # 2. bracket cross-check: measured SGD pair defect vs the independent
    #    leading-order prediction eta^2 (H_a g_b - H_b g_a) via FD HVPs
    eta_small = 1e-3
    th_ab = sgd_step(sgd_step(theta0, batches[0], eta_small), batches[1], eta_small)
    th_ba = sgd_step(sgd_step(theta0, batches[1], eta_small), batches[0], eta_small)
    measured = th_ba - th_ab

    def hvp(batch, v, eps=1e-5):
        nv = np.linalg.norm(v)
        u = v / nv
        return nv * (grad(theta0 + eps * u, *batch) - grad(theta0 - eps * u, *batch)) / (2 * eps)

    ga0, gb0 = grad(theta0, *batches[0]), grad(theta0, *batches[1])
    predicted = eta_small ** 2 * (hvp(batches[0], gb0) - hvp(batches[1], ga0))
    out["bracket_xcheck_rel_err"] = float(
        np.linalg.norm(measured - predicted) / np.linalg.norm(measured))
    # 3. K-1 telescoping on a T=8 word with a random permutation
    T = 8
    w1 = list(range(T))
    w2 = list(RNG.permutation(T))
    swaps = bubble_schedule(w1, w2)
    cur = list(w1)
    total = np.zeros_like(theta0)
    for p in swaps:
        x = run_word(theta0, cur[:p], batches, eta)
        y_old = run_word(x, cur[p:p + 2], batches, eta)
        y_new = run_word(x, [cur[p + 1], cur[p]], batches, eta)
        delta = y_new - y_old
        tail = cur[p + 2:]
        Dk = run_word(y_old + delta, tail, batches, eta) - run_word(y_old, tail, batches, eta)
        total += Dk
        cur[p], cur[p + 1] = cur[p + 1], cur[p]
    direct = run_word(theta0, w2, batches, eta) - run_word(theta0, w1, batches, eta)
    out["telescoping_rel_err"] = float(
        np.linalg.norm(total - direct) / max(np.linalg.norm(direct), 1e-300))
    out["telescoping_gap_norm"] = float(np.linalg.norm(direct))
    out["n_swaps"] = len(swaps)
    ok = (out["gradcheck_max_rel_err"] < 1e-6
          and out["bracket_xcheck_rel_err"] < 0.05
          and out["telescoping_rel_err"] < 1e-9)
    out["PASS"] = bool(ok)
    return out


# ---------------- E-001: linearity of transported defects ----------------

def e001(theta0, batches, eta, T=64):
    positions = {"early": 4, "middle": 32, "late": 60}
    hs = [1e-3, 1e-2, 1e-1, 0.3, 1.0]
    word = list(range(T))
    results = {"eta": eta, "T": T, "hs": hs, "positions": {}}
    # training must be non-trivial for the check to mean anything
    results["loss_start"] = float(np.mean([loss(theta0, b[0], b[1]) for b in batches[:8]]))
    theta_end = run_word(theta0, word, batches, eta)
    results["loss_end"] = float(np.mean([loss(theta_end, b[0], b[1]) for b in batches[:8]]))
    for name, p in positions.items():
        x = run_word(theta0, word[:p], batches, eta)
        y_old = run_word(x, word[p:p + 2], batches, eta)
        y_new = run_word(x, [word[p + 1], word[p]], batches, eta)
        delta = y_new - y_old
        tail = word[p + 2:]
        base = run_word(y_old, tail, batches, eta)
        R = {}
        for h in hs:
            moved = run_word(y_old + h * delta, tail, batches, eta)
            R[h] = float(np.linalg.norm(moved - base) / h)
        Rmin = R[hs[0]]
        results["positions"][name] = {
            "swap_pos": p,
            "delta_norm": float(np.linalg.norm(delta)),
            "R": R,
            "R_ratio_to_hmin": {h: R[h] / Rmin for h in hs},
            "max_departure_pct": float(100 * max(abs(R[h] / Rmin - 1) for h in hs)),
        }
    return results


# ---------------- E-002: eta-scaling of the order defect ----------------

def e002(theta0, batches, eta_burn=0.05):
    etas = np.logspace(-4, -2, 7)
    # common warm start: 5 SGD burn-in steps for theta; EMA burn-in for m
    theta = run_word(theta0, [2, 3, 4, 5, 6], batches, eta_burn)
    beta_w = 0.9
    m = np.zeros_like(theta)
    tmp = theta0
    for t in [2, 3, 4, 5, 6]:
        m = beta_w * m + (1 - beta_w) * grad(tmp, *batches[t])
        tmp = tmp - eta_burn * m
    ga, gb = grad(theta, *batches[0]), grad(theta, *batches[1])
    results = {"etas": list(map(float, etas)), "conditions": {}}
    for beta in (0.0, 0.9):
        dth, dm = [], []
        for eta in etas:
            th_ab, m_ab = momentum_pair(theta, m, 0, 1, batches, eta, beta)
            th_ba, m_ba = momentum_pair(theta, m, 1, 0, batches, eta, beta)
            dth.append(np.linalg.norm(th_ba - th_ab))
            dm.append(np.linalg.norm(m_ba - m_ab))
        sl_th, ic = np.polyfit(np.log(etas), np.log(dth), 1)
        pred = np.polyval([sl_th, ic], np.log(etas))
        r2 = 1 - np.sum((np.log(dth) - pred) ** 2) / np.sum(
            (np.log(dth) - np.mean(np.log(dth))) ** 2)
        cond = {"slope_theta": float(sl_th), "r2_theta": float(r2),
                "delta_theta_norms": list(map(float, dth)),
                "delta_m_norms": list(map(float, dm))}
        if beta > 0:
            sl_m = np.polyfit(np.log(etas), np.log(dm), 1)[0]
            cond["slope_m"] = float(sl_m)
            # K-4b constant check at smallest eta: delta_theta ~= eta*b(1-b)(ga-gb)
            pred_vec = etas[0] * beta * (1 - beta) * (ga - gb)
            th_ab, _ = momentum_pair(theta, m, 0, 1, batches, etas[0], beta)
            th_ba, _ = momentum_pair(theta, m, 1, 0, batches, etas[0], beta)
            meas = th_ba - th_ab
            cond["k4b_ratio"] = float(np.linalg.norm(meas) / np.linalg.norm(pred_vec))
            cond["k4b_cosine"] = float(
                meas @ pred_vec / (np.linalg.norm(meas) * np.linalg.norm(pred_vec)))
        results["conditions"][f"beta={beta}"] = cond
    return results


def main():
    T = 64
    rng = np.random.default_rng(42)
    batches = make_data(rng, T)
    theta0 = init_params(rng)
    eta = 0.05

    r000 = e000(theta0, batches, eta)
    print("E-000 self-checks:", json.dumps(r000, indent=2))
    if not r000["PASS"]:
        print("E-000 FAILED — aborting before subject measurements.")
        sys.exit(1)

    r001 = e001(theta0, batches, eta, T=T)
    print("E-001 linearity:", json.dumps(r001, indent=2))
    r002 = e002(theta0, batches)
    print("E-002 eta-scaling:", json.dumps(r002, indent=2))

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    out = {"commit": commit, "seed_data": 42, "seed_harness": 0,
           "E000": r000, "E001": r001, "E002": r002}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "results", "raw"), exist_ok=True)
    path = os.path.join(root, "results", "raw", "pilot_defects_2026-07-19.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("written:", path)


if __name__ == "__main__":
    main()
