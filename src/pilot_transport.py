"""M3.1 E-003: first K-aware transport attempt (D-011 operator via Cor K-3.1).

Operator: T(theta(H1)) = theta(H1) + sum_{(a,b) in Inv(pi)} Jbar_ab dbar_ab,
side information I_1 = {pi, H1 checkpoints} (D-011). Jbar*dbar is implemented
as replay transport along H1's own trajectory: inject dbar at H1's checkpoint
after the pair slot and replay H1's tail (one tail replay per inversion) --
first-order-equal to the JVP chain of K-3.1 and validated by E-001 linearity.
Exact (machine precision) when pi is a single adjacent transposition; that
case is the inline instrument check.

Compared against degenerate baselines (do-nothing / checkpoint-average /
light fine-tune) on the functional metric d_f (D-006): RMS prediction gap on
fresh eval inputs. Severity sweep pre-declared: block shuffles w=4, w=16,
full permutation.

Hypothesis pre-registered in docs/experiment_log.md (E-003) BEFORE this run.
E-000 self-checks gate the run (instrument before subject).
"""

import json
import os
import subprocess
import sys

import numpy as np

from pilot_defects import (D_IN, N_BATCH, e000, grad, init_params, loss,
                           make_data, run_word, sgd_step, unpack)

N_EVAL = 1024
FT_STEPS = 16


# ---------------- functional metric (D-006) ----------------

def predictions(theta, X):
    W1, b1, w2, b2 = unpack(theta)
    return np.tanh(X @ W1.T + b1) @ w2 + b2


def d_f(theta_a, theta_b, X_eval):
    diff = predictions(theta_a, X_eval) - predictions(theta_b, X_eval)
    return float(np.sqrt(np.mean(diff ** 2)))


# ---------------- permutations ----------------

def inversion_pairs(word2):
    """Pairs (a, b), a < b, whose relative order is reversed in word2 (H1 = identity)."""
    pos = {v: i for i, v in enumerate(word2)}
    T = len(word2)
    return [(a, b) for a in range(T) for b in range(a + 1, T) if pos[b] < pos[a]]


def block_shuffle_word(T, w, rng):
    word = []
    for s in range(0, T, w):
        word.extend(rng.permutation(range(s, min(s + w, T))).tolist())
    return word


# ---------------- the D-011 operator ----------------

def checkpoints(theta0, batches, eta, T):
    chk = [theta0.copy()]
    th = theta0
    for t in range(T):
        th = sgd_step(th, batches[t], eta)
        chk.append(th.copy())
    return chk


def k31_transport(chk, word2, batches, eta, T):
    """theta1 + sum of transported reference defects; budget in grad-evals.

    Per inversion (a, b) with u_a at position a of H1:
      dbar = (u_b then u_a)(chk[a]) - (u_a then u_b)(chk[a])   [new - old, K-1 sign]
      transported = Phi_{H1 tail from a+2}(chk[a+2] + dbar) - theta1
    run(chk[a+2], tail) == theta1 exactly, so the base replay is free.
    """
    theta1 = chk[-1]
    corr = np.zeros_like(theta1)
    inv = inversion_pairs(word2)
    grad_evals = 0
    k1_mass = 0.0
    for a, b in inv:
        x = chk[a]
        y_ab = run_word(x, [a, b], batches, eta)
        y_ba = run_word(x, [b, a], batches, eta)
        dbar = y_ba - y_ab
        tail = list(range(a + 2, T))
        moved = run_word(chk[a + 2] + dbar, tail, batches, eta)
        Dk = moved - theta1
        corr += Dk
        k1_mass += float(np.linalg.norm(Dk))
        grad_evals += 4 + len(tail)
    info = {"K": len(inv), "grad_evals": grad_evals,
            "K1_mass": k1_mass, "corr_norm": float(np.linalg.norm(corr))}
    return theta1 + corr, info


# ---------------- degenerate baselines (design.md 5.4) ----------------

def checkpoint_average(chk, word2):
    """Mean of the I_1 checkpoints the operator itself consumes, plus theta1."""
    used = sorted({a for a, _ in inversion_pairs(word2)})
    stack = [chk[a] for a in used] + [chk[-1]]
    return np.mean(stack, axis=0)


def fresh_batches(teacher, rng, n):
    W1, b1, w2, b2 = unpack(teacher)
    out = []
    for _ in range(n):
        X = rng.normal(size=(N_BATCH, D_IN))
        y = np.tanh(X @ W1.T + b1) @ w2 + b2 + 0.1 * rng.normal(size=N_BATCH)
        out.append((X, y))
    return out


def light_finetune(theta1, teacher, eta, rng):
    batches = fresh_batches(teacher, rng, FT_STEPS)
    th = theta1
    for X, y in batches:
        th = th - eta * grad(th, X, y)
    return th


# ---------------- E-003 ----------------

def e003(theta0, batches, teacher, eta, T):
    eval_rng = np.random.default_rng(7)
    X_eval = eval_rng.normal(size=(N_EVAL, D_IN))
    W1, b1, w2, b2 = unpack(teacher)
    y_eval = np.tanh(X_eval @ W1.T + b1) @ w2 + b2 + 0.1 * eval_rng.normal(size=N_EVAL)

    chk = checkpoints(theta0, batches, eta, T)
    theta1 = chk[-1]
    results = {"eta": eta, "T": T, "n_eval": N_EVAL, "severities": {}}

    # inline instrument check: single adjacent transposition => operator exact
    word_adj = list(range(T))
    word_adj[30], word_adj[31] = word_adj[31], word_adj[30]
    theta2_adj = run_word(theta0, word_adj, batches, eta)
    t_adj, _ = k31_transport(chk, word_adj, batches, eta, T)
    adj_rel = float(np.linalg.norm(t_adj - theta2_adj)
                    / np.linalg.norm(theta1 - theta2_adj))
    results["adjacent_check_rel_residual"] = adj_rel
    if adj_rel > 1e-10:
        results["ABORT"] = "adjacent-transposition exactness check failed"
        return results

    severities = {
        "w4": block_shuffle_word(T, 4, np.random.default_rng(100)),
        "w16": block_shuffle_word(T, 16, np.random.default_rng(101)),
        "full": list(np.random.default_rng(102).permutation(T)),
    }
    for name, word2 in severities.items():
        theta2 = run_word(theta0, [int(t) for t in word2], batches, eta)
        gap_f = d_f(theta1, theta2, X_eval)
        gap_p = float(np.linalg.norm(theta1 - theta2))
        t_op, info = k31_transport(chk, word2, batches, eta, T)
        t_avg = checkpoint_average(chk, word2)
        t_ft = light_finetune(theta1, teacher, eta, np.random.default_rng(200))

        def L(th):
            r = predictions(th, X_eval) - y_eval
            return float(0.5 * np.mean(r ** 2))

        results["severities"][name] = {
            "word": [int(t) for t in word2],
            "K_inversions": info["K"],
            "budget_grad_evals": info["grad_evals"],
            "budget_vs_retrain": info["grad_evals"] / T,
            "gap_f_do_nothing": gap_f,
            "gap_f_noise_floor_ok": bool(gap_f > 1e-8),
            "gap_param_do_nothing": gap_p,
            "r_operator": d_f(t_op, theta2, X_eval) / gap_f,
            "r_ckpt_average": d_f(t_avg, theta2, X_eval) / gap_f,
            "r_finetune": d_f(t_ft, theta2, X_eval) / gap_f,
            "param_residual_operator": float(np.linalg.norm(t_op - theta2) / gap_p),
            "K1_mass": info["K1_mass"],
            "corr_norm": info["corr_norm"],
            "cancellation_ratio": info["corr_norm"] / max(info["K1_mass"], 1e-300),
            "eval_loss": {"theta1": L(theta1), "theta2": L(theta2),
                          "operator": L(t_op), "ckpt_average": L(t_avg),
                          "finetune": L(t_ft)},
        }
    return results


def main():
    T = 64
    eta = 0.05
    rng = np.random.default_rng(42)
    batches, teacher = make_data(rng, T, return_teacher=True)
    theta0 = init_params(rng)

    r000 = e000(theta0, batches, eta)
    print("E-000 gate:", json.dumps(r000, indent=2))
    if not r000["PASS"]:
        print("E-000 FAILED — aborting before subject measurements.")
        sys.exit(1)

    r003 = e003(theta0, batches, teacher, eta, T)
    print("E-003 transport:", json.dumps(r003, indent=2))
    if "ABORT" in r003:
        sys.exit(1)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    out = {"commit": commit, "seed_data": 42, "seeds_perm": [100, 101, 102],
           "seed_eval": 7, "seed_ft": 200, "E000_gate": r000, "E003": r003}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "results", "raw"), exist_ok=True)
    path = os.path.join(root, "results", "raw", "pilot_transport_2026-07-19.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("written:", path)


if __name__ == "__main__":
    main()
