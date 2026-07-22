"""Sigma_E estimator for E-012 (R-3) — the empirical-Fisher functional metric.

Registered in docs/experiment_log.md E-012 (2026-07-22, D-018) as a FROZEN
DECISION PROCEDURE over ranked candidate families; the Phase-0 instrument check
(I-Sigma) selects the cheapest family that reproduces small-perturbation sym-KL
to rel <= 0.25. This module implements the family-1 estimator (diagonal
empirical Fisher). Families 2/3 (block-diagonal Gram; low-rank Jacobian sketch)
are only built if family 1 fails I-Sigma — cheapest-first (project principle 4).

READ-ONLY consumer of cached checkpoints. Runs BACKWARD passes for the Fisher
ONLY (the first real-scale use of gradients in the program; D-018 new-scope
flag) — never for d_f, which stays the certified forward-only path (imported).

The functional metric: locally sym-KL(theta, theta+delta) ~= delta^T Sigma_E delta
(P-1 second-moment metric at real scale). For the softmax LM,
  KL(p_theta || p_{theta+delta}) ~= 1/2 delta^T F delta   (F = per-position Fisher),
and the symmetric KL as coded in e009_divergence.d_f (mean over positions of
0.5*(kl_ab+kl_ba)) is also ~= 1/2 delta^T F delta to leading order. Hence
  Sigma_E := SIGMA_KL_FACTOR * F_hat,  SIGMA_KL_FACTOR = 0.5,
so that delta^T Sigma_E delta predicts the measured sym-KL. This 0.5 is the only
place normalization matters: frac_rec and cos_Sigma_E are RATIOS of Sigma_E
quadratic forms, so any global scale of Sigma_E cancels there — the factor is
load-bearing ONLY for the I-Sigma check (against sym-KL in absolute nats).

F_hat (family 1) = mean over sampled shard tokens of (d log p(y_obs|x)/d theta)^2
per parameter (diagonal; ignores cross-parameter correlation). The exact
per-token empirical Fisher costs one FULL backward per token: measured at
~0.37 s/token on the RTX 2060, so the full 98,304-token shard would be ~10 GPU-h
per build (the E-012 budget note's "GPU-minutes" was ~100x optimistic — recorded
honestly). We therefore average over a DETERMINISTIC strided token subsample
(TOKEN_STRIDE), documented as a necessary realization of family 1: the diagonal
Fisher's QUADRATIC FORMS (delta^T Sigma delta) and the projection RATIOS that use
them aggregate over ~1.6e8 parameters and are stable well below the per-entry
noise, so a few thousand tokens suffice for the scalar readouts I-Sigma / frac_rec
depend on. No frozen gate, tolerance, ladder point, seed, or margin is changed by
this — only the token count of the estimator build. (grad^2 accumulated in
float64 on CPU: stronger than the task's fp32 floor, matching the project's
float64 convention for d_f/Delta_theta; model forward/backward stay fp32.)
"""

import time

import numpy as np
import torch

SIGMA_KL_FACTOR = 0.5          # KL ~= 1/2 delta^T F delta (frozen normalization)
TOKEN_STRIDE = 32              # deterministic subsample: ~64 tokens/seq (frozen)


def param_names(model):
    return [n for n, _ in model.named_parameters()]


def build_empirical_fisher_pass(model, tokens, device, stride=TOKEN_STRIDE,
                                probe_dirs=None, verbose=True):
    """Single per-token backward pass over a deterministic strided token
    subsample. Returns (fisher_diag, fam2_quad, famfull_quad, n_tokens) where:

      * fisher_diag[name] = mean_t (grad log p(y_t|x))^2  -> FAMILY 1
        (diagonal empirical Fisher; float64 CPU dict).
      * fam2_quad[dirname] = mean_t sum_tensor (g_{t,tensor} . dir_tensor)^2
        -> the FAMILY-2 (block-diagonal / per-tensor empirical Gram) quadratic
        form delta^T G_block delta (square each tensor's grad-dot, then sum
        tensors), WITHOUT materializing the (intractable) per-tensor Gram.
      * famfull_quad[dirname] = mean_t (sum_allparams g_t . dir)^2 = the FULL
        empirical Fisher quadratic form delta^T F_full delta -> a diagnostic that
        isolates the empirical-vs-true-Fisher magnitude bias from the diagonal /
        block structural approximation (for random delta it ~= trace, so
        family1 ~= family2 ~= famfull there).
      All accumulated for the UNIT probe directions; scale by scale^2 at eval.
      `probe_dirs`: {dirname: {name: unit-dir tensor}}.

    Deterministic given (weights, tokens, stride) — no RNG (observed y). A fresh
    build reproduces bit-for-bit: the I-repro basis for an empirical family.
    """
    names = param_names(model)
    acc = {n: torch.zeros_like(p, dtype=torch.float64, device="cpu")
           for n, p in model.named_parameters()}
    probe_dirs = probe_dirs or {}
    fam2 = {d: 0.0 for d in probe_dirs}       # block-diagonal (per-tensor)
    famfull = {d: 0.0 for d in probe_dirs}    # full empirical Fisher (diagnostic)
    ntok = 0
    t0 = time.time()
    for i in range(tokens.shape[0]):
        x = torch.from_numpy(tokens[i].astype(np.int64)).unsqueeze(0).to(device)
        out = model(x).logits.float().squeeze(0)
        logp = torch.log_softmax(out, -1)
        tgt = x.squeeze(0)[1:]
        sel = logp[:-1].gather(1, tgt.unsqueeze(1)).squeeze(1)   # log p(y_obs)
        for t in range(0, sel.shape[0], stride):
            model.zero_grad(set_to_none=True)
            sel[t].backward(retain_graph=True)
            # move grads to CPU (float64) ONCE; do acc + probe dots there so no
            # fp64 temporaries pile onto the retained graph on the 6 GB GPU.
            gcpu = {n: p.grad.detach().double().cpu()
                    for n, p in model.named_parameters() if p.grad is not None}
            for n, gc in gcpu.items():
                acc[n] += gc ** 2
            for d, dir_ in probe_dirs.items():
                block_sq = 0.0
                total = 0.0
                for n, gc in gcpu.items():
                    dot = float((gc * dir_[n].double()).sum())
                    block_sq += dot * dot        # per-tensor square -> block Gram
                    total += dot                 # accumulate for full-Fisher
                fam2[d] += block_sq
                famfull[d] += total * total
            ntok += 1
        del out, logp, sel
        if device == "cuda":
            torch.cuda.empty_cache()
        if verbose:
            print(f"    [emp-fisher] seq {i + 1}/{tokens.shape[0]} "
                  f"ntok={ntok} t={time.time() - t0:.0f}s", flush=True)
    model.zero_grad(set_to_none=True)
    fisher = {n: acc[n] / ntok for n in names}
    fam2 = {d: fam2[d] / ntok for d in fam2}
    famfull = {d: famfull[d] / ntok for d in famfull}
    return fisher, fam2, famfull, ntok


def build_diag_true_fisher(model, tokens, device, n_sample=4, seed=20260722,
                           verbose=True):
    """Family-3 object (GGN / TRUE Fisher, J^T H J) realized as its MC-sampled
    PARAMETER-SPACE DIAGONAL — cheaper than the registered low-rank sketch and
    directly compatible with the diagonal-metric projection. Returns
    (fisher_dict, n_seq_samples).

    Per sequence, sample yhat_t ~ p_theta(.|x_{<=t}) at every position and
    backward L = sum_t log p(yhat_t|x); square the grad and accumulate. In
    expectation the cross-position terms vanish (the score has zero mean under
    the model), so E[(sum_t g_t)^2] = sum_t E[g_t^2] = the TRUE Fisher diagonal
    summed over positions — one backward per SEQUENCE per sample (vs the
    empirical family's one backward per TOKEN). Normalized to the per-POSITION
    mean to match sym-KL's per-position scale. Deterministic given `seed`
    (fixed generator) -> the I-repro basis for the GGN family.

    Faithful to family-3's OBJECT (J^T H J); the low-rank Jacobian sketch is the
    fallback realization if this diagonal fails I-Sigma on a structured direction.
    """
    names = param_names(model)
    g = torch.Generator(device=device).manual_seed(seed)
    acc = {n: torch.zeros_like(p, dtype=torch.float64, device="cpu")
           for n, p in model.named_parameters()}
    npos = 0
    t0 = time.time()
    for i in range(tokens.shape[0]):
        x = torch.from_numpy(tokens[i].astype(np.int64)).unsqueeze(0).to(device)
        out = model(x).logits.float().squeeze(0)
        logp = torch.log_softmax(out, -1)
        p = logp.exp()
        L = logp.shape[0]
        for s in range(n_sample):
            yhat = torch.multinomial(p, 1, generator=g).squeeze(1)      # [L]
            loss = logp.gather(1, yhat.unsqueeze(1)).squeeze(1).sum()
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=(s < n_sample - 1))
            for n, pp in model.named_parameters():
                if pp.grad is not None:
                    acc[n] += pp.grad.detach().double().cpu() ** 2
            npos += L
        del out, logp, p
        if device == "cuda":
            torch.cuda.empty_cache()
        if verbose:
            print(f"    [true-fisher] seq {i + 1}/{tokens.shape[0]} "
                  f"npos={npos} t={time.time() - t0:.0f}s", flush=True)
    model.zero_grad(set_to_none=True)
    fisher = {n: acc[n] / npos for n in names}       # per-position mean
    return fisher, npos


def sigma_from_fisher(fisher):
    """Sigma_E = 0.5 * F_hat (float64 CPU diagonal dict)."""
    return {n: SIGMA_KL_FACTOR * fisher[n] for n in fisher}


# ---- diagonal-metric linear algebra (all float64) --------------------------
# All inner products are in the Sigma_E metric: <a,b>_Sigma = sum_p sigma_p a_p b_p.
# Directions/vectors are dicts {name: float tensor} over the SAME key set as sigma.

def sigma_inner(sigma, a, b):
    """<a,b>_Sigma = sum_p sigma_p a_p b_p (float64 scalar)."""
    s = 0.0
    for n in a:
        s += float((sigma[n] * a[n].double() * b[n].double()).sum())
    return s


def sigma_norm2(sigma, a):
    return sigma_inner(sigma, a, a)


def frac_rec_from_basis(sigma, basis, target, target_norm2=None):
    """Recoverable fraction of `target` (= Delta_theta_T) in the Sigma_E metric
    onto span(basis). basis: list of direction dicts (any basis of the subspace;
    frac_rec is subspace-invariant, so orthonormality is unnecessary).

    frac_rec = b^T G^+ b / ||target||^2_Sigma,  G_ij=<s_i,s_j>_Sigma, b_i=<s_i,target>_Sigma.
    Uses pinv (rcond) for rank-deficient/near-singular G. Returns dict with
    frac_rec, residual2, target_norm2, rank(G), gram_cond.
    """
    if target_norm2 is None:
        target_norm2 = sigma_norm2(sigma, target)
    k = len(basis)
    if k == 0:
        return {"frac_rec": 0.0, "residual2": target_norm2,
                "target_norm2": target_norm2, "rank": 0, "gram_cond": None}
    G = np.zeros((k, k))
    bvec = np.zeros(k)
    for i in range(k):
        bvec[i] = sigma_inner(sigma, basis[i], target)
        for j in range(i, k):
            G[i, j] = G[j, i] = sigma_inner(sigma, basis[i], basis[j])
    Gpinv = np.linalg.pinv(G, rcond=1e-10)
    proj = float(bvec @ Gpinv @ bvec)
    rank = int(np.linalg.matrix_rank(G, tol=1e-10 * max(np.diag(G).max(), 1e-300)))
    try:
        cond = float(np.linalg.cond(G))
    except Exception:
        cond = None
    frac = proj / target_norm2 if target_norm2 > 0 else float("nan")
    return {"frac_rec": frac, "residual2": target_norm2 - proj,
            "target_norm2": target_norm2, "rank": rank, "gram_cond": cond}


def cos_sigma(sigma, a, b, a_norm2=None, b_norm2=None):
    """cos_Sigma_E(a,b) = <a,b>_Sigma / (||a||_Sigma ||b||_Sigma)."""
    ab = sigma_inner(sigma, a, b)
    an = a_norm2 if a_norm2 is not None else sigma_norm2(sigma, a)
    bn = b_norm2 if b_norm2 is not None else sigma_norm2(sigma, b)
    denom = (an ** 0.5) * (bn ** 0.5)
    return ab / denom if denom > 0 else float("nan")
