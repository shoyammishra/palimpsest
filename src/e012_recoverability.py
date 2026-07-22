"""E-012 — R-3 access-class recoverability ladder + Sigma_E-weighted directional
lock-in. Registered in docs/experiment_log.md E-012 (2026-07-22, D-018) BEFORE
this code existed (git precedence is the freeze; registration commit 6934d73).

Two phases (Phase 1 runs ONLY if Phase 0's I-Sigma passes):
  Phase 0 — settle + certify the Sigma_E estimator (frozen decision procedure:
    cheapest family that passes I-Sigma) + the other instrument certs.
  Phase 1 — (R3a) recoverable-fraction-vs-access-class ladder frac_rec(I) vs a
    5-seed rank-matched random-subspace floor (gate G-rec, m=0.05); (R3b) the
    Sigma_E-weighted directional lock-in timing t*_func vs F-017's raw t*=128000.

Reuses the certified E-009 measured-number path UNCHANGED BY IMPORT
(d_f / d_theta_rel / load_model / build_eval_tokens) as the functional-distance
ceiling/normalizer and I-df cross-check (D-012). The Sigma_E estimator (sigma_e.py)
is a NEW read-only consumer of cached checkpoints that runs BACKWARD passes for
the Fisher ONLY (first real-scale use of gradients; D-018 new-scope flag) — never
for d_f. No training, no weight updates; touches no recorded JSON.

Canonical parameter access is load_model(...).named_parameters() (fp32
from_pretrained; the E-009/E-011b certified path, autograd-capable) for BOTH the
Fisher model and the Delta_theta vectors, so the Fisher gradients and the
projected vectors share one key convention (the transformers 'lm_head.weight'
naming). block_labels imported from e011a with the e011b lm_head alias.

Run:  HF_HUB_OFFLINE=1 python src/e012_recoverability.py         (real)
      python src/e012_recoverability.py --smoke                  (tiny toy CI)
"""

import copy
import datetime
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from e009_divergence import (  # noqa: E402  (certified, imported unchanged)
    load_model, d_f, d_theta_rel, build_eval_tokens)
from e011a_direction import block_labels  # noqa: E402  (the frozen partition)
import sigma_e  # noqa: E402

ROOT = SRC.parent
RAW = ROOT / "results" / "raw"
DATE = datetime.date.today().isoformat()
SCRATCH = Path(os.environ.get(
    "E012_SCRATCH",
    RAW.parent / "_e012_cache"))          # Fisher caches (NOT committed)

# --- frozen constants (E-012 registration, commit 6934d73) ------------------
T = 143000
REPO_DS1 = "EleutherAI/pythia-160m-data-seed1"   # base metric anchor theta_A
REPO_DS2 = "EleutherAI/pythia-160m-data-seed2"
TOTAL_F015 = 0.2904613848                 # recorded d_f(P1@143000); I-df ref
REL_IDF = 1e-3                            # I-df tolerance
ISIGMA_TOL = 0.25                        # I-Sigma rel-err tolerance (frozen)
ISIGMA_CE_MAX = 0.05                     # I-Sigma probe CE-increase cap (nats)
IREPRO_TOL = 1e-6
IRANDDIR_TOL = 5e-3                       # I-random-dir |cos_Sigma| cap
SEEDS = [20260722, 20260723, 20260724, 20260725, 20260726]
M_GATE = 0.05                            # G-rec margin
RSTAR = "k8"                             # frozen gated ladder point
LADDER_VOID_LIMIT = 2                     # > this many void GATED ladder cells => R3a inconclusive

# F-015 12-point grid (R3b Delta_theta_t; and the update-direction pool)
GRID = [0, 4, 16, 64, 256, 512, 1000, 4000, 16000, 64000, 128000, 143000]
GRID_NZ = GRID[1:]                        # nonzero grid for R3b cosine curve
LAST2_PREV = 142000                       # rank-1 last-2 class: ds1@143000 - ds1@142000

# F-017 raw per-type direction-persistence inner-product share @16000 (cited)
F017_SHARE = {"attn_qkv": 0.616, "mlp_in": 0.103, "embed_in": 0.096,
              "embed_out": 0.089, "mlp_out": 0.078, "attn_dense": 0.017,
              "ln": 0.000, "final_ln": 0.000}


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception as e:  # pragma: no cover
        return f"unknown ({e!r})"


def blabel(key):
    """e011a block_labels with the e011b transformers lm_head alias."""
    if key == "lm_head.weight":
        return "embed_out", None
    return block_labels(key)


# --- canonical parameter access ---------------------------------------------

def param_dict(model, dtype=torch.float32):
    """{name: detached CPU tensor} over the trainable params, model key order."""
    return {n: p.detach().to("cpu", dtype).clone()
            for n, p in model.named_parameters()}


def load_vec(repo, step, device="cpu", dtype=torch.float32):
    m = load_model(repo, step, device)
    d = param_dict(m, dtype)
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return d


def vsub(a, b):
    """a - b as float32 dict over a's keys."""
    return {n: (a[n].float() - b[n].float()) for n in a}


# --- I-Sigma probe-direction machinery --------------------------------------

def scale_delta_to_ce(base_model, unit_dir, tokens, device, sigma1,
                      target_ce=0.02, cap=ISIGMA_CE_MAX, max_iter=4):
    """Choose a scale so the MEASURED CE increase of theta_A + scale*unit_dir is
    <= cap (aim target_ce), then return (scale, sym_kl, dce, ce_pert, ce_base).
    Initial scale from the diagonal family-1 prediction; refine by measured CE.
    Each measurement is a certified d_f call (theta_h passed first -> ce_a)."""
    q1 = sigma_e.sigma_norm2(sigma1, unit_dir)          # unit delta^T Sigma1 delta
    scale = (target_ce / q1) ** 0.5 if q1 > 0 else 1e-4
    last = None
    for _ in range(max_iter):
        m = copy.deepcopy(base_model)
        with torch.no_grad():
            for n, p in m.named_parameters():
                p.add_(scale * unit_dir[n].to(p.device))
        m.eval()
        r = d_f(m, base_model, tokens, device)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        dce = r["ce_a"] - r["ce_b"]
        last = (scale, r["sym_kl"], dce, r["ce_a"], r["ce_b"])
        if 0 < dce <= cap:
            break
        # CE increase ~ quadratic in scale: rescale toward target
        if dce > 0:
            scale = scale * (target_ce / dce) ** 0.5
        else:
            scale = scale * 0.5
    return last


def isigma_eval(name, unit_dir, scale, sym_kl, quad_unit):
    """quad_unit = delta_hat^T Sigma delta_hat for the UNIT direction; scale to
    the probe's scale and compare to measured sym_kl."""
    pred = quad_unit * scale * scale
    rel = abs(pred - sym_kl) / sym_kl if sym_kl > 0 else float("nan")
    return {"direction": name, "scale": scale, "sym_kl": sym_kl,
            "pred": pred, "rel_err": rel, "pass": rel <= ISIGMA_TOL}


# --- Phase 1 helpers (ladder Grams, random floor, cosine curve) -------------

def build_atom_grams(ds1_ckpts, delta_T, sigma, atoms):
    """Stream over keys once, computing for the increment 'atoms':
      A[i,j] = <atom_i, atom_j>_Sigma  (Sigma-Gram, MxM)
      E[i,j] = <atom_i, atom_j>        (Euclidean Gram, for PCA)
      c[i]   = <atom_i, delta_T>_Sigma
      nT2    = ||delta_T||^2_Sigma
    atoms: list of (stepA, stepB) meaning ds1@stepA - ds1@stepB. All float64.
    ds1_ckpts: {step: {name: tensor}}. delta_T, sigma: {name: tensor}."""
    M = len(atoms)
    A = np.zeros((M, M)); E = np.zeros((M, M)); c = np.zeros(M); nT2 = 0.0
    keys = list(delta_T.keys())
    for k in keys:
        sig = sigma[k].double()
        dT = delta_T[k].double()
        nT2 += float((sig * dT * dT).sum())
        av = [(ds1_ckpts[a][k].double() - ds1_ckpts[b][k].double())
              for (a, b) in atoms]
        for i in range(M):
            ci = float((sig * av[i] * dT).sum())
            c[i] += ci
            for j in range(i, M):
                aij = float((sig * av[i] * av[j]).sum())
                A[i, j] += aij; A[j, i] = A[i, j]
                eij = float((av[i] * av[j]).sum())
                E[i, j] += eij; E[j, i] = E[i, j]
    return A, E, c, nT2


def frac_rec_span(A, c, nT2, idx=None, W=None):
    """frac_rec for a subspace of span(atoms). Either idx (select atom columns)
    or W (M x r combination matrix, columns = basis vectors in atom coords).
    Uses the precomputed atom Sigma-Gram A and cross-vector c."""
    if W is None:
        idx = list(idx)
        G = A[np.ix_(idx, idx)]
        b = c[idx]
    else:
        G = W.T @ A @ W
        b = W.T @ c
    if G.shape[0] == 0:
        return {"frac_rec": 0.0, "residual2": nT2, "rank": 0}
    Gpinv = np.linalg.pinv(G, rcond=1e-10)
    proj = float(b @ Gpinv @ b)
    rank = int(np.linalg.matrix_rank(G, tol=1e-10 * max(np.diag(G).max(), 1e-300)))
    return {"frac_rec": proj / nT2 if nT2 > 0 else float("nan"),
            "residual2": nT2 - proj, "rank": rank}


def random_frac_rec(keys, shapes, r, seed, sigma, delta_T, nT2):
    """frac_rec of delta_T onto a random r-dim Gaussian subspace (Sigma metric).
    Directions regenerated per-key deterministically (seed, dir index); no full
    n-vectors held. frac_rec is subspace-invariant, so raw (un-orthonormalized)
    Gaussian bases give the exact random-r-dim-subspace projection."""
    G = np.zeros((r, r)); b = np.zeros(r)
    gens = [torch.Generator().manual_seed(seed * 131 + di) for di in range(r)]
    for k in keys:
        sig = sigma[k].double()
        dT = delta_T[k].double()
        cols = [torch.randn(shapes[k], generator=gens[di]).double()
                for di in range(r)]
        for i in range(r):
            b[i] += float((sig * cols[i] * dT).sum())
            for j in range(i, r):
                G[i, j] += float((sig * cols[i] * cols[j]).sum())
                G[j, i] = G[i, j]
    Gpinv = np.linalg.pinv(G, rcond=1e-12)
    proj = float(b @ Gpinv @ b)
    return proj / nT2 if nT2 > 0 else float("nan")


def cos_curve_point(ds1_t, ds2_t, delta_T, sigma, nT2):
    """cos_Sigma(Delta_t, Delta_T) with Delta_t = ds1_t - ds2_t (streaming)."""
    dot = ntt = 0.0
    for k in delta_T.keys():
        sig = sigma[k].double()
        dt = ds1_t[k].double() - ds2_t[k].double()
        dT = delta_T[k].double()
        dot += float((sig * dt * dT).sum())
        ntt += float((sig * dt * dt).sum())
    denom = (ntt ** 0.5) * (nT2 ** 0.5)
    return dot / denom if denom > 0 else float("nan"), ntt ** 0.5


# ---------------------------------------------------------------------------

def run(base_model, base_params, ds2_T, ds1_142000, tokens, device, det,
        smoke=False, out_path=None):
    """Full E-012 Phase 0 (+ Phase 1 iff Phase 0 passes). Writes JSON of record.
    base_params == ds1@T param dict; keys are the canonical model keys."""
    keys = list(base_params.keys())
    shapes = {k: tuple(base_params[k].shape) for k in keys}
    n_params = int(sum(int(np.prod(shapes[k])) for k in keys))
    delta_T = vsub(base_params, ds2_T)             # Delta_theta_T (fp32 dict)

    out = {
        "experiment": "E-012 (R-3 recoverability ladder + Sigma_E lock-in; D-018)",
        "date": DATE, "commit": git_head(), "torch": torch.__version__,
        "device": device, "deterministic_algorithms": det,
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "smoke": smoke, "n_params": n_params, "n_param_tensors": len(keys),
        "thresholds": {
            "isigma_tol": ISIGMA_TOL, "isigma_ce_max": ISIGMA_CE_MAX,
            "idf_tol": REL_IDF, "irepro_tol": IREPRO_TOL,
            "iranddir_tol": IRANDDIR_TOL, "m_gate": M_GATE, "rstar": RSTAR,
            "seeds": SEEDS, "total_f015": TOTAL_F015,
            "emp_stride": EMP_STRIDE, "true_nsample": TRUE_NSAMPLE},
        "gates": {}, "phase0": {}, "phase1": {},
    }

    def save():
        if out_path:
            out_path.write_text(json.dumps(out, indent=2, default=str),
                                encoding="utf-8")

    # ---- probe directions (unit) for I-Sigma + family-2 accumulation --------
    g = torch.Generator().manual_seed(20260722)
    rand_dir = {k: torch.randn(base_params[k].shape, generator=g,
                               dtype=torch.float32) for k in keys}
    probe_dirs = {
        "random": rand_dir,
        "update_last2": vsub(base_params, ds1_142000),   # ds1@T - ds1@142000
        "delta_T": delta_T,
    }

    # ---- I-df cross-check (certified d_f; skipped in smoke) ------------------
    if not smoke:
        ds2_model = load_model(REPO_DS2, T, device)
        r = d_f(base_model, ds2_model, tokens, device)
        del ds2_model
        if device == "cuda":
            torch.cuda.empty_cache()
        idf_rel = abs(r["sym_kl"] - TOTAL_F015) / TOTAL_F015
        out["gates"]["I_df"] = {"pass": bool(idf_rel <= REL_IDF),
                                "d_f": r["sym_kl"], "ref": TOTAL_F015,
                                "rel_err": idf_rel, "threshold": REL_IDF,
                                "ce_ds1": r["ce_a"], "ce_ds2": r["ce_b"]}
        print(f"[I-df] d_f={r['sym_kl']:.10f} rel={idf_rel:.2e} "
              f"{'PASS' if idf_rel <= REL_IDF else 'FAIL'}", flush=True)
        save()

    # ---- Phase 0: settle + certify the Sigma_E estimator --------------------
    sigma_emp, meta_emp = build_sigma("emp", base_model, tokens, device,
                                      probe_dirs, tag=("smoke_emp" if smoke
                                      else "emp"))
    fam2_unit = meta_emp["fam2_quad_unit"]

    probe_scales = {}
    for name, ud in probe_dirs.items():
        if smoke:
            q = sigma_e.sigma_norm2(sigma_emp, ud)
            scale = (0.01 / q) ** 0.5 if q > 0 else 1e-3
            m = copy.deepcopy(base_model)
            with torch.no_grad():
                for n, p in m.named_parameters():
                    p.add_(scale * ud[n].to(p.device))
            rr = d_f(m, base_model, tokens, device)
            del m
            probe_scales[name] = {"scale": scale, "sym_kl": rr["sym_kl"],
                                  "dce": rr["ce_a"] - rr["ce_b"]}
        else:
            scale, sym_kl, dce, cep, ceb = scale_delta_to_ce(
                base_model, ud, tokens, device, sigma_emp)
            probe_scales[name] = {"scale": scale, "sym_kl": sym_kl, "dce": dce}
            print(f"[I-Sigma probe] {name}: scale={scale:.3e} dCE={dce:.4f} "
                  f"sym_kl={sym_kl:.5f}", flush=True)
    out["phase0"]["probe_scales"] = probe_scales
    save()

    def eval_family(fam_name, quad_fn):
        res = []
        for name in probe_dirs:
            sc = probe_scales[name]
            res.append(isigma_eval(name, probe_dirs[name], sc["scale"],
                                   sc["sym_kl"], quad_fn(name)))
        npass = sum(1 for r in res if r["pass"])
        return {"family": fam_name, "per_direction": res, "n_pass": npass,
                "n_dir": len(res), "pass": npass == len(res)}

    fam1 = eval_family("diag_empirical",
                       lambda name: sigma_e.sigma_norm2(sigma_emp,
                                                        probe_dirs[name]))
    fam2 = eval_family("block_empirical_gram",
                       lambda name: sigma_e.SIGMA_KL_FACTOR * fam2_unit[name])
    # diagnostic (NOT a selectable family): the FULL empirical Fisher quadratic
    # form — isolates the empirical-vs-true-Fisher magnitude bias from structure.
    famfull_unit = meta_emp.get("famfull_quad_unit", {})
    famfull = eval_family("full_empirical_fisher_diagnostic",
                          lambda name: sigma_e.SIGMA_KL_FACTOR
                          * famfull_unit.get(name, 0.0))
    out["phase0"]["family1_isigma"] = fam1
    out["phase0"]["family2_isigma"] = fam2
    out["phase0"]["full_empirical_diagnostic"] = famfull
    print(f"[I-Sigma] family1 {'PASS' if fam1['pass'] else 'FAIL'} "
          f"({fam1['n_pass']}/{fam1['n_dir']}); family2 "
          f"{'PASS' if fam2['pass'] else 'FAIL'} ({fam2['n_pass']}/{fam2['n_dir']})",
          flush=True)
    for tag, fm in (("f1", fam1), ("f2", fam2)):
        for r in fm["per_direction"]:
            print(f"    {tag} {r['direction']}: pred={r['pred']:.5f} "
                  f"sym_kl={r['sym_kl']:.5f} rel={r['rel_err']:.3f} "
                  f"{'ok' if r['pass'] else 'X'}", flush=True)
    save()

    selected = sigma = sel_meta = None
    if fam1["pass"]:
        selected, sigma, sel_meta = "diag_empirical", sigma_emp, meta_emp
    else:
        sigma_true, meta_true = build_sigma("true", base_model, tokens, device,
                                            probe_dirs, tag=("smoke_true" if smoke
                                            else "true"))
        fam3 = eval_family("diag_true_fisher_ggn",
                           lambda name: sigma_e.sigma_norm2(sigma_true,
                                                            probe_dirs[name]))
        out["phase0"]["family3_isigma"] = fam3
        print(f"[I-Sigma] family3 (true-Fisher GGN diag) "
              f"{'PASS' if fam3['pass'] else 'FAIL'} "
              f"({fam3['n_pass']}/{fam3['n_dir']})", flush=True)
        for r in fam3["per_direction"]:
            print(f"    f3 {r['direction']}: pred={r['pred']:.5f} "
                  f"sym_kl={r['sym_kl']:.5f} rel={r['rel_err']:.3f} "
                  f"{'ok' if r['pass'] else 'X'}", flush=True)
        save()
        if fam2["pass"] and not fam3["pass"]:
            out["phase0"]["family2_note"] = (
                "family-2 (block Gram) passed I-Sigma while diagonal families 1 "
                "and 3 did not; its projection needs a block/KFAC metric NOT "
                "implemented here -> reported instrument-limited (would need a "
                "superseding registration for the block projection).")
        if fam3["pass"]:
            selected, sigma, sel_meta = "diag_true_fisher_ggn", sigma_true, meta_true

    if selected is None and smoke:
        # smoke: force-select the true-Fisher family so the Phase-1 linear
        # algebra is still exercised (numeric I-Sigma is meaningless on a tiny
        # random model). Never triggered in the real run.
        selected, sigma, sel_meta = "diag_true_fisher_ggn", sigma_true, meta_true
        out["phase0"]["smoke_forced_select"] = True

    out["gates"]["I_Sigma"] = {
        "selected_family": selected, "pass": bool(selected is not None),
        "note": ("PASS => cheapest diagonal family reproducing small-perturb "
                 "sym-KL to rel<=0.25 on 3/3 probe directions; FAIL(all) => HALT "
                 "at Phase 0 (D-018 reversal 1), NO tolerance loosening.")}
    save()

    if selected is None:
        out["gates"]["I_Sigma"]["halt"] = True
        out["verdict"] = ("E-012 HALTS at Phase 0: no estimator family passed "
                          "I-Sigma. R-3 unrun under this registration; Phase 1 "
                          "NOT executed (frozen rule). No tolerance loosened.")
        print("\nHALT at Phase 0 — no estimator family passed I-Sigma.",
              flush=True)
        save()
        return out
    out["phase0"]["selected_family_meta"] = {
        k: v for k, v in sel_meta.items()
        if k not in ("fam2_quad_unit", "famfull_quad_unit")}
    print(f"[I-Sigma] SELECTED family = {selected}", flush=True)

    # ---- I-random-dir, I-boundary, I-cos-identity ---------------------------
    nT2 = sigma_e.sigma_norm2(sigma, delta_T)
    g2 = torch.Generator().manual_seed(20260777)
    rdir = {k: torch.randn(base_params[k].shape, generator=g2,
                           dtype=torch.float32) for k in keys}
    crd = sigma_e.cos_sigma(sigma, delta_T, rdir, a_norm2=nT2)
    out["gates"]["I_random_dir"] = {
        "pass": bool(abs(crd) <= IRANDDIR_TOL), "cos_sigma": crd,
        "threshold": IRANDDIR_TOL,
        "expected_scale": math.sqrt(2 / (math.pi * n_params))}

    fr_empty = sigma_e.frac_rec_from_basis(sigma, [], delta_T, nT2)["frac_rec"]
    fr_full = sigma_e.frac_rec_from_basis(sigma, [delta_T], delta_T,
                                          nT2)["frac_rec"]
    out["gates"]["I_boundary"] = {
        "pass": bool(fr_empty == 0.0 and abs(fr_full - 1.0) <= 1e-9),
        "frac_rec_empty": fr_empty, "frac_rec_full": fr_full,
        "note": "frac_rec(empty)=0, frac_rec(S>=Delta_T)=1 — designed certs"}

    self_cos = sigma_e.cos_sigma(sigma, delta_T, delta_T, nT2, nT2)
    neg = {k: -delta_T[k] for k in keys}
    neg_cos = sigma_e.cos_sigma(sigma, delta_T, neg, nT2, nT2)
    out["gates"]["I_cos_identity"] = {
        "pass": bool(abs(self_cos - 1.0) <= 1e-9 and abs(neg_cos + 1.0) <= 1e-9),
        "self_cos": self_cos, "neg_cos": neg_cos}
    print(f"[I-random-dir] cos={crd:.3e} | [I-boundary] empty={fr_empty} "
          f"full={fr_full} | self={self_cos:.10f} neg={neg_cos:.10f}", flush=True)
    save()

    # ---- I-repro: fresh Sigma_E build reproduces frac_rec(rank-1) -----------
    last2_dir = vsub(base_params, ds1_142000)
    fr1_a = sigma_e.frac_rec_from_basis(sigma, [last2_dir], delta_T,
                                        nT2)["frac_rec"]
    fam = "emp" if selected == "diag_empirical" else "true"
    sigma_fresh, _ = build_sigma(fam, base_model, tokens, device, probe_dirs,
                                 tag=("smoke_" + fam + "_repro" if smoke
                                      else fam + "_repro"))
    nT2b = sigma_e.sigma_norm2(sigma_fresh, delta_T)
    fr1_b = sigma_e.frac_rec_from_basis(sigma_fresh, [last2_dir], delta_T,
                                        nT2b)["frac_rec"]
    repro_rel = abs(fr1_a - fr1_b) / max(abs(fr1_a), 1e-300)
    out["gates"]["I_repro"] = {"pass": bool(repro_rel <= IREPRO_TOL),
                               "frac_rec_a": fr1_a, "frac_rec_b": fr1_b,
                               "rel_err": repro_rel, "threshold": IREPRO_TOL}
    print(f"[I-repro] frac_rec(rank1) {fr1_a:.8f} vs {fr1_b:.8f} rel={repro_rel:.2e} "
          f"{'PASS' if repro_rel <= IREPRO_TOL else 'FAIL'}", flush=True)
    save()

    cert_keys = ["I_Sigma", "I_random_dir", "I_boundary", "I_cos_identity",
                 "I_repro"] + ([] if smoke else ["I_df"])
    phase0_pass = all(out["gates"][k]["pass"] for k in cert_keys)
    out["phase0"]["all_pass"] = bool(phase0_pass)
    if not phase0_pass and not smoke:
        out["verdict"] = ("E-012 Phase 0 instrument cert FAILED (see gates); "
                          "Phase 1 NOT run — instrument problem, not a subject "
                          "verdict.")
        print("\nPhase 0 cert FAILED — Phase 1 NOT run.", flush=True)
        save()
        return out

    # ==== PHASE 1 ============================================================
    print("\n=== PHASE 1 (authorized) ===", flush=True)
    phase1_ladder(out, sigma, base_params, delta_T, nT2, ds1_142000, keys,
                  shapes, n_params, tokens, device, smoke, save)
    phase1_lockin(out, sigma, base_params, delta_T, nT2, keys, tokens, device,
                  smoke, save)
    out["verdict"] = phase1_verdict(out)
    save()
    return out


# --- estimator build (with scratch cache) -----------------------------------
EMP_STRIDE = 128          # family-1/2 build (diagnostic; ~16 tok/seq over shard).
                          # Empirical families carry the emp-vs-true-Fisher bias
                          # (probe: rel 0.56-1.17) and are expected to FAIL I-Sigma;
                          # a modest token budget suffices to record that verdict.
TRUE_NSAMPLE = 8          # family-3 (true-Fisher GGN) MC samples/seq: enough that
                          # a residual random-direction miss is the diagonal
                          # approximation, not MC noise (instrument-first).
TRUE_SEED = 20260722      # fixed -> deterministic true-Fisher build (I-repro)


def _sigma_cache(tag):
    return SCRATCH / f"sigma_{tag}.pt"


def build_sigma(family, base_model, tokens, device, probe_dirs, tag,
                use_cache=True):
    """Build Sigma_E for a family; cache the raw Fisher to SCRATCH so a crash /
    the I-repro rebuild / Phase-1 resume need not recompute. fam2_quad (family-2
    block-Gram probe predictions) is only produced by the empirical pass."""
    cache = _sigma_cache(tag)
    if use_cache and cache.exists():
        blob = torch.load(cache, weights_only=False)
        return blob["sigma"], blob["meta"]
    if family == "emp":
        fisher, fam2, famfull, ntok = sigma_e.build_empirical_fisher_pass(
            base_model, tokens, device, stride=EMP_STRIDE, probe_dirs=probe_dirs)
        sigma = sigma_e.sigma_from_fisher(fisher)
        meta = {"family": "diag_empirical", "n_tokens": ntok,
                "stride": EMP_STRIDE, "fam2_quad_unit": fam2,
                "famfull_quad_unit": famfull,
                "sigma_kl_factor": sigma_e.SIGMA_KL_FACTOR}
    elif family == "true":
        fisher, npos = sigma_e.build_diag_true_fisher(
            base_model, tokens, device, n_sample=TRUE_NSAMPLE, seed=TRUE_SEED)
        sigma = sigma_e.sigma_from_fisher(fisher)
        meta = {"family": "diag_true_fisher_ggn", "n_pos": npos,
                "n_sample": TRUE_NSAMPLE, "seed": TRUE_SEED,
                "sigma_kl_factor": sigma_e.SIGMA_KL_FACTOR,
                "note": ("family-3 object J^T H J (GGN/true Fisher) realized as "
                         "its MC-sampled parameter-space diagonal; low-rank "
                         "Jacobian sketch is the fallback if this fails I-Sigma")}
    else:
        raise ValueError(family)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if use_cache:
        torch.save({"sigma": sigma, "meta": meta}, cache)
    return sigma, meta


# --- Phase 1 ----------------------------------------------------------------

def phase1_ladder(out, sigma, base_params, delta_T, nT2, ds1_142000, keys,
                  shapes, n_params, tokens, device, smoke, save):
    """(R3a) frac_rec-vs-access-class ladder + 5-seed random-subspace floor +
    gate G-rec at r*=k8. Needs ds1 checkpoints at the grid + 142000."""
    load = (lambda step: base_params if step == T else
            load_vec(REPO_DS1, step, "cpu", torch.float16))
    ds1_ckpts = {}
    for step in sorted(set(GRID) | {LAST2_PREV}):
        ds1_ckpts[step] = base_params if step == T else load(step)
        if not smoke:
            print(f"    [ladder] loaded ds1@{step}", flush=True)

    # atoms: 11 grid increments (idx 0..10, ending at GRID[1..11]) + last2 (idx 11)
    atoms = [(GRID[j], GRID[j - 1]) for j in range(1, len(GRID))]
    LAST2_IDX = len(atoms)
    atoms = atoms + [(T, LAST2_PREV)]
    A, E, c, nT2_atoms = build_atom_grams(ds1_ckpts, delta_T, sigma, atoms)
    # cross-check the Sigma-norm against the diagonal computation
    out["phase1"]["nT2_crosscheck"] = {"from_sigma": nT2, "from_atoms": nT2_atoms,
                                       "rel": abs(nT2 - nT2_atoms) / nT2}

    # grid-increment indices ending at each grid step
    end_step_to_idx = {GRID[j]: j - 1 for j in range(1, len(GRID))}
    latest = [end_step_to_idx[s] for s in [143000, 128000, 64000, 16000,
                                           4000, 1000, 512, 256]]  # k=8 order
    ladder = {}
    ladder["empty"] = {"frac_rec": 0.0, "rank": 0, "dim": 0, "void": False}
    ladder["rank1_last2"] = dict(frac_rec_span(A, c, nT2_atoms, idx=[LAST2_IDX]),
                                 dim=1, void=False)
    for k in (2, 4, 8):
        ladder[f"k{k}"] = dict(frac_rec_span(A, c, nT2_atoms, idx=latest[:k]),
                               dim=k, void=False)
    # r=8 PCA of the 11 grid increments (descriptive); r=32,128 infeasible
    grid_idx = list(range(len(GRID) - 1))
    Eg = E[np.ix_(grid_idx, grid_idx)]
    evals, evecs = np.linalg.eigh(Eg)
    order = np.argsort(evals)[::-1]
    for r in (8, 32, 128):
        if r <= len(grid_idx):
            Wsel = evecs[:, order[:r]]
            W = np.zeros((len(atoms), r))
            W[np.ix_(grid_idx, range(r))] = Wsel
            ladder[f"pca_r{r}"] = dict(frac_rec_span(A, c, nT2_atoms, W=W),
                                       dim=r, void=False, descriptive=True)
        else:
            ladder[f"pca_r{r}"] = {
                "void": True, "frac_rec": None, "dim": r, "descriptive": True,
                "void_reason": (f"rank-{r} trajectory summary needs >= {r} "
                                f"increment vectors; only {len(grid_idx)} grid "
                                "increments cached (checkpoint-density limit)")}
    out["phase1"]["ladder"] = ladder

    # ---- random-subspace floor (5 seeds), ranks {1,2,4,8}; gate at r*=8 -----
    floor = {}
    for r in (1, 2, 4, 8):
        vals = {}
        for seed in SEEDS:
            vals[str(seed)] = random_frac_rec(keys, shapes, r, seed, sigma,
                                              delta_T, nT2_atoms)
            if not smoke:
                print(f"    [floor] r={r} seed={seed} "
                      f"frac_rec={vals[str(seed)]:.5f}", flush=True)
        floor[f"r{r}"] = {"per_seed": vals, "F_random_max": max(vals.values())}
    out["phase1"]["random_floor"] = floor

    # gate G-rec at r*=k8
    frac_rstar = ladder["k8"]["frac_rec"]
    f_rand_rstar = floor["r8"]["F_random_max"]
    grec_pass = frac_rstar > f_rand_rstar + M_GATE
    void_gated = sum(1 for k in ("rank1_last2", "k2", "k4", "k8")
                     if ladder[k].get("void"))
    out["gates"]["G_rec"] = {
        "pass": bool(grec_pass), "r_star": RSTAR,
        "frac_rec_rstar": frac_rstar, "F_random_rstar": f_rand_rstar,
        "margin": M_GATE, "threshold": f_rand_rstar + M_GATE,
        "gated_ladder_void_count": void_gated,
        "r3a_inconclusive": bool(void_gated > LADDER_VOID_LIMIT),
        "note": ("PASS => real trajectory span recovers order-function beyond a "
                 "random equal-rank subspace; FAIL => indistinguishable from "
                 "random at matched rank (first-class strong negative).")}
    print(f"[G-rec] frac_rec(k8)={frac_rstar:.5f} vs F_random(8)+m="
          f"{f_rand_rstar + M_GATE:.5f} "
          f"{'PASS' if grec_pass else 'FAIL'}", flush=True)

    # ---- per-block Sigma-weighted share of Delta_T (descriptive, L) ---------
    by_type = {}
    for k in keys:
        typ, _ = blabel(k)
        s = float((sigma[k].double() * delta_T[k].double() ** 2).sum())
        by_type[typ] = by_type.get(typ, 0.0) + s
    denom = sum(by_type.values())
    out["phase1"]["sigma_weighted_block_share"] = {
        t: {"share": by_type[t] / denom,
            "f017_raw_share": F017_SHARE.get(t)} for t in by_type}
    save()
    for step in list(ds1_ckpts):
        if step != T:
            ds1_ckpts[step] = None
    return ds1_ckpts


def phase1_lockin(out, sigma, base_params, delta_T, nT2, keys, tokens, device,
                  smoke, save):
    """(R3b) Sigma_E-weighted directional lock-in cos_Sigma(Delta_t, Delta_T)
    across the grid; t*_func = first t with cos_Sigma >= 0.9."""
    curve = []
    for t in GRID_NZ:
        ds1_t = base_params if t == T else load_vec(REPO_DS1, t, "cpu",
                                                    torch.float16)
        ds2_t = load_vec(REPO_DS2, t, "cpu", torch.float16)
        cos, norm = cos_curve_point(ds1_t, ds2_t, delta_T, sigma, nT2)
        curve.append({"t": t, "cos_sigma": cos, "norm_sigma_delta_t": norm})
        if not smoke:
            print(f"    [R3b] t={t} cos_Sigma={cos:.5f}", flush=True)
        del ds1_t, ds2_t
    tstar = next((r["t"] for r in curve if r["cos_sigma"] >= 0.9), None)
    out["phase1"]["lockin"] = {
        "curve": curve, "t_star_func": tstar,
        "t_star_raw_f017": 128000, "cos_raw_16000_f017": 0.294,
        "functional_frac_16000_f015": 0.932,
        "note": ("H2 predicts t*_func < 128000 (functional locks earlier than "
                 "raw). Late-t cos->1 is partly trivial trajectory-continuity "
                 "(cos(Delta_t,Delta_T)->1 as t->T is the self-identity limit).")}
    save()


def phase1_verdict(out):
    grec = out["gates"]["G_rec"]
    lk = out["phase1"]["lockin"]
    return (f"E-012 Phase 0 PASS (selected {out['gates']['I_Sigma']['selected_family']}); "
            f"Phase 1: G-rec {'PASS' if grec['pass'] else 'FAIL'} "
            f"(frac_rec(k8)={grec['frac_rec_rstar']:.4f} vs floor+m="
            f"{grec['threshold']:.4f}); t*_func={lk['t_star_func']} vs raw t*=128000.")


# --- smoke test (tiny toy GPTNeoX; validates all linear algebra + gates) ----

def smoke():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64,
                        max_position_embeddings=64)
    base_model = GPTNeoXForCausalLM(cfg).eval()
    device = "cpu"
    bp = param_dict(base_model)
    keys = list(bp.keys())
    # fake checkpoints: base + deterministic small perturbations per step
    def pert(seed, sc):
        g = torch.Generator().manual_seed(seed)
        return {k: bp[k] + sc * torch.randn(bp[k].shape, generator=g)
                for k in keys}
    ds2_T = pert(101, 0.05)
    ds1_142000 = pert(102, 0.001)
    # smoke grid checkpoints (ds1_t drift toward base at t=0, ds2_t independent)
    smoke_ds1 = {t: (bp if t == T else pert(200 + i, 0.02 * (1 - i / 12)))
                 for i, t in enumerate(GRID_NZ)}
    smoke_ds1[T] = bp
    smoke_ds1[0] = pert(210, 0.03)               # init checkpoint (grid step 0)
    smoke_ds1[LAST2_PREV] = ds1_142000           # last-2 anchor (consistency)
    smoke_ds2 = {t: pert(300 + i, 0.05) for i, t in enumerate(GRID_NZ)}
    tokens = np.random.RandomState(0).randint(0, 64, size=(3, 40)).astype(np.int32)
    out_path = SCRATCH / "e012_smoke.json"
    SCRATCH.mkdir(parents=True, exist_ok=True)

    # monkeypatch ds1 grid loads for the ladder + lockin (smoke has no HF)
    global load_vec
    real_load_vec = load_vec

    def fake_load_vec(repo, step, dev="cpu", dtype=torch.float32):
        if repo == REPO_DS1:
            return {k: smoke_ds1[step][k].to(dtype) for k in keys}
        return {k: smoke_ds2[step][k].to(dtype) for k in keys}
    load_vec = fake_load_vec
    try:
        res = run(base_model, bp, ds2_T, ds1_142000, tokens, device, False,
                  smoke=True, out_path=out_path)
    finally:
        load_vec = real_load_vec

    # assertions on the designed boundaries (must hold exactly)
    gb = res["gates"]
    assert gb["I_boundary"]["frac_rec_empty"] == 0.0, gb["I_boundary"]
    assert abs(gb["I_boundary"]["frac_rec_full"] - 1.0) <= 1e-9, gb["I_boundary"]
    assert abs(gb["I_cos_identity"]["self_cos"] - 1.0) <= 1e-9
    assert abs(gb["I_cos_identity"]["neg_cos"] + 1.0) <= 1e-9
    assert gb["I_repro"]["rel_err"] <= 1e-6, gb["I_repro"]
    # ladder monotonicity (nested classes) + interior (not boundary)
    L = res["phase1"]["ladder"]
    assert L["empty"]["frac_rec"] == 0.0
    assert L["k8"]["frac_rec"] >= L["k4"]["frac_rec"] - 1e-9
    assert 0.0 < L["k8"]["frac_rec"] < 1.0 + 1e-9
    print("\nSMOKE OK — boundary certs exact, ladder monotone, repro deterministic.")
    print("  frac_rec:", {k: (round(v["frac_rec"], 4) if v.get("frac_rec")
                               is not None else None) for k, v in L.items()})
    print("  G-rec pass:", gb["G_rec"]["pass"],
          "  t*_func:", res["phase1"]["lockin"]["t_star_func"])
    return res


def resolve_out_path():
    today = RAW / f"e012_recoverability_{DATE}.json"
    if today.exists():
        return today
    existing = sorted(RAW.glob("e012_recoverability_*.json"))
    return existing[-1] if existing else today


def main():
    if "--smoke" in sys.argv[1:]:
        smoke()
        return
    RAW.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch.use_deterministic_algorithms(True)
        det = True
    except Exception:
        det = False
    tokens = build_eval_tokens()
    out_path = resolve_out_path()
    print(f"E-012 -> {out_path} (device={device}, det={det})", flush=True)

    base_model = load_model(REPO_DS1, T, device)      # theta_A (autograd)
    base_params = param_dict(base_model)               # ds1@T fp32 dict
    ds2_T = load_vec(REPO_DS2, T, "cpu", torch.float32)
    ds1_142000 = load_vec(REPO_DS1, LAST2_PREV, "cpu", torch.float32)

    res = run(base_model, base_params, ds2_T, ds1_142000, tokens, device, det,
              smoke=False, out_path=out_path)
    print("\nVERDICT:", res.get("verdict"), flush=True)
    print("done ->", out_path, flush=True)


if __name__ == "__main__":
    main()
