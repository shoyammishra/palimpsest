"""E-011b — R-2 spatial localization: per-block functional weight of the
training-order trace via alpha-interpolation grafts. Registered in
docs/experiment_log.md E-011b (2026-07-22, D-017) BEFORE this code existed
(git precedence is the freeze; registration commit 2739af7).

FORWARD-EVAL ON CACHED ENDPOINTS ONLY. Reuses the certified E-009
measured-number path UNCHANGED BY IMPORT (d_f / d_theta_rel / load_model /
build_eval_tokens) so every d_f here sits next to F-015 (D-012). No gradients,
no training, no measured-number code path or recorded JSON touched; writes its
own raw file (incremental/resumable, E-009 pattern).

Object: theta_h(b, alpha) = ds1@T with block b's params moved alpha of the way
to ds2@T. Copy semantics at alpha in {0,1} (verbatim tensors — registration
provenance item (i)). Readout phi_move(b,alpha) = d_f(theta_h, ds1)/total.

Memory scheme (post-review): originals ds1/ds2 live on CPU; per cell only the
anchor + the hybrid are resident on GPU (the E-009 two-model peak). Hybrids are
materialized on CPU, d_theta arithmetic runs on CPU, then the hybrid moves to
GPU for the certified d_f call only.

Frozen gates (E-011b): (I-1) alpha=0 identity d_f=0 exactly; (I-2) all-blocks
alpha=1 identity (partition-completeness cert); (I-3) d_f(ds1,ds2) reproduces
F-015's 0.2904613848 to rel<=1e-3 (recomputed value becomes the normalizer);
(I-4) repro of attn_qkv@0.5 to rel<=1e-6; (I-5) arithmetic d_theta prediction
per cell (rel<=1e-3); (V) CE void band CE_h > 3.001560232923117; (G) phi_move
(b*, alpha_G) > F_random(alpha_G) + 0.05 with the frozen alpha_G fallback.
Abort-not-guess: >2 of 8 Tier-1 alpha=1 primary cells void => b*
undeterminable => (G) void, downstream spend (RF/T2/SYM) skipped.
Descriptive (NOT gated): (L) profiles + magnitude null + F-017 bridge,
(A) alpha-scaling exponents (quadratic p=2 reference), (D) symmetry spot-check.

Run:  HF_HUB_OFFLINE=1 python src/e011b_graft.py
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

import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from e009_divergence import (  # noqa: E402  (certified, imported unchanged)
    load_model, d_f, d_theta_rel, build_eval_tokens)
from e011a_direction import block_labels  # noqa: E402  (the frozen partition)

ROOT = SRC.parent
RAW = ROOT / "results" / "raw"
DATE = datetime.date.today().isoformat()

# --- frozen constants (E-011b registration, commit 2739af7) ------------------
T = 143000
REPO_A = "EleutherAI/pythia-160m-data-seed1"   # base ds1
REPO_B = "EleutherAI/pythia-160m-data-seed2"   # donor ds2
ALPHAS_T1 = [0.25, 0.5, 1.0]
ALPHAS_T2 = [0.5, 1.0]
TOTAL_F015 = 0.2904613848          # recorded d_f(P1@143000); I-3 reference
REL_I3 = 1e-3
REL_I4 = 1e-6
REL_I5 = 1e-3
CE_PARENT_MAX = 2.501560232923117  # recorded CE ds1@T (>= ds2@T's 2.4998...)
DELTA_CE = 0.5
CE_VOID = CE_PARENT_MAX + DELTA_CE           # 3.001560232923117
CE_DEGRADED = CE_PARENT_MAX + 0.25           # valid-but-degraded flag
SEEDS = [20260722, 20260723, 20260724, 20260725, 20260726]
M_GATE = 0.05
TIE_FRAC = 0.10
COUNT_TOL = 0.01                   # random-mask actual count within +-1%
VOID_LIMIT_TOTAL = 10              # > 10 graft cells void => R-2 inconclusive
VOID_LIMIT_T1A1 = 2                # > 2 of 8 Tier-1 alpha=1 => (G) void, abort
DTHETA_BAND = 1.2                  # F-014 independent-init-scale tripwire
TYPES = ["embed_in", "embed_out", "attn_qkv", "attn_dense",
         "mlp_in", "mlp_out", "ln", "final_ln"]
LAYERED_TYPES = ["attn_qkv", "attn_dense", "mlp_in", "mlp_out", "ln"]
N_LAYERS = 12
REPRO_BLOCK, REPRO_ALPHA = "attn_qkv", 0.5

# F-017 per-type raw direction-persistence inner-product shares at t=16000
# (results/raw/e011a_direction_2026-07-22.json; cited for the bridge only)
F017_SHARE = {"attn_qkv": 0.616, "mlp_in": 0.103, "embed_in": 0.096,
              "embed_out": 0.089, "mlp_out": 0.078, "attn_dense": 0.017,
              "ln": 0.000, "final_ln": 0.000}


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception as e:  # pragma: no cover
        return f"unknown ({e!r})"


def resolve_out_path():
    today = RAW / f"e011b_graft_{DATE}.json"
    if today.exists():
        return today
    existing = sorted(RAW.glob("e011b_graft_*.json"))
    return existing[-1] if existing else today


# --- materialization (CPU-side) ----------------------------------------------

def param_names(model):
    return [k for k, _ in model.named_parameters()]


def type_keysets(names):
    """{type: set(param names)}; asserts the 8-type partition is complete."""
    sets = {t: set() for t in TYPES}
    for k in names:
        typ, _ = block_labels(k)
        assert typ in sets, f"unpartitioned param {k!r} -> {typ!r}"
        sets[typ].add(k)
    assert set().union(*sets.values()) == set(names)
    return sets


def layer_keyset(names, typ, layer):
    return {k for k in names if block_labels(k) == (typ, layer)}


def make_hybrid(base_model, donor_params, selection, alpha):
    """Deep-copy base (CPU), move `selection` params alpha toward the donor.
    selection: {name: True (whole tensor) | bool mask tensor}. Copy semantics
    at alpha==1.0 (donor verbatim); alpha==0.0 returns an untouched copy."""
    m = copy.deepcopy(base_model)
    with torch.no_grad():
        for k, p in m.named_parameters():
            sel = selection.get(k)
            if sel is None or alpha == 0.0:
                continue
            d = donor_params[k]
            if sel is True:
                if alpha == 1.0:
                    p.copy_(d)
                else:
                    p.copy_(p + alpha * (d - p))
            else:
                if alpha == 1.0:
                    p.copy_(torch.where(sel, d, p))
                else:
                    p.copy_(torch.where(sel, p + alpha * (d - p), p))
    m.eval()
    return m


def sel_norm2(base_params, donor_params, selection):
    """||Delta restricted to selection||^2 in float64."""
    s = 0.0
    for k, sel in selection.items():
        d = (donor_params[k].double() - base_params[k].double())
        if sel is True:
            s += float(d.pow(2).sum())
        else:
            s += float(d[sel].pow(2).sum())
    return s


def state_norm(model):
    return math.sqrt(sum(float(v.double().pow(2).sum())
                         for v in model.state_dict().values()))


def random_selection(names, shapes, p, seed):
    """Per-coordinate Bernoulli mask over all trainable coords (CPU generator,
    deterministic given seed). Returns ({name: bool mask}, actual_count)."""
    g = torch.Generator().manual_seed(seed)
    sel, count = {}, 0
    for k in names:
        mask = torch.rand(shapes[k], generator=g) < p
        sel[k] = mask
        count += int(mask.sum())
    return sel, count


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    if len(xs) < 2:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


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
    print(("resuming from " if OUT.exists() else "creating ") + str(OUT),
          flush=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {
        "experiment": "E-011b (R-2 spatial localization; D-017)",
        "date": DATE, "commit": git_head(), "torch": torch.__version__,
        "device": device, "deterministic_algorithms": det,
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "T": T, "alphas_t1": ALPHAS_T1, "alphas_t2": ALPHAS_T2,
        "thresholds": {
            "total_f015": TOTAL_F015, "rel_i3": REL_I3, "rel_i4": REL_I4,
            "rel_i5": REL_I5, "ce_parent_max": CE_PARENT_MAX,
            "delta_ce": DELTA_CE, "ce_void": CE_VOID,
            "ce_degraded": CE_DEGRADED, "seeds": SEEDS, "m_gate": M_GATE,
            "tie_frac": TIE_FRAC, "count_tol": COUNT_TOL,
            "void_limit_total": VOID_LIMIT_TOTAL,
            "void_limit_t1a1": VOID_LIMIT_T1A1, "dtheta_band": DTHETA_BAND,
        },
        "cells": {},
    }
    cells = results["cells"]

    def save():
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # --- models: originals live on CPU; anchor+hybrid only on GPU per cell ---
    model_A = load_model(REPO_A, T, "cpu")     # base ds1
    model_B = load_model(REPO_B, T, "cpu")     # donor ds2
    params_A = dict(model_A.named_parameters())
    params_B = dict(model_B.named_parameters())
    names = param_names(model_A)
    shapes = {k: params_A[k].shape for k in names}
    tsets = type_keysets(names)
    n_params = int(sum(params_A[k].numel() for k in names))
    results["n_params"] = n_params
    results["n_param_tensors"] = len(names)
    norm_A, norm_B = state_norm(model_A), state_norm(model_B)

    def prep_anchor(anchor):
        """Demote the non-anchor original to CPU, promote the anchor."""
        for m in (model_A, model_B):
            if m is not anchor:
                m.to("cpu")
        anchor.to(device)
        if device == "cuda":
            torch.cuda.empty_cache()
        return anchor

    # per-type ||Delta||^2 shares (arithmetic, 0 forward cells)
    d2 = {t: sel_norm2(params_A, params_B, {k: True for k in tsets[t]})
          for t in TYPES}
    d2_total = sum(d2.values())
    results["dtheta2_share"] = {t: d2[t] / d2_total for t in TYPES}

    def graft_cell(key, selection, alpha, base_model, donor, anchor,
                   base_norm, sel_n2, extra=None):
        """One certified-d_f cell: hybrid (passed FIRST -> ce_a) vs anchor.
        Materialize + d_theta on CPU, then GPU forward with 2 models peak."""
        if key in cells:
            return cells[key]
        m_h = make_hybrid(base_model, donor, selection, alpha)
        dtr, _ = d_theta_rel(m_h, base_model)          # CPU, certified
        norm_h = state_norm(m_h)
        pred = alpha * math.sqrt(sel_n2) / (0.5 * (base_norm + norm_h))
        i5_ok = (abs(dtr - pred) <= REL_I5 * max(pred, 1e-300)
                 and dtr < DTHETA_BAND) if alpha > 0 else True
        prep_anchor(anchor)
        m_h.to(device)
        df = d_f(m_h, anchor, tokens, device)          # certified, verbatim
        ce_h = df["ce_a"]
        cell = {
            "sym_kl": df["sym_kl"], "ce_h": ce_h, "ce_anchor": df["ce_b"],
            "disagree_rate": df["disagree_rate"],
            "dtheta_rel_vs_base": dtr, "dtheta_pred": pred, "i5_ok": i5_ok,
            "void": (ce_h > CE_VOID) or not i5_ok,
            "void_reason": ("ce_band" if ce_h > CE_VOID else
                            ("i5" if not i5_ok else None)),
            "degraded": CE_DEGRADED < ce_h <= CE_VOID,
        }
        if extra:
            cell.update(extra)
        cells[key] = cell
        del m_h
        if device == "cuda":
            cell["cuda_peak_bytes"] = int(torch.cuda.max_memory_allocated())
            torch.cuda.empty_cache()
        save()
        print(f"  [{key}] sym_kl={cell['sym_kl']:.6g} ce_h={ce_h:.4f}"
              + (" VOID" if cell["void"] else "")
              + (" DEGRADED" if cell["degraded"] else ""), flush=True)
        return cell

    # --- certs ---------------------------------------------------------------
    print("certs:", flush=True)
    if "I3:ds2_vs_ds1" not in cells:
        # the one two-original cell (exactly the E-009 run_cell precedent)
        model_A.to(device)
        model_B.to(device)
        df = d_f(model_B, model_A, tokens, device)
        cells["I3:ds2_vs_ds1"] = {
            "sym_kl": df["sym_kl"], "ce_ds2": df["ce_a"], "ce_ds1": df["ce_b"],
            "rel_err_vs_f015": abs(df["sym_kl"] - TOTAL_F015) / TOTAL_F015}
        if device == "cuda":
            cells["I3:ds2_vs_ds1"]["cuda_peak_bytes"] = int(
                torch.cuda.max_memory_allocated())
        model_B.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()
        save()
    i3 = cells["I3:ds2_vs_ds1"]
    total = i3["sym_kl"]              # in-run normalizer (registration)
    print(f"  [I3] d_f(ds2,ds1)={total:.10f} rel_err="
          f"{i3['rel_err_vs_f015']:.2e}", flush=True)

    sel_qkv = {k: True for k in tsets[REPRO_BLOCK]}
    n2_qkv = sel_norm2(params_A, params_B, sel_qkv)
    graft_cell("I1:attn_qkv@a0", sel_qkv, 0.0, model_A, params_B, model_A,
               norm_A, n2_qkv)
    sel_all = {k: True for k in names}
    n2_all = sel_norm2(params_A, params_B, sel_all)
    graft_cell("I2:ALL@a1_vs_ds2", sel_all, 1.0, model_A, params_B, model_B,
               norm_A, n2_all)
    graft_cell("I2:ALL@a1_vs_ds1", sel_all, 1.0, model_A, params_B, model_A,
               norm_A, n2_all)

    # --- Tier 1: 8 types x alphas vs ds1, + alpha=1 vs ds2 -------------------
    print("Tier 1:", flush=True)
    n2_type = {t: sel_norm2(params_A, params_B, {k: True for k in tsets[t]})
               for t in TYPES}
    for t in TYPES:
        sel = {k: True for k in tsets[t]}
        for a in ALPHAS_T1:
            graft_cell(f"T1:{t}@a{a}", sel, a, model_A, params_B, model_A,
                       norm_A, n2_type[t])
        graft_cell(f"T1r:{t}@a1_vs_ds2", sel, 1.0, model_A, params_B, model_B,
                   norm_A, n2_type[t])

    # (I-4) repro: fresh materialization of the designated cell
    if "repro:attn_qkv@a0.5" not in cells:
        m_h = make_hybrid(model_A, params_B, sel_qkv, REPRO_ALPHA)
        prep_anchor(model_A)
        m_h.to(device)
        df = d_f(m_h, model_A, tokens, device)
        first = cells[f"T1:{REPRO_BLOCK}@a{REPRO_ALPHA}"]["sym_kl"]
        rel = abs(df["sym_kl"] - first) / max(abs(first), 1e-300)
        cells["repro:attn_qkv@a0.5"] = {
            "sym_kl": df["sym_kl"], "first": first, "rel_err": rel}
        del m_h
        if device == "cuda":
            torch.cuda.empty_cache()
        save()
    print(f"  [I4] repro rel_err={cells['repro:attn_qkv@a0.5']['rel_err']:.2e}",
          flush=True)

    # --- b* (frozen argmax rule, alpha=1 with 0.5 fallback per block) --------
    def phi(key):
        c = cells.get(key)
        return None if c is None or c.get("void") else c["sym_kl"] / total

    score, t1a1_void = {}, 0
    for t in TYPES:
        v = phi(f"T1:{t}@a1.0")
        if v is None:
            t1a1_void += 1
            v = phi(f"T1:{t}@a0.5")
        score[t] = v
    ranked = sorted((t for t in TYPES if score[t] is not None),
                    key=lambda t: -score[t])
    bstar_undeterminable = t1a1_void > VOID_LIMIT_T1A1 or not ranked

    def finalize(g_entry, b_star=None, tie=None):
        graft_keys = [k for k in cells
                      if k.split(":")[0] in ("I1", "I2", "T1", "T1r", "RF",
                                             "T2", "SYM")]
        void_count = sum(1 for k in graft_keys if cells[k].get("void"))
        results["b_star"] = {"block": b_star, "score": score,
                             "tie_within_10pct": tie,
                             "t1_alpha1_void_count": t1a1_void,
                             "undeterminable": bstar_undeterminable}
        results["gates"] = {
            "I1_alpha0_identity": {
                "pass": cells["I1:attn_qkv@a0"]["sym_kl"] == 0.0,
                "sym_kl": cells["I1:attn_qkv@a0"]["sym_kl"]},
            "I2_partition_cert": {
                "pass": (cells["I2:ALL@a1_vs_ds2"]["sym_kl"] == 0.0 and
                         abs(cells["I2:ALL@a1_vs_ds1"]["sym_kl"] - total)
                         <= 1e-12 * total),
                "vs_ds2": cells["I2:ALL@a1_vs_ds2"]["sym_kl"],
                "vs_ds1": cells["I2:ALL@a1_vs_ds1"]["sym_kl"]},
            "I3_df_crosscheck": {"pass": i3["rel_err_vs_f015"] <= REL_I3,
                                 "rel_err": i3["rel_err_vs_f015"]},
            "I4_repro": {"pass": cells["repro:attn_qkv@a0.5"]["rel_err"]
                         <= REL_I4,
                         "rel_err": cells["repro:attn_qkv@a0.5"]["rel_err"]},
            "I5_materialization": {
                "pass": all(cells[k].get("i5_ok", True) for k in graft_keys)},
            "V_void_tally": {"void_count": void_count,
                             "voids": [k for k in graft_keys
                                       if cells[k].get("void")],
                             "inconclusive": void_count > VOID_LIMIT_TOTAL},
            "G_graft": g_entry,
        }
        save()

    if bstar_undeterminable:
        # frozen rule: (G) void, abort — never guess b*, spend nothing more
        finalize({"void": True, "reason": "b_star_undeterminable",
                  "alpha_g": None, "b_star": None, "phi_b_star": None,
                  "f_random": None, "margin": M_GATE, "pass": None})
        print("\nABORT: b* undeterminable "
              f"({t1a1_void} Tier-1 alpha=1 cells void) -> (G) void; "
              "RF/T2/SYM skipped per registration.", flush=True)
        print(f"done -> {OUT}", flush=True)
        return

    b_star = ranked[0]
    tie = (len(ranked) > 1 and
           (score[ranked[0]] - score[ranked[1]]) / score[ranked[0]] < TIE_FRAC)
    print(f"b* = {b_star} (phi={score[b_star]:.4f}, tie={tie})", flush=True)

    # --- random-subset floor (count-matched to |b*|) -------------------------
    print("random floor:", flush=True)
    n_bstar = int(sum(params_A[k].numel() for k in tsets[b_star]))
    results["n_bstar"] = n_bstar
    p_mask = n_bstar / n_params
    for seed in SEEDS:
        sel, count = random_selection(names, shapes, p_mask, seed)
        in_tol = abs(count - n_bstar) <= COUNT_TOL * n_bstar
        n2 = sel_norm2(params_A, params_B, sel)
        for a in ALPHAS_T2:
            graft_cell(f"RF:seed{seed}@a{a}", sel, a, model_A, params_B,
                       model_A, norm_A, n2,
                       extra={"mask_count": count, "count_in_tol": in_tol,
                              "d2_share_captured": n2 / d2_total})
        del sel

    # --- Tier 2: per-layer refinement of b* (frozen contingency) -------------
    t2_type = b_star if b_star in LAYERED_TYPES else next(
        (t for t in ranked if t in LAYERED_TYPES), None)
    results["tier2_type"] = t2_type
    if t2_type is not None:
        print(f"Tier 2 ({t2_type}):", flush=True)
        for layer in range(N_LAYERS):
            lset = layer_keyset(names, t2_type, layer)
            sel = {k: True for k in lset}
            n2 = sel_norm2(params_A, params_B, sel)
            for a in ALPHAS_T2:
                graft_cell(f"T2:L{layer}@a{a}", sel, a, model_A, params_B,
                           model_A, norm_A, n2)

    # --- symmetry spot-check: reverse graft of b* ----------------------------
    print("symmetry:", flush=True)
    sel_b = {k: True for k in tsets[b_star]}
    n2_rev = sel_norm2(params_B, params_A, sel_b)
    for a in ALPHAS_T2:
        graft_cell(f"SYM:rev@a{a}", sel_b, a, model_B, params_A, model_B,
                   norm_B, n2_rev)

    # --- gate (G) ------------------------------------------------------------
    def valid_rf(a):
        return [phi(f"RF:seed{s}@a{a}") for s in SEEDS
                if phi(f"RF:seed{s}@a{a}") is not None
                and cells[f"RF:seed{s}@a{a}"]["count_in_tol"]]

    alpha_g = None
    for a in (1.0, 0.5):
        if phi(f"T1:{b_star}@a{a}") is not None and len(valid_rf(a)) >= 3:
            alpha_g = a
            break
    if alpha_g is None:
        g_entry = {"void": True, "reason": "no_valid_alpha_g",
                   "alpha_g": None, "b_star": b_star, "phi_b_star": None,
                   "f_random": None, "margin": M_GATE, "pass": None}
    else:
        f_random = max(valid_rf(alpha_g))
        phi_b = phi(f"T1:{b_star}@a{alpha_g}")
        g_entry = {"void": False, "alpha_g": alpha_g, "b_star": b_star,
                   "phi_b_star": phi_b, "f_random": f_random,
                   "margin": M_GATE, "pass": phi_b > f_random + M_GATE}
    finalize(g_entry, b_star=b_star, tie=tie)

    # --- descriptive readouts ------------------------------------------------
    profiles = {t: {str(a): phi(f"T1:{t}@a{a}") for a in ALPHAS_T1}
                for t in TYPES}
    psi = {t: (cells[f"T1r:{t}@a1_vs_ds2"]["sym_kl"] / total
               if not cells[f"T1r:{t}@a1_vs_ds2"].get("void") else None)
           for t in TYPES}
    sum_phi = (sum(phi(f"T1:{t}@a1.0") for t in TYPES)
               if all(phi(f"T1:{t}@a1.0") is not None for t in TYPES)
               else None)

    def exponent(lo, hi, a_lo, a_hi):
        if lo and hi and lo > 0 and hi > 0:
            return math.log(hi / lo) / math.log(a_hi / a_lo)
        return None

    alpha_scaling = {t: {
        "p_early": exponent(phi(f"T1:{t}@a0.25"), phi(f"T1:{t}@a0.5"),
                            0.25, 0.5),
        "p_late": exponent(phi(f"T1:{t}@a0.5"), phi(f"T1:{t}@a1.0"),
                           0.5, 1.0)} for t in TYPES}

    ok = [t for t in TYPES if phi(f"T1:{t}@a1.0") is not None]
    sp_mag = spearman([phi(f"T1:{t}@a1.0") for t in ok],
                      [results["dtheta2_share"][t] for t in ok])
    sp_f017 = spearman([phi(f"T1:{t}@a1.0") for t in ok],
                       [F017_SHARE[t] for t in ok])
    sym = {str(a): phi(f"SYM:rev@a{a}") for a in ALPHAS_T2}
    fwd05, rev05 = phi(f"T1:{b_star}@a0.5"), sym.get("0.5")
    asym = (abs(fwd05 - rev05) / fwd05
            if fwd05 and rev05 is not None else None)

    results["descriptive"] = {
        "phi_profiles_t1": profiles, "psi_remain": psi,
        "sum_phi_alpha1": sum_phi, "alpha_scaling": alpha_scaling,
        "quadratic_reference_note":
            "p=2 is the pre-declared small-perturbation KL reference; "
            "p_hat is empirical (no alpha here is truly perturbative).",
        "spearman_phi_vs_dtheta2": sp_mag,
        "spearman_phi_vs_f017_share": sp_f017,
        "tier2_phi": {f"L{ly}": {str(a): phi(f"T2:L{ly}@a{a}")
                                 for a in ALPHAS_T2} for ly in range(N_LAYERS)},
        "symmetry_rev_phi": sym, "fwd_rev_gap_rel_a0.5": asym,
    }
    save()

    print(f"\ndone -> {OUT}", flush=True)
    print("GATES:", {g: (v.get("pass") if "pass" in v else v)
                     for g, v in results["gates"].items()}, flush=True)


if __name__ == "__main__":
    main()
