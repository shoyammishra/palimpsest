"""E-011a — R-1 temporal localization: directional persistence of the endpoint
order-difference. Registered in docs/experiment_log.md E-011a (2026-07-22, D-017)
BEFORE this code existed (git precedence is the freeze).

READ-ONLY, FORWARD-FREE. Loads cached parameter vectors only — no forward
passes, no gradients, no training. Touches NO measured-number code path of
E-009/E-010 and NO recorded JSON (comparability, D-012); writes its own raw file.

Load paths (both certified, reused unchanged):
  * ds1/ds2  -> e009_audit_checkpoints.load_params (the F-014 .bin path;
    buffers excluded, numerical comparison, signed-zero safe).
  * main     -> e009_divergence.load_model(...).state_dict() (from_pretrained
    safetensors, the F-016 pattern) so main's cached safetensors are reused with
    ZERO new downloads (its pytorch_model.bin is not cached for the grid).

Object: Delta_t := theta(A@t) - theta(B@t) over the 148 trainable tensors.
Primary readout: cos(Delta_t, Delta_T) across the grid, computed with float64
accumulation (fp32/fp64 upcast; the fp16 STORAGE floor is what bounds early-t
interpretability, not the compute dtype).

Frozen gates (E-011a): (I-1) repro of cos(Delta_16000, Delta_T) <= 1e-6;
(I-2) designed-identity cells self=+1, neg=-1, random~1/sqrt(n); (I-3) F-014
continuity/non-degeneracy tripwire (d_theta cross-checked vs F-015 raw <= 1e-3);
(F-1) random-direction floor; (F-2) same-run drift-coherence floor at t=16000.
Descriptive (NOT gated): t* timing, per-type/per-layer (W) decomposition, and
the P2/P3 correlated replicates (F-016 caveat).

Run:  HF_HUB_OFFLINE=1 python src/e011a_direction.py
"""

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
from e009_audit_checkpoints import load_params, compare, BUFFER_SUFFIXES  # noqa: E402

ROOT = SRC.parent
RAW = ROOT / "results" / "raw"
DATE = datetime.date.today().isoformat()
OUT = RAW / f"e011a_direction_{DATE}.json"

# --- frozen grid / constants (E-011a registration) --------------------------
# t=0 excluded (Delta_0 ~ 0, undefined direction); t=1 void (F-014).
GRID = [4, 16, 64, 256, 512, 1000, 4000, 16000, 64000, 128000, 143000]
T = 143000
GATE_CELL_T = 16000            # F-2 gate cell (late window)
LATE_WINDOW_MIN = 16000        # F-1 gated over t >= this
F2_MARGIN = 0.10               # frozen
T_STAR_THRESHOLD = 0.9         # frozen; t* is DESCRIPTIVE-only
QUANT_FLOOR = 5e-3             # d_theta_rel below this = fp16-quantization-limited
RANDOM_SEED = 20260722
F1_RANDOM_MULT = 20            # |cos| must exceed MULT/sqrt(n)

REPOS = {
    "ds1": "EleutherAI/pythia-160m-data-seed1",
    "ds2": "EleutherAI/pythia-160m-data-seed2",
    "main": "EleutherAI/pythia-160m",
}
PAIRS = {"P1": ("ds1", "ds2"), "P2": ("main", "ds1"), "P3": ("main", "ds2")}
GATED_PAIR = "P1"
FLOOR_RUN = "ds1"
FLOOR_WINDOWS = [(1000, 2000), (16000, 17000), (142000, 143000)]  # same-run ds1 drift

# F-015 recorded P1 d_theta_rel (results/raw/e009_divergence_2026-07-20.json)
# — the I-3 cross-check reference. Cited, never re-measured.
F015_P1_DTHETA = {
    4: 2.438512532022806e-05, 16: 3.908250215936978e-04,
    64: 3.2335390858213658e-03, 256: 0.025188091938048083,
    512: 0.07956511598292232, 1000: 0.2239392442554392,
    4000: 0.7019051630663556, 16000: 0.965503254712774,
    64000: 0.9640058546072626, 128000: 0.9608667840535671,
    143000: 0.953914612259891,
}


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception as e:  # pragma: no cover
        return f"unknown ({e!r})"


# --- parameter loading (certified paths) ------------------------------------
_main_loader = None


def load_theta(tag, step):
    """Return {key: cpu tensor} of the 148 trainable params (buffers excluded)."""
    if tag == "main":
        global _main_loader
        if _main_loader is None:
            from e009_divergence import load_model
            _main_loader = load_model
        m = _main_loader(REPOS[tag], step, "cpu")
        sd = {k: v for k, v in m.state_dict().items()
              if not k.endswith(BUFFER_SUFFIXES)}
        del m
        return sd
    sd, _ = load_params(REPOS[tag], f"step{step}")
    return sd


def common_keys(a, b):
    keys = sorted(set(a) & set(b))
    return keys


def block_labels(key):
    """(type, layer) for a param key; layer is int or None."""
    if key == "gpt_neox.embed_in.weight":
        return "embed_in", None
    if key == "embed_out.weight":
        return "embed_out", None
    if key.startswith("gpt_neox.final_layer_norm"):
        return "final_ln", None
    layer = None
    if key.startswith("gpt_neox.layers."):
        layer = int(key.split(".")[2])
    if "input_layernorm" in key or "post_attention_layernorm" in key:
        return "ln", layer
    if "attention.query_key_value" in key:
        return "attn_qkv", layer
    if "attention.dense" in key:
        return "attn_dense", layer
    if "mlp.dense_h_to_4h" in key:
        return "mlp_in", layer
    if "mlp.dense_4h_to_h" in key:
        return "mlp_out", layer
    return "OTHER", layer


def diff_dict(a_tag, b_tag, step, keys):
    """Delta = A - B as fp32 dict over `keys`, plus certified d_theta_rel(A,B)."""
    A = load_theta(a_tag, step)
    B = load_theta(b_tag, step)
    d = {k: (A[k].float() - B[k].float()) for k in keys}
    dtr = compare(A, B)["rel_l2"]   # certified rel-L2 over common keys (double)
    del A, B
    return d, dtr


def cosine(X, Y, keys, blocks=False):
    """float64-accumulated cosine of two fp32 difference-dicts; optional
    per-type / per-layer breakdown (cos and share-of-global-inner-product)."""
    dotG = nX = nY = 0.0
    tdot, tnx, tny = {}, {}, {}   # per-type
    ldot, lnx, lny = {}, {}, {}   # per-layer
    for k in keys:
        x = X[k].double()
        y = Y[k].double()
        dk = float((x * y).sum())
        xk = float(x.pow(2).sum())
        yk = float(y.pow(2).sum())
        dotG += dk
        nX += xk
        nY += yk
        if blocks:
            typ, lay = block_labels(k)
            tdot[typ] = tdot.get(typ, 0.0) + dk
            tnx[typ] = tnx.get(typ, 0.0) + xk
            tny[typ] = tny.get(typ, 0.0) + yk
            lk = "none" if lay is None else str(lay)
            ldot[lk] = ldot.get(lk, 0.0) + dk
            lnx[lk] = lnx.get(lk, 0.0) + xk
            lny[lk] = lny.get(lk, 0.0) + yk
    cosG = dotG / (math.sqrt(nX) * math.sqrt(nY)) if nX > 0 and nY > 0 else float("nan")
    out = {"cos": cosG, "dot": dotG, "norm_x": math.sqrt(nX), "norm_y": math.sqrt(nY)}
    if blocks:
        def pack(dd, nx, ny):
            r = {}
            for g in dd:
                denom = math.sqrt(nx[g]) * math.sqrt(ny[g])
                r[g] = {"cos": dd[g] / denom if denom > 0 else float("nan"),
                        "share": dd[g] / dotG if dotG != 0 else float("nan")}
            return r
        out["by_type"] = pack(tdot, tnx, tny)
        out["by_layer"] = pack(ldot, lnx, lny)
    return out


def random_dict(keys, ref):
    g = torch.Generator().manual_seed(RANDOM_SEED)
    return {k: torch.randn(ref[k].shape, generator=g, dtype=torch.float32) for k in keys}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    try:
        torch.use_deterministic_algorithms(True)
        det = True
    except Exception:
        det = False

    out = {
        "experiment": "E-011a (R-1 temporal localization; D-017)",
        "date": DATE, "commit": git_head(), "torch": torch.__version__,
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "deterministic_algorithms": det,
        "grid": GRID, "T": T, "gate_cell_t": GATE_CELL_T,
        "thresholds": {
            "f2_margin": F2_MARGIN, "t_star_threshold": T_STAR_THRESHOLD,
            "quant_floor_d_theta": QUANT_FLOOR, "random_seed": RANDOM_SEED,
            "f1_random_mult": F1_RANDOM_MULT, "late_window_min": LATE_WINDOW_MIN,
        },
        "pairs": {}, "gates": {},
    }

    # ---- endpoint delta for the gated pair (P1), n_params, designed cells ----
    a0, b0 = PAIRS[GATED_PAIR]
    ATd = load_theta(a0, T)
    BTd = load_theta(b0, T)
    keys = common_keys(ATd, BTd)
    dT = {k: (ATd[k].float() - BTd[k].float()) for k in keys}
    del ATd, BTd
    n_params = int(sum(dT[k].numel() for k in keys))
    out["n_params"] = n_params
    out["n_param_tensors"] = len(keys)
    nT2 = sum(float(dT[k].double().pow(2).sum()) for k in keys)

    # (I-2) designed-identity audit cells — self=+1, neg=-1, random~1/sqrt(n)
    self_cos = cosine(dT, dT, keys)["cos"]
    neg = {k: -dT[k] for k in keys}
    neg_cos = cosine(dT, neg, keys)["cos"]
    del neg
    rnd = random_dict(keys, dT)
    random_cos = cosine(dT, rnd, keys)["cos"]
    del rnd
    expected_random_scale = math.sqrt(2.0 / (math.pi * n_params))
    out["designed_identities"] = {
        "self_cos": self_cos, "neg_cos": neg_cos, "random_cos": random_cos,
        "random_expected_scale": expected_random_scale,
        "note": ("self=+1 / neg=-1 are DESIGNED normalization certs, NOT signal; "
                 "random_cos ~ sqrt(2/(pi*n)) is the random-direction floor scale."),
    }

    # ---- P1 curve (gated) ----
    def build_curve(pair):
        a_tag, b_tag = PAIRS[pair]
        # pair-specific endpoint delta + keys (P1 reuses dT/nT2/keys)
        if pair == GATED_PAIR:
            pkeys, pdT, pnT2 = keys, dT, nT2
        else:
            ATp = load_theta(a_tag, T)
            BTp = load_theta(b_tag, T)
            pkeys = common_keys(ATp, BTp)
            pdT = {k: (ATp[k].float() - BTp[k].float()) for k in pkeys}
            del ATp, BTp
            pnT2 = sum(float(pdT[k].double().pow(2).sum()) for k in pkeys)
        curve = []
        for t in GRID:
            dt, dtr = diff_dict(a_tag, b_tag, t, pkeys)
            c = cosine(dt, pdT, pkeys, blocks=True)
            del dt
            f015 = F015_P1_DTHETA.get(t) if pair == "P1" else None
            dtr_relerr = (abs(dtr - f015) / f015) if f015 else None
            curve.append({
                "t": t, "cos": c["cos"], "norm_delta_t": c["norm_x"],
                "d_theta_rel": dtr, "d_theta_rel_f015": f015,
                "d_theta_relerr_vs_f015": dtr_relerr,
                "quant_limited": dtr < QUANT_FLOOR,
                "by_type": c["by_type"], "by_layer": c["by_layer"],
            })
            print(f"  [{pair}@{t}] cos={c['cos']:.6f} d_theta={dtr:.6g}"
                  + (f" (f015 relerr {dtr_relerr:.2e})" if dtr_relerr is not None else "")
                  + (" QUANT-LIMITED" if dtr < QUANT_FLOOR else ""), flush=True)
        # t* (descriptive): first grid t with cos >= threshold
        tstar = next((r["t"] for r in curve if r["cos"] >= T_STAR_THRESHOLD), None)
        if pair != GATED_PAIR:
            del pdT
        return {"pair": pair, "runs": [a_tag, b_tag], "endpoint_norm": math.sqrt(pnT2),
                "curve": curve, "t_star": tstar}

    print("P1 (gated):", flush=True)
    out["pairs"]["P1"] = build_curve("P1")

    # (I-1) reproducibility: recompute cos(Delta_16000, Delta_T) fresh
    dt_rep, _ = diff_dict(a0, b0, GATE_CELL_T, keys)
    cos_rep = cosine(dt_rep, dT, keys)["cos"]
    del dt_rep
    cos_first = next(r["cos"] for r in out["pairs"]["P1"]["curve"] if r["t"] == GATE_CELL_T)
    repro_rel = abs(cos_first - cos_rep) / max(abs(cos_first), 1e-300)
    out["repro"] = {"cos_first": cos_first, "cos_second": cos_rep,
                    "rel_err": repro_rel, "pass": repro_rel <= 1e-6}

    # ---- F-2: same-run drift-coherence floor (ds1 adjacent windows) ----
    # drift vector D = ds1@t2 - ds1@t1 over each adjacent window
    drift_vecs = {}
    for (t1, t2) in FLOOR_WINDOWS:
        A = load_theta(FLOOR_RUN, t2)
        B = load_theta(FLOOR_RUN, t1)
        drift_vecs[(t1, t2)] = {k: (A[k].float() - B[k].float()) for k in keys}
        del A, B
    wlist = FLOOR_WINDOWS
    pairwise = {}
    for i in range(len(wlist)):
        for j in range(i + 1, len(wlist)):
            cij = cosine(drift_vecs[wlist[i]], drift_vecs[wlist[j]], keys)["cos"]
            pairwise[f"{wlist[i]}|{wlist[j]}"] = cij
    C_drift = max(pairwise.values())
    del drift_vecs
    out["floor_drift"] = {"windows": [list(w) for w in FLOOR_WINDOWS],
                          "pairwise_cos": pairwise, "C_drift": C_drift}

    # ---- gate evaluations (P1 only) ----
    p1 = out["pairs"]["P1"]["curve"]
    cos_gate = next(r["cos"] for r in p1 if r["t"] == GATE_CELL_T)
    f1_thr = F1_RANDOM_MULT / math.sqrt(n_params)
    late = [r for r in p1 if r["t"] >= LATE_WINDOW_MIN]
    f1_pass = all(abs(r["cos"]) > f1_thr for r in late)
    f2_pass = cos_gate > C_drift + F2_MARGIN

    i2_pass = (abs(self_cos - 1.0) <= 1e-12 and abs(neg_cos + 1.0) <= 1e-12
               and abs(random_cos) <= 5e-4)
    # I-3 tripwire: P1 d_theta cross-check vs F-015 and shared-init band
    i3_bad = {}
    for r in p1:
        if r["d_theta_relerr_vs_f015"] is not None and r["d_theta_relerr_vs_f015"] > 1e-3:
            i3_bad[f"P1@{r['t']}"] = r["d_theta_relerr_vs_f015"]
        if r["d_theta_rel"] > 1.2:   # collapse toward independent-init scale
            i3_bad[f"P1@{r['t']}_band"] = r["d_theta_rel"]

    out["gates"] = {
        "I1_repro": {"pass": out["repro"]["pass"], "rel_err": out["repro"]["rel_err"],
                     "threshold": 1e-6},
        "I2_designed": {"pass": i2_pass, "self_cos": self_cos, "neg_cos": neg_cos,
                        "random_cos": random_cos},
        "I3_tripwire": {"pass": not i3_bad, "violations": i3_bad,
                        "note": "d_theta cross-check vs F-015 raw (rel<=1e-3) + band<1.2"},
        "F1_random": {"pass": f1_pass, "threshold": f1_thr,
                      "late_window_min": LATE_WINDOW_MIN,
                      "late_abs_cos": {r["t"]: abs(r["cos"]) for r in late}},
        "F2_drift": {"pass": f2_pass, "cos_gate_cell": cos_gate, "C_drift": C_drift,
                     "margin": F2_MARGIN, "gate_cell_t": GATE_CELL_T},
    }

    # ---- P2/P3 replicates (descriptive; F-016 correlated-pairs caveat) ----
    for pair in ("P2", "P3"):
        print(f"{pair} (replicate):", flush=True)
        out["pairs"][pair] = build_curve(pair)

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\ndone -> {OUT}")
    print("GATES:", {g: v.get("pass") for g, v in out["gates"].items()})
    print(f"C_drift={C_drift:.4f}  cos(Delta_16000,Delta_T)={cos_gate:.4f}  "
          f"F2 {'PASS' if f2_pass else 'FAIL'} (need > {C_drift + F2_MARGIN:.4f})")
    print("t* (P1/P2/P3):", [out['pairs'][p]['t_star'] for p in ('P1', 'P2', 'P3')])


if __name__ == "__main__":
    main()
