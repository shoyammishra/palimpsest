"""E-013a — camera-ready control for VyfE W2: is the low mid-training raw
endpoint-cosine specific to data order, or what any diverging trajectory shows
under the LR schedule? Registered in docs/experiment_log.md E-013a (2026-10-05,
D-021) BEFORE this code existed (git precedence is the freeze, commit e8ef761).

READ-ONLY, FORWARD-FREE. Reuses the E-011a load path (e009_audit_checkpoints.
load_params, the certified .bin path, buffers excluded) and E-011a's cosine()
unchanged. Touches no measured-number code path and no recorded JSON.

Curves, each cos(v_t, v_T) over the grid:
  R      P1 reference          v_t = th(ds1@t) - th(ds2@t)            (= F-017)
  J_raw  J1 raw (confounded)   v_t = th(seed1@t) - th(seed2@t)        NOT in rule
  J_acc  J1 init-removed       v_t = (th(seed1@t)-th(seed2@t)) - (same @0)
  S_run  single-run displacement v_t = th(run@t) - th(run@0), run in
         {ds1, ds2, seed1, seed2}

Frozen rule at t=16000, m=0.10, N = {J_acc, S_ds1, S_ds2, S_seed1, S_seed2}:
  ORDER-SPECIFIC if cos_P1 < min(N) - m
  GENERIC        if cos_P1 >= min(N) - m and cos_P1 >= median(N) - m
  MIXED          otherwise

Run:  HF_HUB_OFFLINE=1 python src/e013a_controls.py
"""

import datetime
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from e009_audit_checkpoints import load_params, compare  # noqa: E402
from e011a_direction import cosine, random_dict, git_head  # noqa: E402

ROOT = SRC.parent
RAW = ROOT / "results" / "raw"
DATE = datetime.date.today().isoformat()
OUT = RAW / f"e013a_controls_{DATE}.json"

# --- frozen constants (E-013a registration) ---------------------------------
GRID = [4, 16, 64, 256, 512, 1000, 4000, 16000, 64000, 128000, 143000]
T = 143000
GATE_CELL_T = 16000
MARGIN = 0.10
F017_P1_COS_16000 = 0.293705318182925   # E-011a recorded (I-1 reference)
NORM_NOISE_MULT = 1e3                   # I-4: ||v|| >= 1e3 x fp32 recompute noise

REPOS = {
    "ds1": "EleutherAI/pythia-160m-data-seed1",
    "ds2": "EleutherAI/pythia-160m-data-seed2",
    "seed1": "EleutherAI/pythia-160m-seed1",
    "seed2": "EleutherAI/pythia-160m-seed2",
}
E009_RAW = RAW / "e009_divergence_2026-07-20.json"
J1_TRIPWIRE = {"t_min": 4, "t_max": 1000, "d_theta_min": 0.5}


def theta(tag, step):
    sd, _ = load_params(REPOS[tag], f"step{step}")
    return sd


def sub(A, B, keys):
    return {k: A[k].float() - B[k].float() for k in keys}


def sqnorm(v, keys):
    return sum(float(v[k].double().pow(2).sum()) for k in keys)


def fp32_noise_floor(v, keys):
    """Scale of fp32 rounding in v: eps_fp32 * ||v|| (relative recompute noise)."""
    return torch.finfo(torch.float32).eps * math.sqrt(sqnorm(v, keys))


def pair_vec(a, b, t, keys, base=None):
    """Cross-run difference at t (optionally minus the step-0 difference)."""
    A, B = theta(a, t), theta(b, t)
    v = sub(A, B, keys)
    dtr = compare(A, B)["rel_l2"]
    del A, B
    if base is not None:
        v = {k: v[k] - base[k] for k in keys}
    return v, dtr


def run_vec(r, t, keys, th0):
    A = theta(r, t)
    v = {k: A[k].float() - th0[k] for k in keys}
    del A
    return v


def curve(name, vec_at, vT, keys):
    nT = math.sqrt(sqnorm(vT, keys))
    rows = []
    for t in GRID:
        v, extra = vec_at(t)
        c = cosine(v, vT, keys)
        noise = fp32_noise_floor(v, keys)
        void = c["norm_x"] < NORM_NOISE_MULT * noise if c["norm_x"] > 0 else True
        row = {"t": t, "cos": c["cos"], "norm_t": c["norm_x"], "void": void, **extra}
        rows.append(row)
        del v
        print(f"  [{name}@{t}] cos={c['cos']:.6f}" + (" VOID" if void else ""), flush=True)
    return {"endpoint_norm": nT, "curve": rows}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    try:
        torch.use_deterministic_algorithms(True)
        det = True
    except Exception:
        det = False

    out = {
        "experiment": "E-013a (camera-ready W2 control; D-021)",
        "date": DATE, "commit": git_head(), "torch": torch.__version__,
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "deterministic_algorithms": det, "grid": GRID, "T": T,
        "gate_cell_t": GATE_CELL_T, "margin": MARGIN, "curves": {}, "gates": {},
    }

    ref = theta("ds1", T)
    keys = sorted(ref)
    del ref

    # ---- R: P1 reference --------------------------------------------------
    print("R (P1):", flush=True)
    vT, _ = pair_vec("ds1", "ds2", T, keys)
    n_params = int(sum(vT[k].numel() for k in keys))
    out["n_params"] = n_params
    out["curves"]["R_P1"] = curve(
        "R", lambda t: (lambda v, d: (v, {"d_theta_rel": d}))(*pair_vec("ds1", "ds2", t, keys)),
        vT, keys)
    del vT

    # ---- J_raw and J_acc ---------------------------------------------------
    e009 = json.loads(E009_RAW.read_text(encoding="utf-8"))["cells"]
    s1_0, s2_0 = theta("seed1", 0), theta("seed2", 0)
    j0 = sub(s1_0, s2_0, keys)
    del s1_0, s2_0

    jT, _ = pair_vec("seed1", "seed2", T, keys)
    # I-2 designed identities + random control on the J1 endpoint vector
    neg = {k: -jT[k] for k in keys}
    rnd = random_dict(keys, jT)
    ident = {"self_cos": cosine(jT, jT, keys)["cos"],
             "neg_cos": cosine(jT, neg, keys)["cos"],
             "random_cos": cosine(jT, rnd, keys)["cos"],
             "random_expected_scale": math.sqrt(2.0 / (math.pi * n_params))}
    del neg, rnd

    def j_raw_at(t):
        v, d = pair_vec("seed1", "seed2", t, keys)
        rec = e009.get(f"J1@{t}", {}).get("d_theta_rel")
        return v, {"d_theta_rel": d, "d_theta_rel_e009": rec,
                   "d_theta_relerr_vs_e009": abs(d - rec) / rec if rec else None}

    print("J_raw (J1, confounded by init gap; not in rule):", flush=True)
    out["curves"]["J_raw"] = curve("J_raw", j_raw_at, jT, keys)

    jT_acc = {k: jT[k] - j0[k] for k in keys}
    del jT
    print("J_acc (J1 init-removed):", flush=True)
    out["curves"]["J_acc"] = curve(
        "J_acc", lambda t: (pair_vec("seed1", "seed2", t, keys, base=j0)[0], {}),
        jT_acc, keys)
    del jT_acc, j0

    # ---- S: single-run displacement ---------------------------------------
    for r in ("ds1", "ds2", "seed1", "seed2"):
        print(f"S_{r}:", flush=True)
        th0 = {k: v.float() for k, v in theta(r, 0).items() if k in set(keys)}
        dT = run_vec(r, T, keys, th0)
        out["curves"][f"S_{r}"] = curve(
            f"S_{r}", lambda t, r=r, th0=th0: (run_vec(r, t, keys, th0), {}), dT, keys)
        del th0, dT

    # ---- gates --------------------------------------------------------------
    cos_at = lambda name, t: next(x["cos"] for x in out["curves"][name]["curve"] if x["t"] == t)
    c_p1 = cos_at("R_P1", GATE_CELL_T)
    i1_rel = abs(c_p1 - F017_P1_COS_16000) / F017_P1_COS_16000
    i2_pass = (abs(ident["self_cos"] - 1) <= 1e-12 and abs(ident["neg_cos"] + 1) <= 1e-12
               and abs(ident["random_cos"]) <= 5e-4)
    i3_bad = {}
    for x in out["curves"]["J_raw"]["curve"]:
        e = x.get("d_theta_relerr_vs_e009")
        if e is not None and e > 1e-3:
            i3_bad[f"J1@{x['t']}_relerr"] = e
        if (J1_TRIPWIRE["t_min"] <= x["t"] <= J1_TRIPWIRE["t_max"]
                and x["d_theta_rel"] < J1_TRIPWIRE["d_theta_min"]):
            i3_bad[f"J1@{x['t']}_tripwire"] = x["d_theta_rel"]
    voids = {n: [x["t"] for x in c["curve"] if x["void"]] for n, c in out["curves"].items()}
    gate_voids = {n: v for n, v in voids.items() if GATE_CELL_T in v}

    out["designed_identities"] = ident
    out["gates"] = {
        "I1_repro_vs_F017": {"pass": i1_rel <= 1e-6, "rel_err": i1_rel, "cos": c_p1,
                             "reference": F017_P1_COS_16000},
        "I2_designed": {"pass": i2_pass, **ident},
        "I3_J1_continuity": {"pass": not i3_bad, "violations": i3_bad},
        "I4_norm_noise": {"pass": not gate_voids, "void_cells": voids},
    }

    # ---- frozen decision rule ----------------------------------------------
    null_names = ["J_acc", "S_ds1", "S_ds2", "S_seed1", "S_seed2"]
    N = {n: cos_at(n, GATE_CELL_T) for n in null_names}
    mn, med = min(N.values()), statistics.median(N.values())
    if c_p1 < mn - MARGIN:
        verdict = "ORDER-SPECIFIC"
    elif c_p1 >= mn - MARGIN and c_p1 >= med - MARGIN:
        verdict = "GENERIC"
    else:
        verdict = "MIXED"
    instrument_ok = all(g["pass"] for g in out["gates"].values())
    out["decision"] = {"cos_P1": c_p1, "nulls": N, "min_N": mn, "median_N": med,
                       "verdict": verdict if instrument_ok else "INCONCLUSIVE (instrument gate failed)",
                       "J_raw_16000_not_in_rule": cos_at("J_raw", GATE_CELL_T)}

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\ndone -> {OUT}")
    print("GATES:", {g: v["pass"] for g, v in out["gates"].items()})
    print(json.dumps(out["decision"], indent=2))


if __name__ == "__main__":
    main()
