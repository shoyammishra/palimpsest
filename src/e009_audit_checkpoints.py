"""E-009 phase-0b — checkpoint TRAJECTORY-CONTINUITY audit for the D-013 pairs.

Motivated by the C1@1 anomaly in results/raw/e009_divergence_2026-07-20.json
(sym_kl = 0 for an init-only pair). The D-013 gates (src/e009_gates.py) only
inspected `step0`, which is the one revision where the decoupled repos are
well-formed; this script checks the revisions the divergence curve actually
consumes.

Two gates, both about the *checkpoints*, not the metric:

  G-v  (trajectory continuity)  For a run R whose init is distinct from a
       reference init, R must stay near its own init over the first few
       optimizer steps:
           cont(R, t) := d(R@step0, R@t) / d(R@step0, REF@step0)
       cont ≈ 0 for small t. cont ≈ 1 means R@t sits as far from R@step0 as an
       independent random init does — i.e. R@t is NOT a continuation of R@step0.

  G-vi (pair non-degeneracy)  For a pair (A, B) intended to differ,
       d(A@t, B@t) > 0 at every t >= 1. Zero means the two branches serve the
       same numbers.

Distances are numerical (relative L2 over the common *parameter* tensors), not
byte hashes: these repos store fp16 and differ in signed-zero bits (-0.0 vs
+0.0) across separately-uploaded files, which is a serialization artifact and
NOT a weight difference. Hashing raw bytes reports such pairs as different;
`torch.equal` correctly reports them equal. Buffers (causal masks, `masked_bias`
= -inf, `inv_freq`) are excluded — `masked_bias` alone would poison every norm
with inf/nan.

Read-only: forward-free, no training, no gradients. Uses whatever is in the HF
cache; set HF_HUB_OFFLINE=1 to guarantee no network spend.
"""

import datetime
import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
DATE = datetime.date.today().isoformat()
OUT = RAW / f"e009_ckpt_audit_{DATE}.json"

# non-learnable buffers present in the fp16 bins but not in model.state_dict()
BUFFER_SUFFIXES = ("attention.bias", "masked_bias", "inv_freq")

REPOS = {
    "ds1": "EleutherAI/pythia-160m-data-seed1",
    "ds2": "EleutherAI/pythia-160m-data-seed2",
    "ws1": "EleutherAI/pythia-160m-weight-seed1",
    "ws2": "EleutherAI/pythia-160m-weight-seed2",
}
REVS = ["step0", "step1", "step4", "step16", "step64", "step256"]

# pairs as declared in D-013 / E-009
PAIRS = {"P1": ("ds1", "ds2"), "C1": ("ws1", "ws2")}
REF = "ds1"  # reference init for the continuity denominator


def load_params(repo, rev):
    """Parameter tensors only, from the checkpoint file itself."""
    p = hf_hub_download(repo, "pytorch_model.bin", revision=rev)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    return {k: v for k, v in sd.items() if not k.endswith(BUFFER_SUFFIXES)}, p


def compare(sa, sb):
    keys = sorted(set(sa) & set(sb))
    d2 = a2 = b2 = 0.0
    ndiff = 0
    maxabs = 0.0
    for k in keys:
        ta, tb = sa[k].double(), sb[k].double()
        m = (ta - tb).abs().max().item()
        if m > 0:
            ndiff += 1
        maxabs = max(maxabs, m)
        d2 += (ta - tb).pow(2).sum().item()
        a2 += ta.pow(2).sum().item()
        b2 += tb.pow(2).sum().item()
    denom = 0.5 * (a2 ** 0.5 + b2 ** 0.5)
    return {
        "rel_l2": (d2 ** 0.5 / denom) if denom > 0 else float("nan"),
        "n_keys": len(keys), "n_differing": ndiff, "max_abs_diff": maxabs,
    }


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment": "E-009 phase 0b (checkpoint trajectory-continuity audit)",
        "date": DATE, "torch": torch.__version__,
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "repos": REPOS, "revisions": REVS,
        "buffer_suffixes_excluded": list(BUFFER_SUFFIXES),
        "cells": {}, "gates": {},
    }

    cache = {}

    def get(tag, rev):
        if (tag, rev) not in cache:
            cache[(tag, rev)] = load_params(REPOS[tag], rev)
        return cache[(tag, rev)]

    # --- all pairwise distances at each revision ---
    tags = list(REPOS)
    for rev in REVS:
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                key = f"{a}|{b}@{rev}"
                try:
                    (sa, _), (sb, _) = get(a, rev), get(b, rev)
                    out["cells"][key] = compare(sa, sb)
                except Exception as e:
                    out["cells"][key] = {"void": True, "error": repr(e)}
                print(f"[{key}] {out['cells'][key]}", flush=True)

    # --- continuity: does each run stay near its own init? ---
    try:
        ref0, _ = get(REF, "step0")
        init_scale = {}
        for tag in tags:
            if tag == REF:
                continue
            s0, _ = get(tag, "step0")
            init_scale[tag] = compare(ref0, s0)["rel_l2"]
        out["init_scale_vs_ref"] = init_scale

        cont = {}
        for tag in tags:
            s0, _ = get(tag, "step0")
            scale = init_scale.get(tag)
            if not scale or scale == 0:
                continue  # same init as REF -> no meaningful denominator
            for rev in REVS[1:]:
                st, _ = get(tag, rev)
                cont[f"{tag}@{rev}"] = compare(s0, st)["rel_l2"] / scale
        out["continuity_ratio"] = cont
        # G-v: a continuation must be far closer to its own init than an
        # independent init is. Threshold 0.5 is generous by orders of magnitude.
        bad = {k: v for k, v in cont.items() if v > 0.5}
        out["gates"]["G-v_trajectory_continuity"] = {
            "pass": not bad, "threshold": 0.5, "violations": bad,
        }
    except Exception as e:
        out["gates"]["G-v_trajectory_continuity"] = {"void": True, "error": repr(e)}

    # --- G-vi: intended-to-differ pairs must actually differ at t >= 1 ---
    degenerate = {}
    for pname, (a, b) in PAIRS.items():
        for rev in REVS:
            if rev == "step0" and pname == "P1":
                continue  # P1 shares its init by construction (G-i)
            c = out["cells"].get(f"{a}|{b}@{rev}") or out["cells"].get(f"{b}|{a}@{rev}")
            if c and not c.get("void") and c["n_differing"] == 0:
                degenerate[f"{pname}@{rev}"] = c["rel_l2"]
    out["gates"]["G-vi_pair_non_degeneracy"] = {
        "pass": not degenerate, "degenerate_cells": degenerate,
    }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\ndone -> {OUT}")
    for g, v in out["gates"].items():
        print(f"  {g}: {'PASS' if v.get('pass') else 'FAIL'}")


if __name__ == "__main__":
    main()
