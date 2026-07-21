"""E-009 phase 1 — checkpoint-matched divergence curves for PolyPythias 160M pairs.

Pre-registered in docs/experiment_log.md E-009 (2026-07-20). Runs ONLY if the
phase-0 gates passed (src/e009_gates.py exit 0).

Cells: d_f (mean symmetric KL over ~98k eval tokens) and d_theta (relative L2)
for pairs P1 (order-only), C1 (init-only), J1 (joint) on the frozen checkpoint
grid; plus floor (same-run adjacent-checkpoint drift), ceiling (cross-size gap),
and one pre-declared reproducibility cell (P1@16000 twice, rel. 1e-6).

Incremental/resumable: results JSON is rewritten after every cell; existing
cells are skipped on rerun. Forward evals only, no gradients.
"""

import datetime
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer, GPTNeoXForCausalLM

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
DATE = datetime.date.today().isoformat()
TOKENS_NPY = RAW / "e009_eval_tokens.npy"


def resolve_out_path():
    """Single resume target for this experiment: today's file if present, else
    the most recent existing e009_divergence_*.json (append to it — never start
    a second partial file), else a fresh today-stamped file. Date-stamped names
    sort chronologically, so the last lexical match is the newest."""
    today = RAW / f"e009_divergence_{DATE}.json"
    if today.exists():
        return today
    existing = sorted(RAW.glob("e009_divergence_*.json"))
    return existing[-1] if existing else today

GRID = [0, 1, 4, 16, 64, 256, 512, 1000, 4000, 16000, 64000, 128000, 143000]
N_BLOCKS, BLOCK = 48, 2048

PAIRS = {
    "P1": ("EleutherAI/pythia-160m-data-seed1", "EleutherAI/pythia-160m-data-seed2"),
    "C1": ("EleutherAI/pythia-160m-weight-seed1", "EleutherAI/pythia-160m-weight-seed2"),
    "J1": ("EleutherAI/pythia-160m-seed1", "EleutherAI/pythia-160m-seed2"),
}
FLOOR_RUN = "EleutherAI/pythia-160m-data-seed1"
FLOOR_ANCHORS = [(1000, 2000), (16000, 17000), (142000, 143000)]
CEILING = ("EleutherAI/pythia-160m", "EleutherAI/pythia-410m", 143000)


def build_eval_tokens():
    """First 48 non-overlapping 2048-token blocks of NeelNanda/pile-10k,
    docs concatenated in dataset order with EOS separators (frozen shard)."""
    if TOKENS_NPY.exists():
        return np.load(TOKENS_NPY)
    import pyarrow.parquet as pq
    repo = "NeelNanda/pile-10k"
    fname = next(f for f in sorted(list_repo_files(repo, repo_type="dataset"))
                 if f.endswith(".parquet"))
    path = hf_hub_download(repo, fname, repo_type="dataset")
    texts = pq.read_table(path, columns=["text"])["text"].to_pylist()
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    eos = tok.eos_token_id
    ids = []
    need = N_BLOCKS * BLOCK
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        ids.append(eos)
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise RuntimeError(f"shard too small: {len(ids)} < {need}")
    arr = np.array(ids[:need], dtype=np.int32).reshape(N_BLOCKS, BLOCK)
    np.save(TOKENS_NPY, arr)
    return arr


def load_model(repo, step, device):
    m = GPTNeoXForCausalLM.from_pretrained(
        repo, revision=f"step{step}", torch_dtype=torch.float32)
    m.eval().to(device)
    return m


def d_theta_rel(ma, mb):
    """Relative L2 param distance. Returns (value, reason); value is None with a
    reason when key sets or shapes differ (e.g. the cross-size ceiling cell) so
    the caller keeps d_f rather than voiding the whole cell."""
    sa, sb = ma.state_dict(), mb.state_dict()
    if set(sa) != set(sb):
        return None, "key_mismatch"
    diff2 = na2 = nb2 = 0.0
    for k in sa:
        if sa[k].shape != sb[k].shape:
            return None, "shape_mismatch"
        ta, tb = sa[k].double(), sb[k].double()
        diff2 += (ta - tb).pow(2).sum().item()
        na2 += ta.pow(2).sum().item()
        nb2 += tb.pow(2).sum().item()
    return diff2 ** 0.5 / (0.5 * (na2 ** 0.5 + nb2 ** 0.5)), None


@torch.no_grad()
def d_f(ma, mb, tokens, device):
    """Mean-over-token symmetric KL + secondaries, accumulated in float64."""
    kl_ab = kl_ba = dis = ce_a = ce_b = 0.0
    n_pos = n_tgt = 0
    for i in range(tokens.shape[0]):
        x = torch.from_numpy(tokens[i].astype(np.int64)).unsqueeze(0).to(device)
        la = ma(x).logits.float().squeeze(0)
        lb = mb(x).logits.float().squeeze(0)
        lpa, lpb = torch.log_softmax(la, -1), torch.log_softmax(lb, -1)
        pa, pb = lpa.exp(), lpb.exp()
        kl_ab += (pa * (lpa - lpb)).sum(-1).double().sum().item()
        kl_ba += (pb * (lpb - lpa)).sum(-1).double().sum().item()
        dis += (lpa.argmax(-1) != lpb.argmax(-1)).double().sum().item()
        tgt = x.squeeze(0)[1:]
        ce_a += torch.nn.functional.cross_entropy(
            lpa[:-1], tgt, reduction="sum").double().item()
        ce_b += torch.nn.functional.cross_entropy(
            lpb[:-1], tgt, reduction="sum").double().item()
        n_pos += lpa.shape[0]
        n_tgt += tgt.shape[0]
    return {
        "sym_kl": 0.5 * (kl_ab + kl_ba) / n_pos,
        "kl_ab": kl_ab / n_pos, "kl_ba": kl_ba / n_pos,
        "disagree_rate": dis / n_pos,
        "ce_a": ce_a / n_tgt, "ce_b": ce_b / n_tgt,
        "n_pos": n_pos, "n_tgt": n_tgt,
    }


def run_cell(repo_a, step_a, repo_b, step_b, tokens, device):
    ma = load_model(repo_a, step_a, device)
    mb = load_model(repo_b, step_b, device)
    out = d_f(ma, mb, tokens, device)
    val, reason = d_theta_rel(ma, mb)   # non-fatal: d_f must survive a mismatch
    out["d_theta_rel"] = val
    if reason is not None:
        out["d_theta_reason"] = reason
    del ma, mb
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


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
    if OUT.exists():
        print(f"resuming from {OUT}", flush=True)
    else:
        print(f"no prior file — creating {OUT}", flush=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {
        "experiment": "E-009 phase 1 (divergence curves)", "date": DATE,
        "device": device, "deterministic_algorithms": det,
        "torch": torch.__version__, "grid": GRID,
        "n_blocks": N_BLOCKS, "block_len": BLOCK, "cells": {},
    }
    cells = results["cells"]

    def save():
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # cell list: pairs x grid, floor anchors, ceiling, repro
    jobs = []
    for name, (ra, rb) in PAIRS.items():
        for t in GRID:
            jobs.append((f"{name}@{t}", ra, t, rb, t))
    for (t1, t2) in FLOOR_ANCHORS:
        jobs.append((f"floor:{t1}-{t2}", FLOOR_RUN, t1, FLOOR_RUN, t2))
    jobs.append((f"ceiling@{CEILING[2]}", CEILING[0], CEILING[2], CEILING[1], CEILING[2]))
    jobs.append(("repro:P1@16000", PAIRS["P1"][0], 16000, PAIRS["P1"][1], 16000))

    for key, ra, ta, rb, tb in jobs:
        if key in cells:
            continue
        print(f"[{key}] {ra}@step{ta} vs {rb}@step{tb} ...", flush=True)
        try:
            cells[key] = run_cell(ra, ta, rb, tb, tokens, device)
            dtr = cells[key]["d_theta_rel"]
            dtr_s = f"{dtr:.6g}" if dtr is not None else \
                f"n/a ({cells[key].get('d_theta_reason')})"
            print(f"  sym_kl={cells[key]['sym_kl']:.6g}  "
                  f"d_theta={dtr_s}  "
                  f"disagree={cells[key]['disagree_rate']:.4f}", flush=True)
        except Exception as e:
            cells[key] = {"void": True, "error": repr(e)}
            print(f"  VOID: {e!r}", flush=True)
        save()

    # pre-declared reproducibility verdict
    if "P1@16000" in cells and "repro:P1@16000" in cells and \
            not cells["P1@16000"].get("void") and not cells["repro:P1@16000"].get("void"):
        a, b = cells["P1@16000"]["sym_kl"], cells["repro:P1@16000"]["sym_kl"]
        results["repro_rel_err"] = abs(a - b) / max(abs(a), 1e-300)
        results["repro_pass"] = results["repro_rel_err"] <= 1e-6
    save()
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
