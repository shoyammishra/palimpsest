"""E-009 phase 0 — D-013 verification gates on PolyPythias 160M step0 checkpoints.

Gates (pre-registered in docs/experiment_log.md E-009 before any download):
  G-i   step0(data-seed1) == step0(data-seed2)  bitwise  (MUST PASS)
  G-ii  step0(weight-seed1) != step0(weight-seed2)       (MUST PASS)
  G-iii classify step0(data-seedN) vs step0(pythia-160m) (recorded)
  G-iv  optimizer/LR config read from EleutherAI/pythia repo configs (recorded)

Writes results/raw/e009_gates_<date>.json and saves the fetched training config
under results/raw/. Forward-only downloads (~330 MB x 5); no evaluation.
"""

import datetime
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
DATE = datetime.date.today().isoformat()

REPOS = {
    "main": "EleutherAI/pythia-160m",
    "data-seed1": "EleutherAI/pythia-160m-data-seed1",
    "data-seed2": "EleutherAI/pythia-160m-data-seed2",
    "weight-seed1": "EleutherAI/pythia-160m-weight-seed1",
    "weight-seed2": "EleutherAI/pythia-160m-weight-seed2",
}
REVISION = "step0"

CONFIG_URLS = [
    # Pythia suite training config (GPT-NeoX format); PolyPythias reuses it with new seeds.
    "https://raw.githubusercontent.com/EleutherAI/pythia/main/models/160M/pythia-160m.yml",
]
CONFIG_KEYS = [
    "optimizer", "lr", "betas", "eps", "warmup", "lr-decay-style", "lr_decay_style",
    "min_lr", "min-lr", "train-iters", "train_iters", "train_batch_size",
    "train-batch-size", "seq-length", "seq_length", "seed", "weight-decay",
    "weight_decay", "lr-decay-iters", "lr_decay_iters",
]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_step0(repo):
    files = list_repo_files(repo, revision=REVISION)
    if "model.safetensors" in files:
        fname = "model.safetensors"
    elif "pytorch_model.bin" in files:
        fname = "pytorch_model.bin"
    else:
        raise FileNotFoundError(f"{repo}@{REVISION}: no weight file in {files}")
    path = hf_hub_download(repo, fname, revision=REVISION)
    if fname.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    cfg_path = hf_hub_download(repo, "config.json", revision=REVISION)
    cfg = json.loads(Path(cfg_path).read_text())
    dtypes = sorted({str(t.dtype) for t in sd.values()})
    n_params = sum(t.numel() for t in sd.values())
    return {
        "repo": repo, "file": fname, "sha256": sha256(path),
        "n_tensors": len(sd), "n_params": n_params, "dtypes": dtypes,
        "hf_config": cfg, "_sd": sd,
    }


def compare(a, b):
    """Tensor-by-tensor comparison of two state dicts."""
    sa, sb = a["_sd"], b["_sd"]
    keys_a, keys_b = set(sa), set(sb)
    out = {
        "pair": (a["repo"], b["repo"]),
        "same_keys": keys_a == keys_b,
        "only_in_a": sorted(keys_a - keys_b),
        "only_in_b": sorted(keys_b - keys_a),
    }
    common = sorted(keys_a & keys_b)
    n_equal, max_abs, max_rel, mismatched = 0, 0.0, 0.0, []
    for k in common:
        ta, tb = sa[k], sb[k]
        if ta.shape != tb.shape:
            mismatched.append(k)
            continue
        if torch.equal(ta, tb):
            n_equal += 1
            continue
        mismatched.append(k)
        d = (ta.float() - tb.float()).abs()
        max_abs = max(max_abs, d.max().item())
        # rel. to tensor L2 norm: the old per-element clamp_min(1e-12) denominator
        # produced meaningless ~1e10 ratios against near-zero fp16 weights.
        nrm = tb.float().norm().clamp_min(1e-12).item()
        max_rel = max(max_rel, d.max().item() / nrm)
    out.update({
        "n_common": len(common), "n_bitwise_equal": n_equal,
        "n_mismatched": len(mismatched),
        "mismatched_tensors": mismatched[:20],
        "max_abs_diff": max_abs, "max_abs_diff_over_norm": max_rel,
        # params-identical is independent of storage-format keyset differences
        # (buffers present in .bin but not .safetensors); keyset diff stays
        # visible via same_keys / only_in_a / only_in_b above.
        "params_identical": len(common) > 0 and n_equal == len(common),
        "file_sha256_equal": a["sha256"] == b["sha256"],
    })
    return out


def fetch_training_config():
    out = []
    for url in CONFIG_URLS:
        entry = {"url": url}
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode("utf-8")
            fname = RAW / f"e009_training_config_{Path(url).name}"
            fname.write_text(text, encoding="utf-8")
            entry["saved_to"] = str(fname.relative_to(ROOT))
            hits = []
            for line in text.splitlines():
                s = line.strip().strip(",")
                if any(re.match(rf'^"?{re.escape(k)}"?\s*[:=]', s) for k in CONFIG_KEYS):
                    hits.append(s)
            entry["extracted_lines"] = hits
        except Exception as e:  # recorded, not fatal — gate iv is a readout
            entry["error"] = repr(e)
        out.append(entry)
    return out


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for name, repo in REPOS.items():
        print(f"loading {repo}@{REVISION} ...", flush=True)
        loaded[name] = load_step0(repo)
        m = loaded[name]
        print(f"  {m['file']}  tensors={m['n_tensors']}  params={m['n_params']:,}  "
              f"dtypes={m['dtypes']}  sha256={m['sha256'][:16]}...", flush=True)

    results = {
        "experiment": "E-009 phase 0 (D-013 verification gates)",
        "date": DATE,
        "revision": REVISION,
        "torch": torch.__version__,
        "checkpoints": {k: {kk: vv for kk, vv in v.items() if kk != "_sd"}
                        for k, v in loaded.items()},
        "gate_i_data_seeds": compare(loaded["data-seed1"], loaded["data-seed2"]),
        "gate_ii_weight_seeds": compare(loaded["weight-seed1"], loaded["weight-seed2"]),
        "gate_iii_ds1_vs_main": compare(loaded["data-seed1"], loaded["main"]),
        "gate_iii_ds2_vs_main": compare(loaded["data-seed2"], loaded["main"]),
        "gate_iii_ws1_vs_main": compare(loaded["weight-seed1"], loaded["main"]),
        "gate_iv_training_config": fetch_training_config(),
    }
    results["verdict"] = {
        "G_i_pass": results["gate_i_data_seeds"]["params_identical"],
        "G_ii_pass": not results["gate_ii_weight_seeds"]["params_identical"],
        "G_iii_main_in_P1_class": results["gate_iii_ds1_vs_main"]["params_identical"],
    }

    out_path = RAW / f"e009_gates_{DATE}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["verdict"], indent=2))
    print(f"wrote {out_path}")
    if not (results["verdict"]["G_i_pass"] and results["verdict"]["G_ii_pass"]):
        print("GATE FAILURE — phase 1 must not run under this registration (D-013).")
        sys.exit(2)


if __name__ == "__main__":
    main()
