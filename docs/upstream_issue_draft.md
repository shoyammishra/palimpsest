# DRAFT — upstream report to EleutherAI (NOT YET FILED)

Status: draft only. Filing is an outward-facing action and needs user sign-off (D-015).
Suggested venue: HF discussion on `EleutherAI/pythia-160m-weight-seed1` (cross-link
`-weight-seed2`), or the EleutherAI/pythia GitHub issues. As of 2026-07-21 no existing
public report of this defect was found (Opus survey, D-015 inputs).

Numbers below trace to: `results/raw/e009_ckpt_audit_2026-07-20.json`,
`results/raw/e009_ckpt_audit_j1_2026-07-21.json` (harness
`src/e009_audit_checkpoints.py`).

---

**Title:** `pythia-160m-weight-seed{1,2}` step≥1 checkpoints appear inconsistent with
their own `step0` initialization

We are using the decoupled PolyPythias 160M runs and observed that the `weight-seed`
checkpoints do not appear to continue from their published `step0` inits.

At `step0`, `data-seed1`, `weight-seed1`, and `weight-seed2` carry three distinct
initializations (pairwise relative L2 ≈ 1.2475). But at `step1`, all four decoupled
repos (`data-seed1/2`, `weight-seed1/2`) are numerically identical across all 148
parameter tensors (differences only in signed-zero bits), coinciding with the shared
data-seed initialization. `weight-seed1` vs `weight-seed2` then diverges from
≈4.69e-5 at `step4` to ≈1.38e-2 at `step256`, so the two runs are genuinely different
downstream (not duplicate uploads), yet neither descends from its own `step0`:
`d_θ(weight-seed1@step0, weight-seed1@stepN) / d_θ(weight-seed1@step0, data-seed1@step0)`
= 1.000 / 1.000 / 0.9999 / 1.000 / 1.001 at N = 1/4/16/64/256 (same to 4 decimals for
`weight-seed2`). Two runs from independent random inits cannot converge four orders of
magnitude in four optimizer steps, which is what a genuine
`weight-seed1 step0 → step4` continuation would require.

Method notes: comparisons are numerical (`torch.equal` / relative L2 over the 148
trainable tensors; buffers such as `masked_bias` excluded), not byte hashes; the step1
files carry distinct SHA-256s, so this is not local cache duplication.

For contrast, the standard seed runs audit clean: `pythia-160m-seed{1,2}` each stay at
their own `step0` init (continuity ratio ≈ 5.6e-5 at `step4`) and differ from each
other at every revision — the issue appears confined to the `weight-seed` family.
One additional minor observation: in both `pythia-160m-seed{1,2}`, the `step1`
revision is numerically identical to `step0` (an init copy rather than a
post-one-step checkpoint).

Our reading is that the `weight-seed` training runs may have started from a
shared/default initialization rather than the per-seed init recorded at `step0`
(reminiscent of the documented Pythia 6.9B/12B init-config issue), with the seed still
driving the dataloader shuffle — which would also explain why the two `weight-seed`
runs diverge from each other despite the model cards describing their data order as
fixed. But we cannot determine the upstream cause from the artifacts alone. Could you
confirm how the `weight-seed` runs' initialization was configured, and whether the
`step0` branches match the inits actually used? Happy to share the full per-step audit
JSONs.
