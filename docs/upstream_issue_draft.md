# Upstream report to EleutherAI

Status: **FILED 2026-07-21** — https://github.com/EleutherAI/pythia/issues/203
(authorized by user; D-015 outward-facing action).
Venue: primary = GitHub issue on `EleutherAI/pythia` (posted as `shoyammishra`);
secondary = short cross-link discussion on `EleutherAI/pythia-160m-weight-seed1`
(mentioning `-weight-seed2`) so downloaders of that artifact see the pointer —
**NOT yet posted**: the local Hugging Face token is invalid (401 on whoami);
needs `hf auth login --force`, then post the text at the bottom of this file.

Prior art: GitHub issue [#198](https://github.com/EleutherAI/pythia/issues/198)
("Weight inconsistencies", 2026-05-18, open) reports that steps 0 and 2 across all
models appear to share identical weights. That is **related but distinct** from the
report below — #198 concerns early-step checkpoints duplicating the init generally,
whereas the main claim here is that the `weight-seed` runs do not descend from their
published `step0` at *any* audited step. Our secondary observation (`step1` ≡ `step0`
in the standard seed runs) is consistent with #198 and is framed as confirming it with
numbers, not as a new discovery.

Numbers below trace to: `results/raw/e009_ckpt_audit_2026-07-20.json`,
`results/raw/e009_ckpt_audit_j1_2026-07-21.json` (harness
`src/e009_audit_checkpoints.py`).

---

## GitHub issue text (primary)

**Title:** `pythia-160m-weight-seed{1,2}` step≥1 checkpoints appear inconsistent with
their own `step0` initialization

We are using the decoupled PolyPythias 160M runs and observed that the `weight-seed`
checkpoints do not appear to continue from their published `step0` inits. Related to
but distinct from #198 (see the note at the end).

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

Our reading is that the `weight-seed` training runs may have started from a
shared/default initialization rather than the per-seed init recorded at `step0`
(reminiscent of the documented Pythia 6.9B/12B init-config issue), with the seed still
driving the dataloader shuffle — which would also explain why the two `weight-seed`
runs diverge from each other despite the model cards describing their data order as
fixed. But we cannot determine the upstream cause from the artifacts alone. Could you
confirm how the `weight-seed` runs' initialization was configured, and whether the
`step0` branches match the inits actually used? Happy to share the full per-step audit
JSONs.

**Relation to #198.** #198 reports steps 0 and 2 sharing identical weights across
models. We can confirm the adjacent case with numbers: in both `pythia-160m-seed1` and
`-seed2`, the `step1` revision is numerically identical to `step0` (an init copy rather
than a post-one-step checkpoint). We did not audit `step2`, so we can neither confirm
nor refute that specific step in #198. The `weight-seed` problem above is a separate
and, we think, more consequential one: there the mismatch does not resolve at later
steps, and it makes the `weight-seed` pair unusable as an initialization-only control.

Why this matters to us, for context: we were using the decoupled PolyPythias to
separate the effect of data *order* from the effect of *initialization* on where
training ends up. The `weight-seed` pair was our only candidate for an
initialization-only comparison, and this defect removes it.

---

## HF discussion text (secondary, `pythia-160m-weight-seed1`)

**Title:** step≥1 checkpoints here don't appear to descend from this repo's `step0`

Cross-posting a pointer: we audited this repo and `pythia-160m-weight-seed2` against
their own `step0` branches and the `step1` checkpoints of all four decoupled 160M repos
are numerically identical to the `data-seed` initialization, with
`d_θ(step0, stepN) / d_θ(step0, data-seed1@step0)` ≈ 1.000 at N = 1/4/16/64/256. The two
`weight-seed` runs do differ from each other downstream, so these are not duplicate
uploads — but neither appears to continue from the init published at `step0`.

Details, numbers, and method are in EleutherAI/pythia GitHub issue:
https://github.com/EleutherAI/pythia/issues/203
