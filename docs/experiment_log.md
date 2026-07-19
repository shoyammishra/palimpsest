# Experiment Log

M1 novelty gate passed 2026-07-17 (D-009, findings F-001..F-004) — compute spend unblocked for the M3 cheapest-first ladder. Rule: hypothesis logged BEFORE the run; one variable at a time; results appended, never edited into the hypothesis; instrument checks precede subject measurements.

---

## Entry template (copy for each experiment; fill hypothesis BEFORE running)

### E-XXX (date) — <one-line name>
- **Hypothesis**:
- **Variable changed** (exactly one):
- **Held constant**:
- **Config**: (model, data, optimizer, seeds, commit hash)
- **Degenerate-strategy baseline & expected score**:
- **Result**:
- **Verdict** (confirmed / refuted / inconclusive — and why):
- **Follow-up**:

---

### E-000 (2026-07-19) — Instrument self-checks (must pass before E-001/E-002 count)
- **Hypothesis**: the harness (`src/pilot_defects.py`, numpy-only, manual-backprop tanh MLP, float64) is correct: (1) manual backprop matches central finite differences to rel. error < 1e-6; (2) the measured SGD pair defect matches the *independently computed* leading-order prediction η²(H_a g_b − H_b g_a) (finite-difference HVPs, different code path) to rel. error < 5% at η = 1e-3; (3) K-1 telescoping: Σ_k Δ_k over a multi-swap schedule (branched replay per swap) equals Φ_{H₂}(x₀) − Φ_{H₁}(x₀) to rel. error < 1e-9 — K-1 is an exact identity, so any mismatch is a code bug by definition. *(Amended pre-run: an originally drafted "identical-steps commute" check was removed as vacuous — it compared a computation with itself and would pass under any bug; replaced by check (2), which is independent.)*
- **Variable changed**: none (pure instrument check).
- **Held constant**: everything; fixed seed.
- **Config**: 1-hidden-layer tanh MLP (~300 params), synthetic regression batches, SGD, float64, seed 0.
- **Degenerate-strategy baseline & expected score**: n/a (checks have exact expected values).
- **Result**: PASS after one checker fix. First run FAILED on gradcheck (1.07e-6 vs 1e-6 tol); diagnosis: FD cancellation noise at eps=1e-6 on a |g_i|≈2e-5 component — a checker artifact, not a gradient bug (same components agree to ~3e-8 at eps∈{1e-4,1e-5}; HVP bracket check independently validates gradients). With eps=1e-5: gradcheck 3.05e-8; bracket cross-check rel. err **7.0e-4** (measured pair defect vs independent FD-HVP prediction); K-1 telescoping rel. err **3.4e-17** over 21 swaps (machine precision — the exact identity verified in code, gap norm 1.5e-2).
- **Verdict**: CONFIRMED — instrument cleared. The telescoping check at float-roundoff level is the strongest possible code validation of the K-1 bookkeeping.
- **Follow-up**: none; harness is the base for E-003+ (transport operator pilots).

### E-001 (2026-07-19) — Linearity of transported defects (K-2 instrument validation, §2.2 gate rung 1)
- **Hypothesis**: for plain SGD (T = 64 steps, stable η), the tail responds linearly to the injected swap defect: R(h) := ‖Φ_A(y + h·δ) − Φ_A(y)‖/h is constant within ~5% over h ∈ [1e-3, 1] for late-training swaps, with departures growing with defect size and tail length, consistent with Lemma K-2's quadratic error term. Early-position swaps (longer tails) depart more than late ones.
- **Variable changed** (exactly one): defect scale h. (Secondary sweep, reported separately: swap position p ∈ {early, middle, late}.)
- **Held constant**: model, data multiset, η, T, seed, which pair is swapped.
- **Config**: as E-000; T = 64; commit hash recorded in results file.
- **Degenerate-strategy baseline & expected score**: h below float-noise floor excluded by monitoring absolute differences vs machine epsilon (a "perfectly constant" R(h) at noise-floor h would be a harness artifact, not linearity).
- **Falsifier**: R(h) varies > 20% across the h-range for typical late-training swaps at small η → JVP estimator invalid at pilot scale → D-010 reversal clause (windowed local fallback).
- **Result**: R(h) constant across h ∈ [1e-3, 1] to **0.019% (early, p=4), 0.0026% (middle, p=32), 0.0012% (late, p=60)** — far inside the 5% hypothesis band. Departure ordering matches the tail-length prediction (early > middle > late), though positions also differ in ‖δ‖ (2.0e-3 / 8.4e-4 / 1.2e-3), so that secondary claim is confounded and reported as consistent, not confirmed. Signal 8 orders above the float noise floor (‖moved−base‖ ≥ 1e-6 vs ~1e-14). Training non-trivial: batch loss 0.271 → 0.0227 over T=64. Incidental observation: transported-defect norms are *smaller* than the injected defects (R·1/‖δ‖ ≈ 0.36–0.81) — tails are contractive in this late-training small-model regime; relevant later to the integrable-phase story, not a claim yet.
- **Verdict**: CONFIRMED — linear response holds at pilot scale; the JVP/linear-transport estimator is valid here. Audit note on the suspicious cleanliness: values are not at a metric boundary (ratios ≠ 1 exactly, monotone in h, position-dependent) and the perturbation propagation is exact float64 replay, so near-perfect linearity at these tiny defect norms is the *expected* behavior of a smooth map, not a harness artifact.
- **Follow-up**: rerun at larger η (less contractive regime) and larger T before trusting the estimator beyond this scale — gate stays per-scale.

### E-002 (2026-07-19) — Order-defect η-scaling: SGD vs EMA momentum (first empirical contact for K-4b / F-005)
- **Hypothesis**: the two-step swap defect scales ‖δ^θ(η)‖ ∝ η^p with p ≈ 2.0 for SGD (β = 0) and p ≈ 1.0 for EMA momentum (β = 0.9, warm state), log-log slope over η ∈ [1e-4, 1e-2], fixed batches (a, b), fixed θ and seed. Predicted by Lemma K-4b: δ^θ = ηβ(1−β)(g_a − g_b) + O(η²). Additionally ‖δ^m(η)‖ has slope ≈ 0 (the state remembers order at O(1)).
- **Variable changed** (exactly one): η. Two pre-declared conditions: β ∈ {0, 0.9}.
- **Held constant**: model, θ, batches a/b, seed, momentum warm-start procedure (fixed burn-in steps).
- **Config**: as E-000; EMA momentum m⁺ = βm + (1−β)g, θ⁺ = θ − ηm⁺ (matches theory.md K-4b exactly); slopes fit by least squares on log-log with R² reported.
- **Degenerate-strategy baseline & expected score**: defect magnitudes monitored against the float64 noise floor — a slope fit through noise-dominated values is excluded; the momentum condition additionally checks the K-4b *constant and direction* (ratio and cosine against ηβ(1−β)(g_a − g_b)), not just the exponent. *(Amended pre-run: an "identical batches ⇒ δ=0" inline check was dropped as vacuous — with a = b both orders are the same computation by construction.)*
- **Falsifier**: momentum slope ≈ 2 (no first-order term) → K-4b wrong or regime inapplicable → STOP and understand before H4 proceeds (surprising-result rule).
- **Caveats declared up front**: two-step defect only (no tails — E-001 covers tails); tiny scale; not a "positive transport result," so the off-NTK constraint (design.md §5.2) does not bind here.
- **Result**: over η ∈ [1e-4, 1e-2]: SGD (β=0) slope **1.9998** (R² = 0.99999998); momentum (β=0.9) slope **1.0022** (R² = 0.999998); state-block slope **−0.020 ≈ 0** (δ^m ≈ 5.4e-3, O(1) in η). Constant-and-direction check at η=1e-4: measured δ^θ vs predicted ηβ(1−β)(g_a−g_b): **ratio 1.000117, cosine 0.99999999**. All magnitudes ≥ 4.9e-6, far above float noise.
- **Verdict**: CONFIRMED — K-4b/F-005 verified quantitatively (exponent, constant, and direction), not just qualitatively. The p≈2 vs p≈1 split is exactly the derived SGD/momentum order gap. Audit note: slopes are near-exact because the measured object is a two-step float64 computation whose leading term is an algebraic identity; the small deviations (1.0022, ratio 1.000117) are the expected O(η) subleading corrections, and the values are cross-validated against an analytically derived constant computed through a different code path.
- **Follow-up**: E-003 candidate: same sweep under Adam-style normalization (K-4d, sketch-level) — does the Θ(η) law survive the nonlinear state readout? Then first 𝓚-aware transport attempt (D-011 operator) vs degenerate baselines.

### E-003 (2026-07-19) — First 𝓚-aware transport: K-3.1 reference operator vs degenerate baselines
- **Hypothesis**: the D-011 operator T(θ(H₁)) = θ(H₁) + Σ_{(i,j)∈Inv(π)} J̄_ij δ̄_ij (single-trajectory K-3.1 reference attribution; 𝓘₁ = {π, H₁ checkpoints}; J̄δ̄ implemented as replay transport along H₁'s own trajectory — one tail replay per inversion, first-order-equal to the JVP and validated by E-001 linearity) closes most of the functional gap: r := d_f(T(θ₁), θ₂) / d_f(θ₁, θ₂) ≤ 0.2 at mild reordering (w=4 block shuffle); r is monotone non-decreasing with severity (w=4 → w=16 → full permutation, since the K-3.1 error budget grows with K and evaluation-point drift); and T beats all degenerate baselines at every severity. Secondary diagnostic: the parameter-space residual ‖T(θ₁)−θ₂‖/‖θ₁−θ₂‖ tracks r (localizes any failure to attribution error vs metric artifacts).
- **Variable changed** (exactly one): the operator applied to θ(H₁) — {K-3.1 operator, do-nothing, checkpoint-average, light fine-tune}. Secondary pre-declared sweep: permutation severity w ∈ {4, 16, full}.
- **Held constant**: model, θ₀, data multiset, η = 0.05, T = 64, seeds, eval inputs, functional metric.
- **Config**: harness as E-000 (~241-param tanh MLP, float64, data seed 42); H₁ = identity word, H₂ = π·H₁ with seeded block-shuffle permutations; d_f = RMS prediction gap on 1024 fresh eval inputs (D-006; P-1's Σ_E-style metric, isotropic input distribution); budget counted in gradient-eval-equivalents (per inversion: 4 pair evals + one tail replay) and reported against retraining = T — honestly, per D-011, even if it exceeds retraining at this K.
- **Degenerate-strategy baseline & expected score**: do-nothing is r = 1 by definition (the denominator); checkpoint-average (mean of the 𝓘₁ checkpoints the operator itself uses, plus θ₁) expected r > 1 (mid-training states drag the endpoint backwards); light fine-tune (16 SGD steps on fresh same-distribution batches, blind to θ₂) expected r ≈ 1 — if fine-tune matches the operator, the metric is not path-specific (flag, don't celebrate). Guards: do-nothing d_f must sit ≥ 6 orders above float noise; inline instrument check — for a single adjacent transposition the reference operator coincides with the exact K-1 computation, so T(θ₁) must equal θ₂ to ~1e-10 relative or the code is wrong (abort).
- **Falsifier**: r ≥ 1 at w=4 → the K-3.1 reference attribution is invalid at pilot scale → localize (parameter residual vs functional ratio; per-inversion audit) before any 𝓚 number is trusted; D-010's windowed-local fallback becomes the next rung.
- **Caveats declared up front**: tiny scale; SGD only; 𝓒_perm pairs constructed by us; η = 0.05 is far from the asymptotic regime where the O(KTη³) bound is proven small — this is exactly what the severity sweep probes; a same-data fine-tune baseline partially violates budget clause (a) but is kept as the "you retrained a bit" comparator.
- **Result**:
- **Verdict**:
- **Follow-up**:
