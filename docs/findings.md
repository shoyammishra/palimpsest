# Findings

What we have learned, with evidence. Novelty-review results land here first (M1), experiment findings after.

## F-001 (2026-07-17) — M1.1 kill-scan verdict: REFORMULATE (provisional, pending deep read of two gating papers)

Source: Opus 4.8 kill-scan subagent report, reviewed by principal engineer. Verdict is PROVISIONAL until the two gating papers below are read end-to-end and their deltas logged (agent verified them via abstract/full-text search, but the gate decision should not rest on a secondhand summary).

### Spec-mandated probability estimates (agent's, endorsed provisionally)
- P(core question already answered): **~0.20**
- P(reviewers call it incremental): **~0.65** ← the dangerous number
- P(hidden overlap not yet found): **~0.30** (OpenReview 2026 pools and forward-citation graphs not exhaustively searched)

### The two papers that gate this project (read fully before finalizing M1.3)
1. **Sweeney, "The Geometry of Sequential Learning: Lie-Bracket Prediction of Transfer Order," ICML 2026** (arXiv:2606.24993). Claims the Lie-bracket/holonomy framing of training-order effects in staged fine-tuning, and exploits it to predict better orderings. Occupies our mathematical angle #1 (non-commutativity as holonomy). Does NOT ask reversibility or build a transport operator.
2. **Yu, He, Goyal, Arora, "On the Impossibility of Retrain Equivalence in Machine Unlearning," 2025** (arXiv:2510.16629). Proves two same-data different-order models cannot both be reconciled to retrain equivalence by a *local, path-oblivious* rule; empirical path-dependence of unlearning at 1B–14B scale. Occupies the negative result for the path-oblivious operator class. Our transport operator survives only if explicitly history-aware and functionally targeted — the carve-out must be crisp or reviewers will say "impossibility already shown."

### Also colonizing the framing
- Xu, arXiv:2602.10496 (2026): measures SGD "commutator defect," uses the words "integrable/non-integrable dynamics" (toy modular-arithmetic transformers; no curvature formalism, no transport).
- Sweeney, arXiv:2606.29554 (2026): Adam's moment buffers make batch order a first-order noise source — pre-empts part of the "optimizer memory ⇒ non-conservative" story.
- Goodbrake, "Unnatural Algorithms" (arXiv:2312.04739): sorts optimizers by reparameterization-naturality — must be distinguished from our conservativeness/holonomy taxonomy.
- NTK/lazy regime: path-independence is trivially true there — any positive transport result must be demonstrated OFF the lazy manifold or it's folklore, not a finding.

### Confirmed genuine gaps (the surviving questions)
1. **Positive transport operator with a phase boundary** — does a budget-constrained, history-aware T exist with T(θ(H₁)) ≈_f θ(H₂), and where is the integrable/path-burned boundary? Nobody has built or bounded this. Best framing: "when is Yu et al.'s impossibility escapable by spending a small budget?"
2. **A quantitative invariant linking holonomy to transport cost** — Sweeney predicts which order is better; Xu measures the defect; neither ties an invariant to *minimum compute to transport between endpoints*. Target: "transport cost ≥ f(accumulated holonomy)" — would subsume both as special cases.
3. **Conservative-vs-non-conservative optimizer taxonomy tied to reversibility** — does non-conservativeness (holonomy sense) *predict* which histories are path-burned, distinct from Amari-naturality? "Shuffled SGD approximately integrable, Adam not" would be a crisp, testable, unclaimed dichotomy.

### Useful infrastructure confirmed
PolyPythias (ICLR 2025, arXiv:2503.09543): 45 runs varying seed and data order over fixed Pythia data — ready-made history-variation testbed; describes stability, builds no transport operator. TracIn explicitly *assumes* commutativity — citable as the assumption we stress-test.

### Citation hygiene flags (from agent, re-verify before writing)
SISA (IEEE S&P 2021), certified removal (Guo, ICML 2020), Model Soups (ICML 2022), EWC (PNAS 2017) — cited from memory by the agent, not re-verified this scan.

## F-002 (2026-07-17) — Sweeney (arXiv:2606.24993) does NOT tie its Lie-bracket to transport cost; Q2's delta survives

Source: principal engineer, targeted full-text check of the arXiv HTML (three probes: abstract, full-body semantic search, exhaustive keyword sweep). This resolves the specific kill-check flagged when picking a new core: "does Sweeney's invariant predict anything about the cost of moving between endpoints, or only which order is better?"

**Answer: order only.** Evidence:
- The bracket is used exclusively to (a) predict pairwise transfer direction (which order A→B vs B→A yields lower target loss), (b) rank many-domain curricula via tournament Borda scoring, (c) estimate stakes/confidence of order decisions.
- Keyword sweep over the full text incl. appendices: "transport", "holonomy", "budget", "effort", "reversib-/irreversib-", "undo", "repair" **never appear**. All "cost" usages concern the planner's own runtime (gradient/HVP counts vs. brute-force order trials). "Recover" appears only as "recovered fraction" (task-performance metric).
- Notable correction to F-001's characterization: the paper never uses the word "holonomy" at all. Its quantity is the *local, first-order* Lie-bracket commutator with k-step lookahead — not an accumulated path invariant. The delta to our Q2 (accumulated holonomy → minimum transport cost between endpoints) is therefore *wider* than the kill-scan implied.

**Implication for the core pick**: Q2 (transport-cost ≥ f(accumulated holonomy) invariant) survives its designated kill-check. Sweeney remains a citation and a special case to subsume, not an occupant of the question.

**Caveat**: this was a targeted check via summarizer-assisted fetches against the full text — strong for this narrow question, but it does not discharge D-008's requirement of an end-to-end principal read of both gating papers (full Sweeney delta log + Yu/Arora carve-out check still pending before the gate finalizes).
**Superseded 2026-07-17 (same day):** the end-to-end principal read is now done — see F-003, which confirms and extends this finding.

## F-003 (2026-07-17) — Sweeney (arXiv:2606.24993) end-to-end read: full delta log

Source: principal engineer, complete read of the paper (main body + appendices A–E) from the arXiv HTML full text. Discharges the Sweeney half of D-008's deep-read requirement.

### What the paper actually is
A **prospective order planner**. Core object: the local Lie-bracket vector b_AB = H_B g_A − H_A g_B at θ₀ (second-order BCH leading term of θ_AB − θ_BA for two gradient steps, Lemma 2.1). Directional score σ = ⟨g_E, b_AB⟩ predicts which order (A→B vs B→A) gives lower target loss (Prop 2.3); a drift-matched "Trotter" reference θ_ref = θ₀ − η(g_A+g_B) reduces the error to O(η⁴) on the target-gradient side (Prop 2.5). Scales to N domains via a Borda/row-sum "Lie-Bracket Tournament" (one HVP per source). Validated on SFT/DPO/Pile-domain/diffusion; accuracy decays from 93% (k=1) to 65.3% (k=50) as the θ₀-local bracket loses validity along the trajectory.

### Confirmed deltas (each is a boundary of Sweeney's claims, verified against the full text)
1. **Local, not accumulated.** Every quantity is evaluated at θ₀ (or θ_ref, O(η) away). There is no integrated/accumulated bracket along a trajectory, no path invariant, no curvature-of-connection formalism. Their own long-horizon decay (Table 3) is *evidence that a θ₀-local quantity is insufficient* — they leave the accumulated object unbuilt. Our Q2 invariant (accumulated holonomy) starts exactly where their theory stops.
2. **Prospective, not retrospective.** The planner chooses an order *before* training. Nothing in the paper takes two *completed* histories and relates them — no transport, no reconciliation, no cost of moving between endpoints (confirms F-002 by direct read).
3. **Closest overlap — Appendix E.8 "bracket-control": must cite and delimit.** They take one correction step θ_ctrl = θ_ref − 0.5·sign(σ̂)·η²b_AB and beat the uncorrected shared-drift point in 87.7% of cases (beats both sequential endpoints in only 46.6%). This is a *single infinitesimal control step at planning time in the k=1 two-step setting* — not endpoint-to-endpoint transport between full histories. But it is the paper's one gesture toward "bracket direction as a repair direction," so our related-work section must name it and state the difference precisely, or a reviewer will.
4. **Q3 adjacency — Appendix D.7:** the SGD-derived bracket fails under AdamW (47.7% at k=5, i.e. below chance), and they *name* "an augmented-state commutator on (θ, m, v)" as the correct object, explicitly deferred as future work. This is the seed of our Q3 (optimizer taxonomy) held by someone else as a stated intention: Q3 as a *standalone* core is now riskier (a Sweeney follow-up likely exists in the pipeline), which further supports Q2-as-core with Q3 demoted to an ablation axis.
5. **Prior art we didn't have:** Rukhovich, Podolskiy, Piontkovskaya (arXiv:2501.15556, 2025) introduced the same bracket-projection ⟨g_E, b_AB⟩ as a descriptive local optimality criterion before Sweeney. Add to citation map. Also Sweeney (ICML 2026) "The geometry of updates: Fisher alignment at vocabulary scale" — same author, adjacent geometry.

### Correction to F-001
F-001 described Sweeney as claiming "the Lie-bracket/holonomy framing." The word holonomy never appears; the framing is strictly the local commutator. The mathematical territory Sweeney occupies is narrower than the kill-scan implied; the delta for an accumulated-holonomy → transport-cost invariant is wide and clean.

## F-004 (2026-07-17) — Yu/He/Goyal/Arora (arXiv:2510.16629) end-to-end read: the history-aware carve-out HOLDS

Source: principal engineer, complete read (main body + proof appendix C + ablations H) from the arXiv HTML full text. Discharges the Yu/Arora half of D-008's deep-read requirement.

### What the theorem actually proves (and what it doesn't)
Setting: overparametrized linear regression, two-stage ridge training regularized toward the previous iterate; unlearning = gradient **ascent** on the forget set S_U. Theorem 3.1: the weight difference between the two order-variants evolves as Δθ_t = (I+M_U)^t Δθ₀ with M_U = (2η/k)X_UᵀX_U ⪰ 0, so the *functional* RE-distance between the two models diverges exponentially in unlearning steps t (rate: Rayleigh quotient ρ★ of M_U on the projected difference). Corollary: both models cannot simultaneously reach the retrained target — Retrain Equivalence is ill-posed for this operator class. Empirics: same divergence + recency effect + path-dependent superficial-vs-deep forgetting, Llama/Qwen 1B–14B, GA/NPO/SimNPO.

The operator class killed is precisely: **local** (Def 2.2 — updates depend only on gradients computed on the forget set) **and path-oblivious** (the same rule applied to both histories). The impossibility is that one shared rule cannot serve two histories — the theorem lower-bounds the divergence *between the two unlearned models*, not the error of a history-adapted rule.

### Why our operator class survives (the carve-out, now verified line-by-line)
1. **Explicit scope disclaimer** (§1): "Our work does not discuss the hardness of retrain equivalence for unlearning schemes that (i) use retain-set information, (ii) modify the training process to enable future unlearning, or (iii) rely on certified procedures with stronger assumption of model or data access." History-aware operators with declared side-information are outside the theorem by the authors' own fence.
2. **The impossibility triangle** (§5): path-independence / retrain equivalence / locality — "at most two out of the three." They pursue forgoing RE; forgoing *path-independence* (our direction: an operator that knows and uses the history) is named as the other branch and left entirely unexplored. Our core question is literally "what is the minimum budget at which the third vertex becomes affordable" — their frame, our gap.
3. **Mechanism gift — the off-span invariant.** In their proof, the component (I−P_U)Δθ₀ orthogonal to span(X_U) is *untouched* by any number of local unlearning steps. That is a crisp, provable instance of "path-burned relative to an operator's data access": history differences invisible to the accessible span can never be corrected from within it. Our path-burned definition should generalize exactly this (accessible span → declared side-information/budget), and our transport-cost lower bound should reduce to their divergence result as a special case. This makes Q2's inequality *strictly containing* their theorem a concrete, checkable target.
4. **Divergence mechanism is ascent-specific.** The exponential rate comes from (I+M_U)^t with M_U ⪰ 0 — gradient *ascent* is expansive. A transport operator is not doing ascent; nothing in the proof machinery constrains a contractive, history-conditioned map. (Any positive result must still respect their triangle honestly: declare side-information, show compute ≪ retraining, or reviewers will correctly say "you retrained.")
5. **Their functional metric = our D-006 metric.** RE-distance is defined on predictions over a test set, not parameter distance — our functional-equivalence convention is aligned with the standard this literature already uses.

### Open questions they pose that we partially answer
§5 asks: "is there any way to distinguish path-induced behavior from algorithm-induced behavior?" A holonomy→transport-cost invariant is a quantitative answer candidate — cite this sentence as the hook.

### Net verdict input
Both gating papers read end-to-end. Sweeney: order prediction only; local bracket; no accumulated invariant, no transport, no cost (F-002/F-003). Yu/Arora: impossibility confined to local+path-oblivious; history-aware budgeted transport explicitly out of scope and named-but-unexplored; proof yields a special case our invariant should subsume. **Q2 (transport-cost ≥ f(accumulated holonomy)) survives both kill-checks with a clean, citable delta.** Gate finalization logged as D-009.

## F-005 (2026-07-19) — Derived: momentum makes training-order defects first-order in η (SGD: second-order); formal seed of H4

Source: principal engineer, theory work (docs/theory.md, Lemma K-4b — PROVED by direct computation, no experiment).

For EMA momentum (m⁺ = βm + (1−β)g, θ⁺ = θ − ηm⁺), the exact leading-order swap defect of two updates a, b is δ^θ = ηβ(1−β)(g_a − g_b) + O(η²) and δ^m = (1−β)²(g_a − g_b) + O(η). At β = 0 the η-term vanishes and plain SGD's classical Θ(η²) bracket order is recovered; for any β > 0 the order defect is **Θ(η) — one full order larger**. The optimizer state remembers order at O(1), decays geometrically (factor β/step), and leaks into θ at O(η)/step. The argument extends to any C¹ state readout (Adam's normalization; sketch level).

**Why it matters**: (1) this *derives* what Sweeney arXiv:2606.29554 reported empirically ("Adam's moment buffers make batch order a first-order noise source") — our framework produces their observation as a two-line corollary, evidence the formalism carves reality at a joint; (2) it is the formal seed of H4 (optimizer stratification): per inversion, momentum-class optimizers accumulate holonomy one order of η more than SGD — a quantitative, testable gap prediction for M3/M4. Caveat: leading-order statement on the regularity tube; Adam constant-tracking deferred (theory.md K-4d).

## F-006 (2026-07-19) — Proved: path-burned floor for fixed-subspace access classes (target inequality exact in that case); 𝓚 measurable from a single run

Source: principal engineer, theory work (docs/theory.md, Proposition P-1 — PROVED; Corollary K-3.1 — sketch-level).

1. **P-1 (proved, two lines, linear predictors)**: for any operator class whose total displacement lies in a fixed subspace S (any compute), TransportCost(H₁→H₂; 𝒯_S, ε) = ∞ for every ε below dist_{Σ_E}(Σ_k Δ_k, S) — the functionally irreducible residual. This (a) makes the v0.2 target inequality *exact* for fixed-subspace classes, (b) fixes the definition of the path-burned mass as a projection **distance** (not the naive orthogonal component — reachable motion can partially compensate off-span functional effects when Σ_E mixes subspaces), and (c) contains Yu/Arora's off-span invariance as the instance S = rowspace(X_U) (Lemma K-5a, proved). The frontier moves to state-dependent (conic) reachable sets.
2. **Corollary K-3.1 (sketch)**: the full inversion-indexed attribution of the endpoint gap is computable from **H₁'s trajectory alone** (checkpoints + two map evaluations per pair + one JVP chain) up to O(KTη³) error — 𝓚 is a single-run instrument, not a thought experiment requiring intermediate retraining. This sets the M3.1 pilot design: one instrumented run + branched-replay spot checks as the validation gate.

Caveats travel with both: K-2/K-3 error constants are geometric in tail length (absolute-error bounds; relative error uncontrolled when the tail nearly annihilates a defect), so the branched-replay validation gate of design.md §2.2 is binding before any 𝓚 number is trusted.

## F-007 (2026-07-19) — First empirical contact: K-4b confirmed quantitatively; linear transport valid at pilot scale; K-1 verified in code to machine precision

Source: E-000/E-001/E-002 (docs/experiment_log.md), tiny-scale pilot (~241-param tanh MLP, T=64, float64, seeded; harness src/pilot_defects.py; raw results results/raw/pilot_defects_2026-07-19.json).

1. **K-4b holds with exponent, constant, and direction** (E-002): swap-defect scaling slope 1.9998 for SGD vs 1.0022 for EMA momentum (β=0.9), R² ≈ 1 both; measured momentum defect matches the derived ηβ(1−β)(g_a−g_b) to ratio 1.000117 and cosine 0.99999999. The theory's sharpest derived prediction (F-005) survives first contact intact — and the optimizer-state block is O(1) in η (slope −0.02), as derived.
2. **Linear response of tails** (E-001): transported defects respond linearly to defect scale within 0.02% over three orders of magnitude of h, at all three swap positions — the JVP/linear-transport estimator is valid at this scale. Per-scale gate: revalidate at larger η/T before trusting elsewhere.
3. **K-1 telescoping verified in code at 3.4e-17 relative error** over a 21-swap random permutation (E-000) — the instrument's bookkeeping matches the exact identity at float roundoff.
4. Incidental: tails were *contractive* here (transported norm 0.36–0.81× the defect) — a first hint of the integrable phase in late-training small-model regimes; observation only, no claim.

Caveats: tiny scale, SGD/EMA only (Adam normalization untested — K-4d), two-step defects for E-002, and nothing here is a transport/generalization result (D-006 headline metrics not yet in play).

## F-008 (2026-07-19) — First 𝓚-aware transport works at pilot scale: possible but not cheap; functional/parameter metrics dissociate at high inversion count

Source: E-003 (docs/experiment_log.md), harness src/pilot_transport.py, raw results/raw/pilot_transport_2026-07-19.json. First experiment where the D-006 headline metric (functional gap) is in play.

1. **The D-011 operator transports** (𝓘₁ = {π, H₁ checkpoints}; T(θ₁) = θ₁ + Σ J̄δ̄ via Cor K-3.1, replay-transport implementation): functional gap ratio r = d_f(T(θ₁), θ₂)/d_f(θ₁, θ₂) = **0.145 at mild reordering** (44 inversions), 0.265 at moderate (218), 0.608 at full permutation (982) — monotone in severity as the K-3.1 error budget predicts, and better than every degenerate baseline at every severity (checkpoint-average: 4.8–112×; blind fine-tune: 1.04–29×). The theory's constructive operator survives first contact.
2. **Possible ≠ cheap — first phase-boundary data point**: budget was 25× / 143× / 713× retraining (K = O(T²) tail replays is the driver). Per D-011 this is reported as a finding, not hidden: at this scale, 𝓘₁-transport beats retraining on *nothing yet*; batched JVP (O(T) chains) + inversion sampling are the declared compression paths before any cost claim.
3. **Metric dissociation at high K**: at full permutation the correction *overshoots in parameter space* (residual 1.63 > do-nothing's 1.0) while still cutting the functional gap 40% — large attribution errors live in functionally flat directions. Direct evidence for D-006 (parameter distance is the wrong target) and a localization of the K-3.1 validity edge to K ∈ (218, 982] at η = 0.05, T = 64.
4. **Heavy cancellation**: ‖ΣΔ̄‖/Σ‖Δ̄‖ ≈ 0.12–0.22 — the holonomy mass 𝓚₁ overstates the net gap 5–8×, confirming the design.md §2.1 warning that 𝓚₁ is machinery, never a headline number.

Caveats: tiny scale, SGD only, 𝓒_perm pairs we constructed, single seed per severity, and the fine-tune baseline's poor showing partly reflects the small gap magnitudes at mild severity (its blind displacement dwarfs them) — not evidence fine-tuning is useless, evidence it is not path-targeted.

## F-009 (2026-07-19) — The Θ(η) order-defect law survives Adam's nonlinear state readout; the v-channel alone suffices (K-4d confirmed at pilot scale)

Source: E-004 (docs/experiment_log.md), harness src/pilot_adam.py, raw results/raw/pilot_adam_2026-07-19.json. Index-free smooth-normalized Adam (K-4a variant, no bias correction), warm state, two-step swap defects, η ∈ [1e-5, 1e-3].

1. **Full Adam-class (β₁ = 0.9, β₂ = 0.9): slope 1.0014** (R² ≈ 1), with the frozen-θ leading-order predictor (independent code path: state-only recursion at frozen θ) matching to ratio 1.000046 and cosine 0.9999999. Normalization performs no special cancellation — K-4d's headline claim holds.
2. **The sharp result — normalization-only (β₁ = 0): slope ≈ 1.01 at small η** (frozen-θ check ratio 0.9993, cosine 0.9997). Under a *linear* readout, β₁ = 0 gives slope 2 (E-002/K-4b: the η-term carries factor β(1−β)). So the first-order defect here is produced *entirely by the second-moment channel passing through the nonlinear readout* — direct evidence for K-4d's mechanism ("any C¹ readout of O(1)-differing state arguments leaks order at Θ(η)"), not just its conclusion. Derived leading term (now in theory.md K-4d): δ^θ = η·β₂(1−β₂)[N_v(g_a,v̄)(v−g_b²) + N_v(g_b,v̄)(g_a²−v)] + O(η²).
3. **Both state channels remember order at Θ(1)** (slopes ≈ 0 for δ^m and δ^v), and K-4b's δ^m constant (1−β₁)²‖g_a−g_b‖ was verified incidentally across the two conditions (0.00541 vs 0.01 × 0.541).
4. **H4 chain now complete at pilot scale**: SGD Θ(η²) → EMA momentum Θ(η) → Adam-class Θ(η), with constants and directions checked at each rung. Per inversion, every stateful optimizer tested accumulates holonomy one order of η more than SGD — the quantitative seed of the optimizer-stratification hypothesis is now empirical, not just derived.

Caveats: two-step defects (no tails), tiny scale, single batch pair, index-free variant (canonical Adam's bias correction differs early in training), and condition (ii)'s full-range slope (1.20) is contaminated by unsuppressed O(η²) terms at the top of the range — the Θ(η) claim rests on the pre-declared low-η refit plus the constant-and-direction check, both passing.

## F-010 (2026-07-19) — Transport budget scaling cut from O(KT) to O(K+T) with zero measurable quality loss; pair defects, not transport, are now the cost driver

Source: E-005 (docs/experiment_log.md), harness src/pilot_transport_batched.py, raw results/raw/pilot_transport_batched_2026-07-19.json; accounting rules per D-012.

1. **The batched forward-JVP chain (design.md §2.2's declared cheap estimator) replaces per-inversion tail replay at no measurable cost to transport quality**: functional gap ratios match E-003's replay operator to Δr ≤ 0.0005 at every severity (0.1457 / 0.2655 / 0.6081 vs 0.1452 / 0.2651 / 0.6079), correction vectors agree to cosine ≥ 0.99999. The §2.2 validation gate passed per-sample (median JVP-vs-replay error ~0.1% per inversion, 10 sampled per severity).
2. **Budget compression 5.3× / 9.2× / 11.3×** (severities w4 / w16 / full; E-003 counting rule), and the cost *structure* changed: ~2.4 grad evals per inversion for pair defects + a fixed 124-eval chain independent of K. Under D-012 grad-eval-equivalents the operator now costs **3.8× / 10.4× / 38.6× retraining** (was 25× / 143× / 713×).
3. **Honest phase-boundary update, as pre-declared**: still above retraining at every severity — no cheap-transport claim. But the driver moved: the JVP chain is now negligible; the K pair-defect evaluations dominate. **Inversion sampling is the sole remaining lever**, and the heavy cancellation measured in E-003 (‖ΣΔ̄‖/Σ‖Δ̄‖ ≈ 0.12–0.22) says naive uniform sampling will be high-variance — the sampling design is now the scientifically load-bearing step for the cheap-transport question.
4. **The K-3.1 validity edge is attribution-not-transport**: the full-severity parameter overshoot (residual 1.631) reproduces bit-faithfully under the completely different transport mechanism — the error lives in the reference attribution at high K, not in how defects are pushed through tails.
5. Instrument note: the chain uses an exact forward-over-reverse HVP (new in this harness, cross-checked against the validated FD-HVP to 1.8e-11), which makes the batched-chain-vs-summed-transports check a true code-bug detector — it sat at float precision (5e-16).

Caveats: same tiny-scale / SGD / constructed-𝓒_perm limits as E-003; compression measured in grad-eval counts under declared rules (not wall-clock); retraining comparator = T = 64 evals at this scale.

## F-011 (2026-07-19) — Unbiased inversion sampling is closed as the cheap-transport lever at pilot scale; cancellation sets an exact variance law, and coherence rises with severity (α ≈ 0.19)

Source: E-006 (docs/experiment_log.md), harness src/pilot_sampling.py, raw results/raw/pilot_sampling_2026-07-19.json. With the chain fixed-cost (F-010), sampling M of the K pair defects was the last naive budget lever; this finding is the pre-registered negative that closes it, plus the quantities that say where it reopens.

1. **The sampling error is governed by an exact, pre-registerable law**: for the with-replacement importance estimator, RMS relative error = A_p/√M with A_p² = (Σ‖Δ̄_k‖²/p_k)/‖ΣΔ̄‖² − 1 — verified to ~1% in median across 84 cells (range [0.93, 1.10]). Heavy cancellation (‖ΣΔ̄‖/Σ‖Δ̄‖ = 0.22/0.18/0.12, matching E-003's replay values to 3 decimals) makes A large: even the variance-optimal oracle weighting has A = 4.5/5.5/8.3. Nothing about sampling difficulty here is mysterious — it is the cancellation ratio, exactly.
2. **Budget-honest sampling loses to full enumeration everywhere at this scale**: quality-gate sample sizes M* (median r ≤ r_full + 0.05, fitted functional-sensitivity mapping) are 333× / 36× / 1.4× K for uniform and 359× / 31× / 1.1× K for the free ‖g_a‖‖g_b‖ proxy. At mild severity, sampled transport at every tested M is *worse than doing nothing* (median r up to 4.4) — an under-sampled unbiased correction is noise added to a small gap. The pre-registered falsifier (a budget-honest scheme beating enumeration) did not fire.
3. **But the required sampling fraction collapses as severity grows**: M*/K falls ~2 orders across w4 → full, A_oracle/√K = 0.67 → 0.26, and the fitted coherence exponent is c ~ K^(−0.19) — much slower cancellation growth than the √K of an incoherent random walk. The oracle diagnostic already dips under enumeration at full severity (M*/K = 0.81). Sampling is not dead as a scaling strategy; it is dead *naive and at pilot scale*. α is now a named quantity to re-measure at larger T.
4. **Bias beats variance under heavy cancellation**: deterministic top-M truncation by ‖Δ̄‖ (biased, unweighted, gate-free comparator) outperformed unbiased sampling at almost every M (e.g. w4 top-32: r = 0.45 vs sampled 1.4–2.6) while remaining non-monotone in M. This redirects the budget question toward structured estimators (truncate-the-top + control-variate/stratify the remainder) — logged as an M5 candidate behind the validity-edge probe.
5. **Rank correlation is not variance reduction**: the free proxy tracks true norms at Spearman 0.6–0.77 yet its A is no better than uniform (sometimes worse) — importance weighting punishes magnitude miscalibration through the 1/p_k factor. A proxy must be magnitude-calibrated before it buys anything.

Caveats: with-replacement i.i.d. estimators only (structured designs untested); M* values beyond the grid are extrapolations under the fitted linear noise→d_f mapping (the law itself is measured, the gate crossing is extrapolated); oracle/pair-oracle schemes are diagnostics, not operators; tiny scale, SGD, constructed 𝓒_perm, single seed per severity as before.
