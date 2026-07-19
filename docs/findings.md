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
