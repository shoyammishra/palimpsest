# Design — Formal Framework (draft v0.3, M2 in progress)

Status: DRAFT v0.3. Core question fixed (D-009). The discretization of 𝓚 is now fixed (D-010): the transported-defect (discrete non-abelian-Stokes) form of §2.1, exact by construction. Remaining M2 items: proofs of Lemmas K-2..K-5 (§2.5), the minimal side-information class 𝓘 for first pilots, and the homotopy generalization to non-permutation history pairs. Nothing in this file licenses running experiments beyond the M3 roadmap discipline.

Supersedes v0.2. Changes: §2 rewritten — 𝓚 defined via exact swap defects + telescoping identity (former candidates (a)/(b) absorbed as derived scalars of (c), see D-010); TransportCost sharpened with a reachable-span projection; target inequality v0.2 stated; estimator/instrument section added (§2.2); lemma ledger added (§2.5).

Supersedes v0.1 (pre-literature-review): core question locked to the holonomy→transport-cost invariant (Q2), Q1 as machinery, Q3 as ablation axis; positioning constraints from the two gating papers added.

## 0. Core question (fixed by D-009)

> **Is there a quantitative invariant linking the non-commutativity accumulated along a training trajectory to the minimum budget required to transport between the endpoints of two histories?**
>
> Target form: **TransportCost(H₁ → H₂) ≥ f(accumulated holonomy along H₁, H₂)**, with the integrable/path-burned phase boundary appearing as the region where f stays small vs. diverges.

Containment targets (what the invariant must subsume to defeat "incremental"):
- **Sweeney (arXiv:2606.24993)**: local bracket σ = ⟨g_E, H_B g_A − H_A g_B⟩ predicts which order is better — must fall out as the leading local term of our accumulated quantity (k=1, θ₀-local limit).
- **Xu (arXiv:2602.10496)**: measured SGD commutator defect — a pointwise sample of the same integrand.
- **Yu/Arora (arXiv:2510.16629)**: exponential RE-divergence under local path-oblivious unlearning — should be recoverable as the zero-budget / zero-side-information limit of the cost bound (their off-span invariant (I−P_U)Δθ₀ is the prototype of "path-burned relative to an access class").

## 1. Objects

- **History space 𝓗**: a training history H is the full optimization configuration — data ordering, curriculum, optimizer + hyperparameters, LR schedule, augmentation schedule, replay schedule, staged phases (pretrain → SFT → RLHF). Data *content* (the multiset of examples) may be held fixed or varied; the primary regime fixes the multiset and varies everything else.
- **Training map Φ**: Φ(θ₀, H, ω) = θ_T, with ω the residual stochasticity (e.g., dropout/augmentation seeds) not folded into H. The trajectory is τ(H) = (θ₀, …, θ_T).
- **Functional equivalence**: θ ≈_f θ′ iff the induced functions match in distribution on evaluation measures of interest (generalization behavior), NOT parameter-space closeness. (Aligned with Yu/Arora's RE-distance, which is also prediction-space — cite as the community-standard metric.)

## 2. Central definitions (v0.3 — 𝓚 discretization fixed by D-010)

### 2.0 Histories as words of update maps

- **State space X**: X = Θ for stateless optimizers. For stateful optimizers, the **augmented state** — e.g., Adam: X = Θ × M with x = (θ, m, v). All definitions below live on X; the θ-block is read out at the end. This lift is our answer to the augmented-state commutator Sweeney App. D.7 names but defers as future work (F-003 §4).
- **History as a word**: H = (u_1, …, u_T), each u_t: X → X an update map, u_t(x) = x + η_t f_t(x) with f_t the update field (batch, optimizer rule, and hyperparameters at step t folded in). Φ_H = u_T ∘ ⋯ ∘ u_1; θ(H) = θ-block of Φ_H(x₀).
- **Primary class 𝓒_perm**: H₂ = π·H₁ — same multiset of update maps, reordered. Binding caveat: reordering *data* equals reordering *maps* only where the swapped positions carry identical hyperparameters (e.g., constant LR across the swapped window). First theorems assume step-exchangeable schedules; non-permutation pairs (different optimizer / LR schedule — needed for H4) require the homotopy generalization deferred to §2.5.

### 2.1 Accumulated holonomy 𝓚 — transported-defect form (D-010)

**Exact swap defect.** For update maps u, v and a point x ∈ X:

> δ_{u,v}(x) := (v∘u)(x) − (u∘v)(x)   (apply u then v, minus apply v then u)

No expansion — this is the actual two-step order defect of the real training maps. In the smooth small-η limit, δ_{u,v}(x) = η² [f_u, f_v](x) + O(η³); for SGD fields f = −∇L this leading term is the (H_B g_A − H_A g_B) bracket of Sweeney/Rukhovich, and Xu's measured "commutator defect" is ‖δ‖ sampled at points along training.

**Telescoping identity (Lemma K-1 — exact, proved).** Fix an adjacent-transposition sequence carrying word H₁ to word H₂ through intermediates W⁰ = H₁, …, W^K = H₂. If W^{k−1} = A ∘ (swapped pair) ∘ B and W^k differs only by that swap, set x_k = Φ_B(x₀), y_k = value after the pair under W^{k−1}'s order, δ_k = (pair under W^k's order)(x_k) − (pair under W^{k−1}'s order)(x_k), and define the **transported defect**

> Δ_k := Φ_A(y_k + δ_k) − Φ_A(y_k).

Then, purely by telescoping (three-line proof, no approximation of any kind):

> **Φ_{H₂}(x₀) − Φ_{H₁}(x₀) = Σ_{k=1..K} Δ_k.**

To first order in δ_k, Δ_k ≈ J_{Φ_A}(y_k)·δ_k — the **J·δ factorization**: local non-commutativity (δ_k) times downstream transport/amplification (the tail Jacobian J = Π_t (I + η_t J_{f_t}), structurally the (I+M_U)^t of Yu/Arora).

**Canonical indexing.** Use the Kendall-minimal (bubble-sort) transposition sequence: K = number of inversions of π, and each swap event corresponds to exactly one **inversion pair** (i, j) — a pair of updates whose relative order differs between the two histories. So 𝓚's attribution is indexed by order-reversed update pairs; Sweeney's pairwise score is the single-inversion two-step case.

**The invariant and its derived scalars.**
- **𝓚⃗(H₁→H₂)** := (Δ_1, …, Δ_K) — the attribution of the endpoint gap to inversion events, each factored as J·δ.
- **Functional holonomy 𝓚_E** := Σ_k ⟨g_E(θ(H₂)), Δ_k⟩ — signed, cancellation-aware, sums to the first-order functional gap. Headline-facing (D-006).
- **Holonomy mass 𝓚₁** := Σ_k ‖Δ_k‖ — the norm envelope; 𝓚₁ ≥ ‖Σ_k Δ_k‖ with equality iff no cancellation. Internal machinery, never a headline number.

**Resolution of former candidates (a)–(c)**: (c) is the definition — the ordered-exponential mismatch expanded *exactly* by Lemma K-1 instead of by BCH truncation; (b) survives as the functional projection 𝓚_E (now transported, not θ₀-local); (a) survives as the mass 𝓚₁. Nothing was discarded; the candidates were levels of one object. See D-010.

**Well-definedness off the quadratic regime.** The definition uses only the actual update maps: δ_k and Δ_k are finite compositions and differences of them. No Taylor/BCH truncation, no smoothness, no invertibility, and no quadratic approximation appear anywhere in the *definition* — so 𝓚 is well-defined wherever training itself is defined, including deep in the non-lazy regime. The quadratic regime enters only in (i) *interpreting* δ_k as a Lie bracket (small-η limit, used for the Sweeney containment) and (ii) the JVP *estimator* (§2.2), whose error is controlled segment-locally (Lemma K-2), not by global quadraticity. This discharges the well-definedness half of the first M2 deliverable by construction; the analytic burden moves to estimator error, where it belongs.

**Gauge (schedule) dependence — stated honestly.** The total Σ_k Δ_k is schedule-invariant (it equals the endpoint gap). The attribution {Δ_k} depends on the transposition schedule at second order in δ; at first order it is schedule-independent for reduced (one-swap-per-inversion) schedules (Lemma K-3, to prove). We fix the canonical bubble-sort schedule; wherever an attribution-based claim is made (H2/H5), sensitivity across schedules is reported with it.

**The circularity objection (pre-empted).** "Σ_k Δ_k is exactly the endpoint gap, so the invariant is just the gap — circular." Answer: the raw endpoint gap predicts nothing about cost (a large gap can be cheap to close — fine-tuning does it daily). The content of 𝓚 is (1) the **decomposition** — per-inversion, per-time, per-layer attribution, with J·δ separating *where* non-commutativity accrued from *how much it was amplified downstream* (this is exactly what H2/H4/H5 predict against); (2) the **mass** 𝓚₁ and its cancellation gap 𝓚₁ − ‖ΣΔ_k‖; (3) the **projected component** Π_⊥𝓚 (§2.4), which — not the raw gap — is what enters the cost lower bound.

**Containments (the §0 targets, now structural rather than analogical):**
- **Sweeney**: K = 1 inversion, empty tail (A = id), x = θ₀ ⇒ 𝓚_E = ⟨g_E, δ⟩ ≈ η²σ. His k-decay curve (93% → 65.3%) is the failure of the *untransported* θ₀-local term; the transported Δ_k is the object his theory stops short of (F-003 §1).
- **Xu**: his commutator defect = ‖δ_k‖ at sampled points, untransported.
- **Yu/Arora**: their Δθ_t = (I+M_U)^t Δθ₀ is a transported initial defect — same J·δ algebra; the divergence theorem says gradient-ascent tail Jacobians are expansive; their off-span invariant (I−P_U)Δθ₀ is the Π_⊥ component of §2.4 for the local access class (F-004 §3).

### 2.2 Estimators (instrument spec, seeded for M3)

- **Branched replay (exact, expensive)**: for a sampled inversion, actually retrain the tail from the swapped intermediate word — yields Δ_k exactly. Pilot-scale only; this is the ground-truth instrument.
- **JVP transport (cheap, first-order)**: Δ̂_k = forward-mode JVP of the tail along the H₁ trajectory applied to δ_k; batchable across tracked defects. Error bound = Lemma K-2.
- **Instrument validation gate**: at M3.1 tiny scale, JVP estimates must match branched replay within a stated tolerance on a per-sample audit *before* any 𝓚 number feeds a headline plot (the instrument lies before the subject does). K = O(T²) inversions ⇒ sample inversions with declared estimator variance.

### 2.3 Transport operator (unchanged from v0.2)

T: Θ → Θ attempting T(θ(H₁)) ≈_f θ(H₂), subject to a declared budget: (a) no replay of training data, (b) compute ≪ retraining, (c) access limited to θ(H₁) plus a **declared side-information class 𝓘** (e.g., none / a few checkpoints of H₁ / gradient statistics / the history descriptor H₂ itself). 𝓘 must be stated with every claim — the Yu/Arora triangle honesty requirement: our operators are *history-aware by construction* (the carve-out that evades their impossibility), and the science is in how small 𝓘 and the compute budget can be.

### 2.4 TransportCost and target inequality v0.2 (sharpened)

- **TransportCost(H₁ → H₂; 𝓘, ε)** := minimum compute (gradient-evaluation-equivalents) over budget-respecting T ∈ 𝒯(𝓘) achieving functional gap d_f ≤ ε (d_f per D-006).
- **Reachable span.** An operator class induces, at each state, the set of realizable per-evaluation displacements R(𝓘, θ) (e.g., Yu/Arora's local class: the span of forget-set gradients). Π_⊥ := projection off span(R) — a working linear proxy; the conic/nonlinear refinement is an open M2 item (§2.5).
- **Target inequality v0.2** (to prove, or refute toward H3/the empirical fallback):

> TransportCost(H₁ → H₂; 𝓘, ε) ≥ f( ‖Π_{⊥,𝓘} Σ_k Δ_k‖_E , ε )

with ‖·‖_E a functionally-weighted norm and f monotone, diverging as the path-burned component exceeds the ε-tolerance. Sanity containments: (i) 𝓘 = local path-oblivious ⇒ Π_⊥ contains the off-span component ⇒ recovers Yu/Arora's impossibility as TC = ∞ below an ε floor; (ii) commuting histories (all δ_k = 0) ⇒ bound trivial and transport free — consistent.
- **Approximate path-integrability (ε, class 𝓒)**: H₁, H₂ ∈ 𝓒 are ε-integrable if TransportCost is o(retraining) at gap ε. **Phase boundary restated in 𝓚 terms**: integrable ⇔ 𝓚's mass concentrates inside the reachable span (cheap operators exist); **path-burned** ⇔ mass concentrates in Π_⊥ (no budget within 𝓘 suffices) — generalizing Yu/Arora's off-span component from "span of forget-set data" to an arbitrary declared access class.
- **Empirical fallback (D-009)**: if f resists proof, the deliverable is the measured scaling law: minimum observed budget vs 𝓚₁ / Π_⊥-mass across history pairs.

### 2.5 Open items carried in M2

- **Lemma K-2**: JVP estimator error bound, controlled by curvature along the segment [y_k, y_k + δ_k] and tail-Jacobian conditioning — segment-local, no global quadraticity.
- **Lemma K-3**: first-order schedule-independence of the attribution for reduced transposition schedules.
- **Lemma K-4**: augmented-state lift — statement and leading-order bracket for Adam on (θ, m, v).
- **Lemma K-5**: formal reduction of the v0.2 inequality to Yu/Arora Theorem 3.1 at 𝓘 = local path-oblivious.
- Homotopy generalization of Lemma K-1 to non-permutation pairs (continuous family H(s), surface-integral form) — prerequisite for H4 optimizer contrasts.
- Minimal side-information class 𝓘 for the first transport pilots.
- Conic/nonlinear refinement of Π_⊥.

## 3. Candidate mathematical handles

1. **Non-commutativity as curvature/holonomy** — now the primary handle. The per-phase update operators' brackets are the infinitesimal obstruction; the accumulated obstruction along the two-history loop is 𝓚. Key question inherited from Sweeney's own decay curve (93% at k=1 → 65% at k=50): the θ₀-local bracket loses validity along the path — what integrated object retains it? (Sweeney never constructs one; that decay is our motivation exhibit.)
2. **Conservative-field test** (Q3, ablation axis). Full-batch GD is a gradient flow; shuffled SGD is one in expectation; momentum/Adam generally are not. Prediction to test: non-conservative optimizers accumulate more 𝓚 for the same data and are correspondingly more expensive to transport — "shuffled SGD approximately integrable, Adam not." (Sweeney App. D.7 observes the SGD-bracket fails under AdamW and names an augmented-state commutator on (θ, m, v) as future work — cite, and go where he didn't: tie it to reversibility/cost, not order prediction.)
3. **Mode connectivity / symmetry quotient.** Candidate equivalence classes for 𝓒: histories whose endpoints are LMC-connected after symmetry alignment. Transport within a basin may be cheap; across basins, path-burned. Gives the cheap first transport operators (align + interpolate + light recalibration) for upper bounds.
4. **Information geometry.** Fisher-metric transport; EWC-style quadratic approximations as first-order transport operators, with their known failure modes as evidence about where f(𝓚) grows.

## 4. Competing hypotheses (re-pointed at the invariant)

- **H1 — Broad integrability**: within a fixed data multiset and shared init, late-training endpoints are cheaply transportable; 𝓚 stays bounded and f(𝓚) is small. Prediction: alignment + light recalibration closes most of the functional gap at compute ≪ retraining.
- **H2 — Stratified integrability**: transport cost grows with how early the histories diverge; 𝓚 accumulated in early/high-LR phases dominates f. (In §2.1 terms: early inversions carry the longest transport tails, hence the largest ‖J‖ amplification of their δ_k.) Prediction: monotone relation between divergence time and measured minimum budget.
- **H3 — Fundamental non-integrability**: a cost floor — f(𝓚) ≥ Ω(retraining) for generic history pairs; any cheap operator is secretly memorizing. Target: impossibility theorem generalizing Yu/Arora beyond local path-oblivious rules to a stated 𝓘 class. (A clean H3 result is a first-class deliverable.)
- **H4 — Optimizer stratification** (Q3): for matched data and compute, conservative optimizers sit in the integrable phase, stateful/non-conservative ones in the path-burned phase, and 𝓚 predicts the difference. (In §2.1 terms: Adam's augmented-state defects δ_k are larger and its tail Jacobians more expansive — Lemma K-4.) Prediction: SGD↔SGD transport succeeds at budgets where Adam↔Adam fails, with the gap tracking measured 𝓚.
- **H5 — Layer stratification**: 𝓚 and transport cost are depth/parameter-type dependent (embeddings/late layers cheap, early features path-burned). (In §2.1 terms: the per-layer blocks of Δ_k give this decomposition for free.) Prediction: per-layer transport ablations show systematic ordering matching per-layer bracket accumulation.

## 5. Positioning constraints (binding, from the M1 deep read)

1. Every transport claim declares (𝓘, compute budget, ε) — no undeclared side-information (Yu/Arora triangle).
2. All positive results demonstrated **off the NTK/lazy regime**, or they're folklore.
3. Related work must cite and delimit: Sweeney E.8 bracket-control (one local correction step at planning time ≠ endpoint transport between completed histories); Rukhovich arXiv:2501.15556 (original bracket-projection criterion); Xu 2602.10496; Sweeney 2606.29554 (Adam moment buffers); Goodbrake 2312.04739 (distinguish naturality from conservativeness); TracIn's commutativity assumption (the assumption we stress-test).
4. Degenerate-strategy baselines mandatory for every transport measurement: "do nothing," "average checkpoints," "fine-tune on eval-adjacent data" — the instrument lies before the subject does.
5. Headline numbers are functional/generalization gaps, never parameter distances (D-006).

## 6. Success metric discipline

- Report functional/generalization gaps with the (𝓘, budget, ε) triple attached; caveats travel with the number.
- Negative results are deliverables: a clean characterization of *why* transport fails (which f diverges, which coordinates are path-burned, for which optimizer class) satisfies the spec.
