# Design — Formal Framework (draft v0.2, post-M1 gate)

Status: DRAFT v0.2, revised after the M1.3 gate finalized (D-009; F-002/F-003/F-004). The core question is now fixed; definitions below are working definitions to be sharpened in M2. Nothing in this file licenses running experiments beyond the M3 roadmap discipline.

Supersedes v0.1 (pre-literature-review). Changes: core question locked to the holonomy→transport-cost invariant (Q2), with the transport-operator phase boundary (Q1) as constructive machinery and the optimizer taxonomy (Q3) as an ablation axis; positioning constraints from the two gating papers added; hypotheses re-pointed at the invariant.

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

## 2. Central definitions (working)

- **Accumulated holonomy 𝓚(H₁, H₂)**: a trajectory functional integrating the local non-commutativity (Lie-bracket field [f_i, f_j] = H_j g_i − H_i g_j of the per-phase/per-batch update fields) along the loop formed by the two histories. Candidate discretizations: (a) sum of bracket norms along aligned trajectory segments; (b) target-projected version ⟨g_E, ·⟩ for a functional (task-conditioned) invariant; (c) ordered-exponential (path-ordered product) mismatch between the two composed update sequences. Choosing among (a)–(c) — and proving any is well-defined off the quadratic regime — is the first M2 deliverable.
- **Transport operator**: T: Θ → Θ attempting T(θ(H₁)) ≈_f θ(H₂), subject to a declared budget: (a) no replay of training data, (b) compute ≪ retraining, (c) access limited to θ(H₁) plus a **declared side-information class 𝓘** (e.g., none / a few checkpoints of H₁ / gradient statistics / the history descriptor H₂ itself). 𝓘 must be stated with every claim — this is the Yu/Arora triangle honesty requirement: our operators are *history-aware by construction* (that is the carve-out that evades their impossibility), and the science is in how small 𝓘 and the compute budget can be.
- **TransportCost(H₁ → H₂; 𝓘, ε)**: minimum compute (in gradient-evaluation units) over budget-respecting operators achieving functional gap ≤ ε. The core question asks for lower bounds in terms of 𝓚 (and upper bounds from constructed operators — Q1 machinery).
- **Approximate path-integrability (ε, class 𝓒)**: histories H₁, H₂ ∈ 𝓒 are ε-integrable if TransportCost is o(retraining) at gap ε. The **phase boundary** is the frontier in (𝓒, 𝓘, budget)-space where this fails.
- **Path-burned information**: history coordinates whose effect cannot be transported away within the declared access class — generalizing Yu/Arora's off-span component, which is provably invariant under their local rule.

## 3. Candidate mathematical handles

1. **Non-commutativity as curvature/holonomy** — now the primary handle. The per-phase update operators' brackets are the infinitesimal obstruction; the accumulated obstruction along the two-history loop is 𝓚. Key question inherited from Sweeney's own decay curve (93% at k=1 → 65% at k=50): the θ₀-local bracket loses validity along the path — what integrated object retains it? (Sweeney never constructs one; that decay is our motivation exhibit.)
2. **Conservative-field test** (Q3, ablation axis). Full-batch GD is a gradient flow; shuffled SGD is one in expectation; momentum/Adam generally are not. Prediction to test: non-conservative optimizers accumulate more 𝓚 for the same data and are correspondingly more expensive to transport — "shuffled SGD approximately integrable, Adam not." (Sweeney App. D.7 observes the SGD-bracket fails under AdamW and names an augmented-state commutator on (θ, m, v) as future work — cite, and go where he didn't: tie it to reversibility/cost, not order prediction.)
3. **Mode connectivity / symmetry quotient.** Candidate equivalence classes for 𝓒: histories whose endpoints are LMC-connected after symmetry alignment. Transport within a basin may be cheap; across basins, path-burned. Gives the cheap first transport operators (align + interpolate + light recalibration) for upper bounds.
4. **Information geometry.** Fisher-metric transport; EWC-style quadratic approximations as first-order transport operators, with their known failure modes as evidence about where f(𝓚) grows.

## 4. Competing hypotheses (re-pointed at the invariant)

- **H1 — Broad integrability**: within a fixed data multiset and shared init, late-training endpoints are cheaply transportable; 𝓚 stays bounded and f(𝓚) is small. Prediction: alignment + light recalibration closes most of the functional gap at compute ≪ retraining.
- **H2 — Stratified integrability**: transport cost grows with how early the histories diverge; 𝓚 accumulated in early/high-LR phases dominates f. Prediction: monotone relation between divergence time and measured minimum budget.
- **H3 — Fundamental non-integrability**: a cost floor — f(𝓚) ≥ Ω(retraining) for generic history pairs; any cheap operator is secretly memorizing. Target: impossibility theorem generalizing Yu/Arora beyond local path-oblivious rules to a stated 𝓘 class. (A clean H3 result is a first-class deliverable.)
- **H4 — Optimizer stratification** (Q3): for matched data and compute, conservative optimizers sit in the integrable phase, stateful/non-conservative ones in the path-burned phase, and 𝓚 predicts the difference. Prediction: SGD↔SGD transport succeeds at budgets where Adam↔Adam fails, with the gap tracking measured 𝓚.
- **H5 — Layer stratification**: 𝓚 and transport cost are depth/parameter-type dependent (embeddings/late layers cheap, early features path-burned). Prediction: per-layer transport ablations show systematic ordering matching per-layer bracket accumulation.

## 5. Positioning constraints (binding, from the M1 deep read)

1. Every transport claim declares (𝓘, compute budget, ε) — no undeclared side-information (Yu/Arora triangle).
2. All positive results demonstrated **off the NTK/lazy regime**, or they're folklore.
3. Related work must cite and delimit: Sweeney E.8 bracket-control (one local correction step at planning time ≠ endpoint transport between completed histories); Rukhovich arXiv:2501.15556 (original bracket-projection criterion); Xu 2602.10496; Sweeney 2606.29554 (Adam moment buffers); Goodbrake 2312.04739 (distinguish naturality from conservativeness); TracIn's commutativity assumption (the assumption we stress-test).
4. Degenerate-strategy baselines mandatory for every transport measurement: "do nothing," "average checkpoints," "fine-tune on eval-adjacent data" — the instrument lies before the subject does.
5. Headline numbers are functional/generalization gaps, never parameter distances (D-006).

## 6. Success metric discipline

- Report functional/generalization gaps with the (𝓘, budget, ε) triple attached; caveats travel with the number.
- Negative results are deliverables: a clean characterization of *why* transport fails (which f diverges, which coordinates are path-burned, for which optimizer class) satisfies the spec.
