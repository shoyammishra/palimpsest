# Design — Formal Framework (draft v0.1, pre-literature-review)

Status: DRAFT. Written before the M1 novelty gate; every definition here is provisional and will be revised or discarded based on the literature map. Nothing in this file licenses running experiments.

## 1. Objects

- **History space 𝓗**: a training history H is the full optimization configuration — data ordering, curriculum, optimizer + hyperparameters, LR schedule, augmentation schedule, replay schedule, staged phases (pretrain → SFT → RLHF). Data *content* (the multiset of examples) may be held fixed or varied; the primary regime fixes the multiset and varies everything else.
- **Training map Φ**: Φ(θ₀, H, ω) = θ_T, with ω the residual stochasticity (e.g., dropout/augmentation seeds) not folded into H. The trajectory is τ(H) = (θ₀, …, θ_T).
- **Functional equivalence**: θ ≈_f θ′ iff the induced functions match in distribution on evaluation measures of interest (generalization behavior), NOT parameter-space closeness. Parameter match is neither necessary (permutation/scaling symmetries) nor the scientifically interesting target.

## 2. Central definitions (provisional)

- **Transport operator**: T: Θ → Θ attempting T(θ(H₁)) ≈_f θ(H₂), subject to budget constraints: (a) no replay of training data, (b) compute ≪ retraining, (c) access limited to θ(H₁) plus declared side-information (specifying allowed side-information is itself a design axis — e.g., a few checkpoints, gradient statistics, nothing).
- **Approximate path-integrability (ε, class 𝓒)**: histories H₁, H₂ in a class 𝓒 are ε-integrable if a budget-respecting T exists with functional gap ≤ ε. SGD is "approximately path-integrable over 𝓒" if this holds for all pairs in 𝓒.
- **Path-burned information**: coordinates of H whose value cannot be recovered from θ_T *and* whose effect cannot be transported away — irreversibly encoded history.
- **Trajectory information bottleneck**: θ_T as a lossy compression of τ(H); the recoverable/editable/destroyed trichotomy over history coordinates is the object of study (I(H; θ_T) framing).

## 3. Candidate mathematical handles

1. **Non-commutativity as curvature/holonomy.** Per-example (or per-phase) update operators U_i generally do not commute; the commutator [U_i, U_j] is the infinitesimal obstruction to reordering. Question: does the accumulated obstruction (a holonomy-like quantity along the trajectory) stay bounded in relevant regimes (late training, within a basin) or grow? Small accumulated holonomy ⇒ approximate integrability.
2. **Conservative-field test.** Full-batch GD follows an exact gradient field (trivially "conservative"). SGD's expected update is still a gradient field of the empirical loss, but momentum/Adam/weight-decay-with-schedule generally are not gradient flows of any potential. Characterize which optimizer classes admit a potential (exactly or approximately) — a potential would make endpoints depend only on endpoint-sufficient statistics.
3. **Mode connectivity / symmetry quotient.** Linear mode connectivity modulo permutation suggests trained networks from a shared regime fall into large connected functional basins. Candidate equivalence classes for 𝓒: histories whose endpoints are LMC-connected after symmetry alignment. Transport within a basin may be cheap; across basins, path-burned.
4. **Information geometry.** Fisher-metric transport between endpoints; EWC-style quadratic approximations as first-order transport operators (and their known failure modes as evidence toward non-integrability).

## 4. Competing hypotheses (to be sharpened after M1)

- **H1 — Broad integrability**: within a fixed data multiset and shared initialization, late-training endpoints are functionally transportable; most history coordinates wash out. Prediction: cheap transport (symmetry alignment + light recalibration) closes most of the functional gap.
- **H2 — Stratified integrability**: some history coordinates are transportable (LR-schedule tail, late-phase order), others are path-burned (early curriculum, forgetting events, optimizer family). Prediction: transportability decays with how early in training the histories diverge.
- **H3 — Fundamental non-integrability**: transport to θ(H₂) requires information absent from θ(H₁); any operator that succeeds is secretly retraining or memorizing. Prediction: functional gap floor that no budget-respecting T crosses; target = impossibility theorem under stated assumptions.
- **H4 — Layer stratification**: integrability is depth/parameter-type dependent (e.g., embeddings/late layers transportable, early features path-burned). Prediction: per-layer transport ablations show systematic ordering.

## 5. Success metric discipline

- Report functional/generalization gaps, never parameter distances, as headline numbers.
- Every transport claim requires a degenerate-strategy baseline (what does "do nothing," "fine-tune on eval-adjacent data," or "average checkpoints" score?) — the instrument lies before the subject does.
- Negative results are deliverables: a clean characterization of *why* transport fails, with necessary conditions, satisfies the spec.
