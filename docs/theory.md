# Theory — formal statements and proofs (M2)

Role: mathematical companion to design.md §2. design.md holds the definitions and spec; this file holds statements, proofs, and honest caveats. Lemma numbers match the ledger in design.md §2.5. Status tags: PROVED (complete argument), SKETCH (order-of-magnitude bookkeeping complete, ε-δ constants deferred to M5 write-up), STATED (no proof yet).

## 0. Notation and standing assumptions

- State space X ⊆ ℝⁿ (augmented for stateful optimizers, §K-4). Update maps u_t(x) = x + η_t f_t(x); word H = (u_1, …, u_T) applied left-to-right; Φ_H = u_T ∘ ⋯ ∘ u_1; trajectory x⁽¹⁾_0 = x₀, x⁽¹⁾_t = u_t(x⁽¹⁾_{t−1}) for H₁.
- **(A1) Regularity on a tube**: there is an open tube 𝒯 around the realized trajectories (radius exceeding every displacement appearing below) on which each f_t is C¹ (C² where stated), with F₀ := sup‖f_t‖, F₁ := sup‖Df_t‖, F₂ := sup‖D²f_t‖, and M_t := 1 + η_t F₁ ≥ sup_𝒯 ‖Du_t‖. Write M := max_t M_t, η := max_t η_t.
- **(A2) Tube containment**: all intermediate words' trajectories remain in 𝒯 (verified a posteriori via the K-1 sum; at pilot scale, checked numerically).
- Nonsmooth activations (ReLU): all statements hold with derivatives a.e. and 𝒯 avoiding kink crossings; where the tube crosses kinks the C¹ bounds fail locally — this is an instrument caveat (§2.2 validation gate), not a definitional one (K-1 needs no smoothness).
- Reduced schedule: an adjacent-transposition sequence from H₁ to H₂ performing exactly one swap per inversion pair of π (K = |Inv(π)| ≤ T(T−1)/2).

---

## Lemma K-1 (telescoping identity) — PROVED, exact

**Statement.** For any adjacent-transposition sequence W⁰ = H₁, …, W^K = H₂ and any x₀ for which all maps are defined:

> Φ_{H₂}(x₀) − Φ_{H₁}(x₀) = Σ_{k=1}^{K} Δ_k,  Δ_k := Φ_{A_k}(y_k + δ_k) − Φ_{A_k}(y_k),

where W^{k−1} = A_k ∘ (pair) ∘ B_k and W^k differs only by swapping the pair, x_k = Φ_{B_k}(x₀), y_k = (pair, old order)(x_k), δ_k = (pair, new order)(x_k) − (pair, old order)(x_k).

**Proof.** For each k, W^{k−1} and W^k share prefix word B_k and suffix word A_k. Both apply B_k first, reaching x_k. The old-order pair sends x_k to y_k; the new-order pair sends it to y_k + δ_k by definition of δ_k. Applying the shared suffix, Φ_{W^k}(x₀) − Φ_{W^{k−1}}(x₀) = Φ_{A_k}(y_k + δ_k) − Φ_{A_k}(y_k) = Δ_k. Summing over k telescopes the left side to Φ_{H₂}(x₀) − Φ_{H₁}(x₀). ∎

**Remarks.** (i) No smoothness, no invertibility, no small-η assumption — only that the maps are defined. This is the well-definedness-off-the-quadratic-regime claim, discharged by construction. (ii) The identity holds verbatim on augmented state spaces (K-4a).

---

## Lemma K-2 (JVP estimator error; segment-local) — PROVED

**Statement.** Let A be a tail word of m steps satisfying (A1)–(A2) with C² fields on the segment seg_k := [y_k, y_k + δ_k], and let J_A(y) = DΦ_A(y). Then

> ‖Δ_k − J_A(y_k) δ_k‖ ≤ ½ Λ_A ‖δ_k‖²,

where Λ_A is any Lipschitz constant of DΦ_A on seg_k, and

> Λ_A ≤ Σ_{j=1}^{m} S_j · (Π_{i≠j} M_i) · (Π_{i<j} M_i),  S_j := Lip(Du_j) ≤ η_j F₂.

With uniform constants: Λ_A ≤ η F₂ · M^{m−1} · (M^m − 1)/(M − 1).

**Proof.** By the fundamental theorem of calculus along seg_k:
Δ_k − J_A(y_k)δ_k = ∫₀¹ [J_A(y_k + sδ_k) − J_A(y_k)] δ_k ds, whose norm is ≤ ∫₀¹ Λ_A s‖δ_k‖ · ‖δ_k‖ ds = ½Λ_A‖δ_k‖².
For the composition bound, write DΦ_A(x) = Du_m(x_{m−1}) ⋯ Du_1(x_0) with x_j the partial compositions of x. The difference DΦ_A(x) − DΦ_A(x′) expands as a sum of m terms, the j-th replacing factor j by Du_j(x_{j−1}) − Du_j(x′_{j−1}), which has norm ≤ S_j Π_{i<j} M_i ‖x − x′‖ (the inner trajectory deviation grows by at most M_i per step); the remaining m−1 factors are bounded by Π_{i≠j} M_i. Summing gives the stated Λ_A. ∎

**Honest remarks.** (i) All constants are local to seg_k and the trajectory tube — **no global quadraticity of the loss is used anywhere**; this is where the "well-defined off the quadratic regime" burden actually lands, and it lands on the estimator, as intended. (ii) The geometric factor M^{m−1} is the same tail amplification present in the signal J_A δ_k, so the bound controls **absolute**, not relative, error: if J_A nearly annihilates δ_k, relative error can be large. This is exactly why the branched-replay validation gate (design.md §2.2) is binding and not decorative. (iii) For long tails the bound is vacuous unless η·m·F₁ = O(1); on longer horizons the estimator's validity is an empirical question settled by the gate.

---

## Lemma K-3 (first-order schedule-independence of attribution) — SKETCH

**Reference attribution (computable from H₁'s trajectory alone).** For an inversion pair (i, j) with u_i at position p and u_j at position q > p in H₁, define

> δ̄_{ij} := δ_{u_i, u_j}(x⁽¹⁾_{p−1}) (the pair defect as if adjacent at position p, on H₁'s trajectory),
> J̄_{ij} := D(Φ_{tail after p+1 in H₁})(evaluated along H₁'s trajectory),
> Δ̄_{ij} := J̄_{ij} δ̄_{ij}.

**Statement.** Under (A1)–(A2) with C² fields, for any **reduced** schedule, the attribution Δ_k of the swap realizing inversion (i, j) satisfies

> ‖Δ_k − Δ̄_{ij}‖ ≤ C(T, M, F₀, F₁, F₂) · η³,

while generically ‖Δ̄_{ij}‖ = Θ(η²). Hence the per-inversion attribution is schedule-independent at leading order in the step size. C grows polynomially in T with geometric factors M^{O(T)} (same honesty caveat as K-2(iii)).

**Proof sketch (order bookkeeping).**
1. *Tube control.* By K-1 applied to prefix words, every intermediate word's trajectory deviates from H₁'s by at most Σ‖δ‖ amplified by tails: O(K η² M^T). Within (A2) this keeps all evaluations in 𝒯.
2. *Defect field regularity.* δ_{u,v}(x) = η_u η_v B_{uv}(x) + O(η³) with B_{uv} = [f_u, f_v] + O(η) the bracket field, and Lip(δ_{u,v}) ≤ η_u η_v (F₀F₂ + F₁²) + O(η³) = O(η²).
3. *Evaluation-point shift.* In any reduced schedule the swap of (i, j) occurs at a state differing from x⁽¹⁾_{p−1} by O(Tη): the prefix multiset differs from H₁'s prefix-before-p by at most T near-identity updates (each moving the state O(η)), plus the O(Kη²) drift of step 1. By step 2, the defect changes by O(η²) · O(Tη) = O(Tη³).
4. *Transport-route exchange.* The actual tail at swap time differs from H₁'s tail-after-p by a permutation and by which updates have already crossed. Exchanging "apply near-identity u, then transport" with "transport, then apply u" costs ‖J_u δ(x) − δ(u(x))‖ = O(η · η²) per exchanged step (both equal δ(x) + O(η³)); at most T exchanges, each amplified by tail factors: O(Tη³ M^T).
Adding steps 3–4 gives the claim. ∎ (sketch)

**Corollary K-3.1 (single-trajectory surface formula) — the practically decisive consequence.**

> Φ_{H₂}(x₀) − Φ_{H₁}(x₀) = Σ_{(i,j) ∈ Inv(π)} J̄_{ij} δ̄_{ij} + O(K · T · η³ · M^{O(T)}).

Every term on the right is computable from **H₁'s trajectory alone** (checkpoints + two extra per-pair map evaluations for δ̄ + one JVP chain for J̄). No intermediate retraining, no access to H₂'s trajectory beyond its endpoint. This is what makes 𝓚 measurable as an instrument (design.md §2.2) rather than a thought experiment.

---

## Lemma K-4 (stateful lift; momentum makes order-defects first-order in η) — PROVED (b), SKETCH (d)

**(a) Lift.** K-1 applies verbatim on augmented state X = Θ × M (its proof never uses the structure of X). For Adam, exclude step-indexed bias correction from the swapped window (it is a position-dependent hyperparameter, breaking 𝓒_perm's step-exchangeability; use the index-free variant or hold correction factors fixed per position).

**(b) EMA momentum, exact leading order.** Let u_t act on (θ, m) by m⁺ = β m + (1−β) g_t(θ), θ⁺ = θ − η m⁺ (linear readout; β ∈ [0,1)). For two updates a, b at state (θ, m), direct computation of both orders gives:

> δ^θ = η β (1−β) (g_a(θ) − g_b(θ)) + O(η²),
> δ^m = (1−β)² (g_a(θ) − g_b(θ)) + O(η).

*Proof.* Order a-then-b: m₁ = βm + (1−β)g_a, m₂ = βm₁ + (1−β)g_b(θ₁) with θ₁ = θ − ηm₁; so m₁ + m₂ = (β+β²)m + (1−β)(1+β)g_a + (1−β)g_b + O(η), and θ_ab = θ − η(m₁+m₂). Swapping a and b and subtracting: the m-sums differ by (1−β)[(1+β) − 1](g_a − g_b) = β(1−β)(g_a − g_b), giving δ^θ; the final-m difference is (1−β)[β(g_a − g_b) − (g_a − g_b)] = −(1−β)²(g_a − g_b), giving δ^m (up to the O(η) θ-shift in gradient arguments). ∎

**Consequences.**
- β = 0 recovers plain SGD: the η-order term vanishes and δ^θ = O(η²) (the classical bracket order). ✓
- β > 0: **the order defect is Θ(η), one full order larger than SGD's Θ(η²)** whenever g_a ≠ g_b. Optimizer state remembers order at O(1) (δ^m), decays geometrically (factor β per subsequent step), and leaks into θ at O(η) per step — total transported θ-effect per inversion remains Θ(η).
- This *derives* what Sweeney (arXiv:2606.29554) observed empirically — batch order as a first-order noise source for stateful optimizers — and is the formal seed of H4: for matched data and η, momentum-class optimizers accumulate holonomy one order of η larger per inversion than SGD.
- Sanity: u_a = u_b ⇒ δ ≡ 0 exactly (commuting identical steps).

**(d) Adam (SKETCH).** Adam is EMA momentum composed with a normalization N(m, v) = m/(√v + ε̂) and a v-channel (β₂, squared gradients). The θ-step reads state through N; since the swapped m-arguments differ at O(1), the sum-difference N(·)+N(·) − N(·)−N(·) is generically O(1), so δ^θ = Θ(η) survives any C¹ normalization with bounded derivative (no special cancellation; the linear case (b) exhibits the explicit constant β₁(1−β₁)). The v-channel adds δ^v = Θ(1) mass with the same geometric decay in β₂. Regularity caveat: m/(√v + ε̂) is not C¹ at v = 0; use the smooth variant m/√(v + ε̂²) or note v > 0 after the first step wherever gradients are nonzero. Full constant-tracking is not needed for the order-of-η claim and is deferred.

*Sharpening (β₁ = 0, normalization-only; 2026-07-19).* With no momentum channel a linear readout gives δ^θ = O(η²) (case b at β = 0), so this case isolates the readout's nonlinearity. Freezing θ in the state recursion and expanding N in v: the two orders evaluate N at v-arguments differing by β₂(1−β₂)(v − g_b²) and β₂(1−β₂)(g_a² − v) respectively (elementwise), giving

> δ^θ = η · β₂(1−β₂) [ N_v(g_a, v̄)(v − g_b²) + N_v(g_b, v̄)(g_a² − v) ] + O(η²),  N_v := ∂N/∂v,

generically nonzero: **the v-channel alone makes order defects first-order**. Note the (1−β₂) suppression of the leading constant while O(η²) terms are unsuppressed — empirically this bounds the η-range where the first-order term dominates (visible in E-004's full-range fit). *Empirical status*: confirmed at pilot scale, E-004 / F-009 — slope 1.0014 (β₁ = 0.9) and 1.01 at small η (β₁ = 0), each with the frozen-θ leading-order predictor matching in norm and direction. Status stays SKETCH: constant-tracking and tail transport for the stateful case remain deferred to M5.

---

## Proposition P-1 (path-burned floor for fixed-subspace access classes) — PROVED

**Setting.** Linear predictors: prediction on input x is ⟨x, θ⟩; evaluation distribution with second moment Σ_E; functional metric d_f(θ, θ′) = ‖Σ_E^{1/2}(θ − θ′)‖ (RMS prediction gap — the RE-distance style metric, D-006). Operator class 𝒯_S: every operator (of any compute budget, any number of steps) produces total displacement T(θ) − θ ∈ S for a **fixed** subspace S ⊆ ℝⁿ.

**Statement.** Let Δθ = θ(H₁) − θ(H₂) = −Σ_k Δ_k (K-1). Then for every T ∈ 𝒯_S,

> d_f(T(θ(H₁)), θ(H₂)) ≥ dist_{Σ_E}(Δθ, S) := min_{c ∈ S} ‖Σ_E^{1/2}(Δθ + c)‖.

Consequently **TransportCost(H₁ → H₂; 𝒯_S, ε) = ∞ for every ε < dist_{Σ_E}(Δθ, S)** — no budget within the class suffices; the right side is the functionally irreducible (path-burned) residual.

**Proof.** T(θ(H₁)) = θ(H₁) + c for some c ∈ S, so d_f(T(θ(H₁)), θ(H₂)) = ‖Σ_E^{1/2}(Δθ + c)‖ ≥ min over c ∈ S. The minimum is a projection distance, attained; if ε is below it, no operator in the class reaches gap ε at any compute. ∎

**Significance.** This is the target inequality (design.md §2.4) **proved exactly** in the fixed-subspace case, and it fixes the previously-loose definition of the E-weighted projected norm: ‖Π_⊥ Σ_k Δ_k‖_E := dist_{Σ_E}(Σ_k Δ_k, S). Note the subtlety it resolves: reachable (on-span) motion can partially compensate off-span functional effects when Σ_E mixes the subspaces — the projection distance, not the naive orthogonal component, is the correct invariant.

**Limitation (honest).** Fixed-subspace classes cover Yu/Arora's local rules in linear models (K-5) and simple structured edits (e.g., last-layer-only or fixed-adapter-span edits at linearization). Realistic operators (fine-tuning a nonlinear net) have state-dependent reachable sets; the conic/nonlinear refinement remains the open frontier (design.md §2.5).

---

## Lemma K-5 (reduction to Yu/Arora, arXiv:2510.16629) — PROVED (a), STATED (b)

**(a) Static core (their off-span invariance) = P-1 instance.** In their overparametrized linear-regression setting, a local unlearning rule's every update is built from gradients of the forget-set loss: ∇_θ ½‖X_U θ − y_U‖² = X_Uᵀ(X_U θ − y_U) ∈ rowspace(X_U) =: S_U, a fixed subspace independent of θ. So any sequence of local updates lies in 𝒯_{S_U}, and P-1 gives: no local unlearning sequence, at any compute, brings the two order-variants within functional gap dist_{Σ_E}(Δθ, S_U); their invariant (I − P_U)Δθ₀ is exactly the Π_⊥ component (isotropic-Σ_E case). ∎

**(b) Dynamic refinement (their Theorem 3.1) — identification, proof of containment deferred.** Their divergence recursion Δθ_t = (I + M_U)^t Δθ₀ with M_U = (2η/k)X_UᵀX_U ⪰ 0 is, in our language, a transported defect: initial history-difference Δθ₀ (which K-1 decomposes into Σ_k Δ_k over the training-stage inversions — closed-form in the linear setting), pushed through tail Jacobians J = (I + M_U)^t of the shared unlearning phase. Their theorem is the statement that gradient-*ascent* tails are expansive (M_U ⪰ 0), i.e., not only does the P-1 floor persist — applying the shared local rule actively grows the gap on-span. The containment claim to prove formally in M5: theorem 3.1 = "expansive-transport" special case of the v0.2 inequality machinery with 𝓘 = local path-oblivious, transport phase = ascent. What our framework adds beyond their result: (i) arbitrary declared access classes S (not just forget-set spans), (ii) the attribution of Δθ₀ to inversions via K-1/K-3.1, (iii) the budgeted question — the minimum 𝓘 and compute at which the floor becomes escapable — which is exactly the unexplored vertex of their impossibility triangle (F-004 §2).

---

## Ledger summary

| Result | Status | Where the burden now sits |
|---|---|---|
| K-1 telescoping | PROVED (exact) | — |
| K-2 JVP error | PROVED | constants geometric in tail length → empirical validation gate binding |
| K-3 + Cor K-3.1 | SKETCH | ε-δ constants deferred to M5; sensitivity reporting stays mandatory |
| K-4 stateful lift | (b) PROVED, (d) SKETCH | Adam constant-tracking deferred; smooth-normalization variant assumed |
| P-1 path-burned floor | PROVED | fixed-subspace classes only; conic refinement open |
| K-5 Yu/Arora reduction | (a) PROVED, (b) STATED | formal Thm 3.1 containment proof deferred to M5 |
