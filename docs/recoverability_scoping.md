# M3.3 / M3.4 Recoverability Probes — scoping sketch (toward E-011)

**Status: DESIGN SKETCH ONLY. No compute, no experiment code, no pre-registration.** Authorized as the D-016 parallel non-compute design track. This document ends in (a) a recommended first probe, cheapest-first, and (b) a draft D-017 decision-entry skeleton for the principal to review. **Nothing here licenses a run**: every probe that survives review must get its own E-011 pre-registration (hypotheses, frozen gates/tolerances, degenerate baselines, void conditions, budget) committed *before* any cell, per the M3.2 discipline and the F-014 lesson.

**Why a new file, not a design.md section** (recorded per the supervisor's "your call"): `docs/design.md` is the *formal framework* doc (v0.4 — definitions, lemma ledger, target inequality). Recoverability *experiment scoping* is a different object (probe designs, budgets, testbed specifics) and would dilute the framework doc. This file is the scoping home; any *theory* that a probe forces (e.g. a formal recoverability functional) lands in design.md/theory.md via the normal supersession path.

Read-first pointers: CLAUDE.md → docs/design.md §2 (𝓚, Π_⊥, P-1) → docs/findings.md F-015/F-016 (the result these probes extend) → docs/decision_log.md D-013 (P1 selection + honest limits) / D-015 (no init-only control) → docs/experiment_log.md E-009/E-010 (the instrument these reuse).

---

## 1. What "recoverability" means here, and why it is the dual of transport

Roadmap M3.4 is *"trajectory-information measurements — what is recoverable from θ_T / checkpoint sequences."* Recoverability is the **read-out dual** of the transport-cost question (Q2):

- **Transport** asks: can we *move* θ(H₁) → θ(H₂) cheaply, and what is path-burned (uncorrectable at any budget within an access class 𝓘)?
- **Recoverability** asks: how much of the *history difference* between H₁ and H₂ can we *read back* from the endpoints / checkpoints, and by what 𝓘?

The two are linked by P-1 and the Π_⊥ decomposition (design.md §2.4): you cannot correct what you cannot detect, and the functionally-null (off-Σ_E) part of the endpoint difference is simultaneously the part that is *neither* recoverable functionally *nor* transportable. So a recoverability measurement bounds a transport-cost measurement from the information side — and, crucially, it is runnable at P1 scale where full transport is not (below).

**Why this is the right next measurement, and what F-015/F-016 already gave us.** F-015/F-016 established (n=3, correlated) that at 160M/300B the **order's** functional imprint on where two runs disagree *persists and saturates* (~0.95 of joint), while the init's washes out — i.e. **order is the durable, and therefore the candidate-recoverable, component of the history** (the palimpsest premise, design.md §0 / the project name D-007). Recoverability probes ask the natural follow-on: *given that the order trace survives to θ_T, where does it live (in time, in the network, in which subspace), and how much of it is functionally recoverable vs path-burned?* This is M3.4, it reuses the certified E-009/E-010 instrument, and it directly feeds the Q2 cost-law/Π_⊥ theory.

## 2. The object at P1 scale — scope decision inherited from D-013

The P1 history delta is a **permutation of ~147.16M tokenized 2048-token sequences** (D-013 fact 3), with K ≈ N²/4 ≈ **5×10¹⁵** sequence inversions. Consequences that bound every probe here (D-013 honest-limits, binding):

1. **Exact-order recovery is out of reach and is not the target.** "Recover the permutation π" is hopeless (and, where it *is* solvable, it is the already-solved training-order-recovery problem the brief §3.3 delimits against). Our target is **functional and coarse recoverability**: which *aspects* of the order difference are readable from θ_T, and how much is functionally irreducible.
2. **These are M3.4 divergence/recoverability measurements, not transport** (D-013): no full 𝓚-attribution, no branched replay at scale. Any 𝓚-flavored probe uses only coarse/low-rank trajectory summaries, never the O(K) attribution.
3. **Pythia is Adam + warmup/cosine** (E-009 G-iv): only the *order-of-η* laws (K-4b/K-4d) carry over from the SGD pilots, not constants. Optimizer-contrast recoverability (H4) is **not** a P1 probe — it needs different pairs and is an M4 item.
4. **Sequence-granularity remark (D-013 limit 1, K-1 read at sequence granularity)** is still an open theory.md item (M5); a probe that leans on 𝓚's per-inversion structure inherits it as a caveat.

## 3. Candidate probe classes

Each probe: **question · what it measures · 𝓘 consumed · degenerate baseline(s) · expected floor / ceiling · theory hook · budget · runnable-now? · instrument-skepticism guard**. Floors/ceilings are stated as *anchor definitions and expected ranges*, NOT frozen thresholds — freezing happens in E-011.

Reuse discipline (comparability, D-012): every *functional* measurement below uses the **F-015 eval shard + symmetric-KL metric + 12-point grid** unchanged, so recoverability d_f numbers sit directly next to F-015/F-016. New machinery (directional persistence, layer grafts, subspace projections) is *analysis on top of* the certified forward-eval instrument, never a change to it.

### R-1 — Temporal localization: is the endpoint order-trace "decided early"? (cheapest)
- **Question**: is the *direction* of the endpoint difference Δθ_T := θ(ds1@T) − θ(ds2@T) already set early in training, and if so, when?
- **Measures**: cosine (and Σ_E-weighted / functional cosine) between Δθ_t := θ(ds1@t) − θ(ds2@t) and Δθ_T across the grid; the checkpoint t* at which cos(Δθ_t, Δθ_T) first exceeds, say, 0.9; comparison against the *functional* divergence curve d_f(P1,t) already in F-015 (does the direction lock in before or after the magnitude saturates?).
- **𝓘 consumed**: the P1 checkpoint sequence (both runs) — parameter vectors only, already cached.
- **Degenerate baselines**: (i) cos between two *same-run* adjacent-checkpoint drift directions (ds1@t→t+Δ), the "unstructured drift" floor — is early→late order-direction persistence above what same-run drift coherence gives? (ii) cos of Δθ_t with a *random* fixed direction of matched dimension (≈0 floor).
- **Expected floor / ceiling**: floor = same-run drift-direction coherence (unknown, to measure; plausibly low); ceiling = 1.0 (perfect early lock-in). H2 predicts early lock-in (long tails amplify early inversions).
- **Theory hook**: **H2** (stratified integrability — early/high-LR divergence dominates via longest J·δ tails, design.md §4/K-1); sharpens F-015's saturation-timing open question.
- **Budget**: **near-free** — pure re-analysis of cached P1 (and optionally P2/P3) parameter vectors + a handful of same-run drift cells for the floor. No new large downloads (P1 grid cached from F-015).
- **Runnable now?** Yes (analysis-only; needs a small directional-persistence script — Phase-later, not now).
- **Skepticism guard**: parameter-space cosine can be dominated by high-norm/low-function directions (D-006). Report BOTH raw and Σ_E-weighted cosine; a raw-only "early lock-in" that vanishes under functional weighting is a parameter-space artifact, not recoverability.

### R-2 — Spatial localization: *where* in the network is the order trace written, and is it functionally recoverable there? (highest theory value, still cheap)
- **Question**: is the order-induced difference concentrated by layer / parameter-type (embeddings vs attention vs MLP vs LayerNorm; early vs late depth), and does its *functional* weight track its raw magnitude or dissociate (D-006)?
- **Measures**: (a) per-layer / per-type decomposition of ‖Δθ_T‖ (raw); (b) per-layer *functional* weight via a **graft/interpolation ablation** — form the hybrid θ_h(ℓ,α) = ds1 with layer ℓ's params moved a fraction α toward ds2's, sweep α, measure d_f(θ_h, ds1) and d_f(θ_h, ds2) on the F-015 shard; the layers whose small-α graft moves function most are where the order difference is functionally recoverable.
- **𝓘 consumed**: both endpoints θ(ds1@T), θ(ds2@T) (+ shared init θ₀ for displacement-from-init per layer). Cached.
- **Degenerate baselines**: (i) a *random* parameter subset of matched size/norm — does the order signal concentrate beyond a random split? (ii) magnitude-only prediction — does per-layer functional weight track per-layer d_θ, or dissociate? (iii) init-difference structure as a "non-order difference" reference for the *shape* of concentration (only C1@0 is available — a step-0 reference, F-014).
- **Expected floor / ceiling**: floor = random-subset functional effect at matched norm; ceiling = total d_f(P1@T) = 0.290461 (F-015). Expected (H5): concentration in a subset of layers, with functional weight *not* proportional to raw magnitude.
- **Theory hook**: **H5** (layer stratification — per-layer blocks of Δ_k, design.md §4/§2.1) and **P-1 / Π_⊥** (which subspaces carry functionally-recoverable order info vs functionally-null residual).
- **Budget**: cached endpoints; forward evals for the grafts — ~(#layers × #α-points) cells on the 98k-token shard, each ≈ one F-015 cell. Order ~10²–10³ forward passes; **modest**, GPU-hours not GPU-days.
- **Runnable now?** Yes (needs graft/ablation machinery — Phase-later).
- **Skepticism guard**: a *hard* cross-model graft between two runs 0.95 apart in d_θ can produce a **broken** hybrid whose huge d_f is functional collapse, not recovered order — which would spuriously read as "high recoverability." Guard: use the **α-interpolation sweep** (not a hard swap) and require a **coherence check** — the hybrid's CE loss must stay within a declared band of the two parents; grafts that blow up loss are void cells, not signal. (This guard must be a frozen void-condition in E-011.)

### R-3 — Access-class recoverability ladder: the Π_⊥ / path-burned measurement (deepest; most design needed)
- **Question**: how much of the endpoint functional difference is *recoverable* (in-span of a declared 𝓘) vs *path-burned* (off-span residual), as 𝓘 grows?
- **Measures**: project Δθ_T onto the span S(𝓘) of a small access class and measure the **Σ_E-projection distance** dist_{Σ_E}(Δθ_T, S) (design.md §2.4, P-1) — the functionally-irreducible residual — vs the in-span functional weight, along an 𝓘 ladder.
- **𝓘 ladder** (D-011 shape, measurement version): ∅ (recoverable = 0 baseline) → last-2-checkpoint update direction of run A → k late checkpoints → a low-rank trajectory summary. **Never** the O(K) attribution (scope §2).
- **Degenerate baselines**: (i) ∅ (0% recoverable); (ii) a *random* subspace of matched dimension — does the real update-span recover more than a random span of equal rank? (iii) trivial ceiling: 𝓘 = both full endpoints ⇒ 100% "recoverable," so the science is entirely in the *small-𝓘* regime and its slope.
- **Expected floor / ceiling**: floor = random-subspace recovery at matched rank; ceiling = 1.0 at 𝓘 = both endpoints (uninformative — excluded from claims). The interesting quantity is the **recoverable fraction vs rank(𝓘)** curve and where it plateaus (the path-burned floor).
- **Theory hook**: **P-1** (exact for fixed-subspace classes — the recoverable/path-burned split is literally the theorem's object) and **Yu/Arora off-span** (K-5a); this is the most direct real-scale test of the design.md §2.4 inequality's information side.
- **Budget**: cached checkpoints; parameter-space linear algebra + forward evals to estimate Σ_E (output-sensitivity / empirical Fisher on the shard) and the functional weights — **moderate-to-heavy**, and it needs a defensible **Σ_E estimator** at 160M (the main design risk).
- **Runnable now?** Partially — needs the Σ_E-metric design settled first. Recommend as the *third* rung, after R-1/R-2 de-risk the machinery.
- **Skepticism guard**: a full-rank or near-full-rank 𝓘 trivially "recovers" everything; any recoverable-fraction number must be reported *with rank(𝓘) and against the random-subspace floor at that rank*, or it is meaningless. Σ_E estimator must itself pass an instrument check (e.g. reproduce a known functional distance) before any Π_⊥ number is trusted (F-006 / "instrument lies first").

### R-4 — Order-vs-init recoverability separability (**BLOCKED**)
- **Question**: at θ_T, do the order-difference and the (washed-out) init-difference occupy distinguishable functional subspaces?
- **Blocker**: requires an **init-only control** (same order, different init) — none exists publicly (F-014/D-015). This is the *same* dependency as the D-015 AllenAI track. **Deferred, not designed**; if the AllenAI weights land, continuity-audit first (F-014 discipline), then this becomes a natural E-01x. Listed so the gap is explicit, not silently dropped.

### Cross-cutting: robustness via the correlated pairs
R-1 and R-2 can run on **P1 primarily, with P2/P3 as correlated replicates** (the F-016 pattern) — a cheap n=3 robustness read on any recoverability number, carrying the **same binding correlated-pairs caveat** (3 pairs among 3 runs sharing one init; bounds order-only pair-noise only). No init-to-init robustness is available (F-014).

## 4. Comparability, instrument reuse, and what is genuinely new code
- **Reused unchanged** (D-012): F-015 eval shard (`results/raw/e009_eval_tokens.npy`), symmetric-KL `d_f`, `d_theta_rel`, `load_model`, the 12-point grid — so recoverability functional numbers are directly comparable with F-015/F-016.
- **New analysis code** (to be built under E-011, not now): directional-persistence (R-1), layer-graft/α-interpolation with a loss-coherence void-guard (R-2), subspace projection + a Σ_E estimator (R-3). All are *read-only consumers* of cached checkpoints + the certified forward-eval path; none modify a measured-number code path.
- **F-014 gate obligation (binding)**: any probe consuming a checkpoint *sequence* (R-1, R-3) inherits the continuity requirement. P1's continuity is already backed by the F-015 curve + the E-010 audit of the shared-init family; a fresh E-011 still restates the tripwire (a P1 cell collapsing toward the family scale voids it).

## 5. Open design questions to settle before E-011
1. **Σ_E at 160M** (gates R-3): which functional metric — empirical Fisher on the shard, output-Jacobian Gram, or a cheaper diagonal proxy — and its own instrument check. This is the main design risk; R-1/R-2 deliberately avoid it.
2. **Graft validity band** (gates R-2): what CE-loss coherence band makes a hybrid a valid cell vs a void "broken model," frozen before the run.
3. **Directional metric** (R-1): raw vs Σ_E-weighted cosine — report both; decide which (if any) is gate-bearing.
4. **What counts as a positive recoverability result**: a *fraction* and a *floor* (vs random-subspace / same-run drift), never an absolute number — every probe must beat a declared degenerate baseline, or it is measurement noise (mandatory, per the roadmap §5.4 degenerate-baseline rule).
5. **Scope guard**: keep every probe *functional/coarse* — no drift toward exact-π recovery (the solved, delimited-against problem).

## 6. Recommendation (cheapest-first)
Order the probes by information-per-compute:
1. **R-1 first** — near-free (cached re-analysis), directly extends F-015's saturation-timing question into a directional "when is order written" statement, tests H2, and de-risks the parameter-vs-functional (D-006) handling the later probes need.
2. **R-2 second** — modest forward-eval cost, highest theory value (H5 + Π_⊥ localization), clean degenerate baselines, and it builds the graft machinery.
3. **R-3 third** — deepest (the real-scale P-1/Π_⊥ test) but gated on the Σ_E design; do it after R-1/R-2.
4. **R-4 deferred/blocked** on the init-only control (D-015 AllenAI track).

Each rung is a *separate* E-011(a/b/c) pre-registration. This is a recommendation for the principal, **not** a decision — the decision skeleton below is for review.

---

## 7. Draft decision-entry skeleton — D-017 (NOT YET A DECISION; for principal review)

> ## D-017 (DRAFT — not adopted) — M3.4 recoverability probe program on P1: sequence of functional/coarse read-out measurements, R-1 first
> - **Problem**: F-015/F-016 showed the order trace persists to θ_T at real scale (order is the durable component); M3.4 asks *what is recoverable* about that order from θ_T / the checkpoint sequence, and how much is path-burned. Which probe(s), in what order, and what does each consume/test?
> - **Options**: (a) R-1 temporal-localization only (cheapest, H2); (b) R-1 then R-2 spatial-localization (adds H5 + Π_⊥ localization); (c) full ladder R-1→R-2→R-3 (adds the real-scale P-1/Π_⊥ path-burned measurement, gated on a Σ_E design); (d) wait for the AllenAI init-only control and do R-4 (order-vs-init separability) instead/first; (e) defer recoverability entirely, do M5 writing.
> - **Choice**: *[TBD by principal]* — recommendation on file is (b) as the committed near-term program (R-1 then R-2), with R-3 as a gated stretch behind a Σ_E instrument check and R-4 parked on the D-015 track.
> - **Why**: cheapest-first (R-1 is cached re-analysis); reuses the certified E-009/E-010 instrument (comparability with F-015/F-016); attacks the theory's information side (H2, H5, Π_⊥/P-1) at real scale where transport is out of reach; every probe carries a mandatory degenerate baseline and the correlated-pairs caveat.
> - **Trade-offs**: functional/coarse recoverability only (exact-π is out of scope and out of reach at P1); Adam/LR caveats (K-4 constants don't carry); R-3 rests on a Σ_E estimator that is itself a design risk; R-4 stays blocked without an init-only control.
> - **How to reverse**: each probe is a separate E-011 pre-registration; drop or reorder via a superseding entry. If the AllenAI weights land, R-4 jumps the queue (new registration, continuity-audit first).

---

*End of scoping sketch. Next action if adopted: principal reviews §6/§7, promotes the chosen probe(s) to a D-017 decision, then an E-011 pre-registration is written and committed BEFORE any recoverability cell runs. No compute until then.*
