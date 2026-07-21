# Decision Log

Format per entry: problem, options, choice, why, trade-offs, how to reverse.

---

## D-001 (2026-07-17) — Adopt FABLE5 workflow and docs structure
- **Problem**: New project needs a state/memory system a cold session can resume from.
- **Options**: (a) ad-hoc notes; (b) FABLE5_PROJECT_PROMPT.md structure (CLAUDE.md dashboard + docs/ + results/).
- **Choice**: (b).
- **Why**: User-mandated; makes the repo self-sufficient without conversation history.
- **Reverse**: restructure docs/, update CLAUDE.md pointers.

## D-002 (2026-07-17) — Novelty verification is a hard gate before any compute
- **Problem**: Spec's biggest risk is that the core question is already answered; experiments before that check waste compute and bias us toward defending the idea.
- **Options**: (a) start pilots in parallel with lit review; (b) hard gate — zero experiments until M1 verdict.
- **Choice**: (b).
- **Why**: Spec makes novelty "highest priority"; cheapest-first principle — a literature search is the cheapest possible information purchase; parallel pilots create sunk-cost pressure.
- **Trade-off**: slower start if the idea survives. Acceptable.
- **Reverse**: log a superseding decision if a pilot becomes necessary to *inform* the reformulation.

## D-003 (2026-07-17) — Preserve the original spec verbatim
- **Problem**: The spec is the contract; paraphrases drift.
- **Choice**: docs/research_spec.md holds it verbatim, never edited; changes happen via decision-log supersession.
- **Reverse**: n/a (append-only by convention).

## D-004 (2026-07-17) — Kill-scan delegated to Opus 4.8 subagent
- **Problem**: The M1.1 kill-scan is deep parallel research; burning principal context on raw searching is wasteful.
- **Options**: (a) principal does all searches inline; (b) delegate scan to an Opus subagent with full context, principal reviews and writes verdict.
- **Choice**: (b), per the standing delegation policy (Opus = default engineer for judgment-bearing research; agent returns a report, principal writes files).
- **Trade-off**: agent starts cold — mitigated by passing spec + framing in the task prompt.
- **Reverse**: rerun searches inline if the agent's report fails review.

## D-005 (2026-07-17) — Initialize git at project start
- **Problem**: "Small commits, often" requires a repo from day one.
- **Choice**: `git init` + scaffold commit at M0.
- **Why**: Never leave state uncommitted; enables honest history of the research process.
- **Reverse**: n/a.

## D-008 (2026-07-17) — M1.3 gate: provisional REFORMULATE; deep-read of gating papers required before finalizing
- **Problem**: Kill-scan (F-001) says the broad framing ("holonomy of training order" + "impossibility of reconciling staged histories") is occupied by Sweeney ICML 2026 and Yu/Arora 2025, but three questions survive.
- **Options**: (a) PROCEED with original framing; (b) ABANDON; (c) REFORMULATE around the surviving questions (transport-operator phase boundary; holonomy→transport-cost invariant; conservativeness-reversibility taxonomy).
- **Choice**: (c), PROVISIONAL — not final until arXiv:2606.24993 and arXiv:2510.16629 are read end-to-end by the principal and deltas logged. A gate decision must not rest on a subagent's secondhand summary.
- **Why**: P(incremental)≈0.65 kills (a); three verified gaps and a ready testbed (PolyPythias) argue against (b).
- **Trade-off**: reformulation narrows scope and hard-requires the Yu/Arora carve-out (history-aware operator class) plus off-NTK-regime evidence.
- **Reverse**: superseding entry after the deep read — may upgrade to final REFORMULATE with a chosen question, or downgrade to ABANDON if the deep read shows the carve-outs don't hold.

## D-007 (2026-07-17) — Project name: "Palimpsest"
- **Problem**: Working name "fable" was a placeholder; folder name "path dnn" contained a space.
- **Options**: Palimpsest, Holonomy, Monodromy, Rewind.
- **Choice**: Palimpsest; folder renamed to `palimpsest`.
- **Why**: Captures the core object — final weights as an overwritten record of training history with partially recoverable traces — and covers both the information-theoretic and geometric framings; "Holonomy" would over-commit to one math handle before the M1 verdict.
- **Reverse**: rename folder + title lines; no code depends on the name.

## D-009 (2026-07-17) — M1.3 gate FINAL: REFORMULATE around Q2 (holonomy→transport-cost invariant); supersedes D-008
- **Problem**: D-008 left the REFORMULATE verdict provisional pending an end-to-end principal read of the two gating papers, and left the choice among the three surviving questions open.
- **Inputs**: F-002 (targeted Sweeney kill-check), F-003 (Sweeney full read), F-004 (Yu/Arora full read). Both papers read end-to-end by the principal; D-008's condition is discharged.
- **Options**: core = Q1 (transport operator + phase boundary), Q2 (transport-cost ≥ f(accumulated holonomy) invariant), Q3 (optimizer conservativeness↔reversibility taxonomy); or ABANDON.
- **Choice**: **REFORMULATE, final. Core = Q2**, structured as: Q2 is the headline claim; Q1 is its constructive machinery (you cannot measure minimum transport cost without building budget-constrained history-aware transport operators — the integrable/path-burned phase boundary falls out as where f diverges); Q3 is demoted to an ablation axis and fallback (SGD-vs-Adam trajectories as holonomy/cost contrast).
- **Why**:
  - P(incremental)≈0.65 is the number to beat, and Q2 attacks it structurally: an invariant linking accumulated holonomy to minimum transport cost would subsume Sweeney's order prediction and Xu's commutator defect as special cases, and should contain Yu/Arora's divergence theorem as the zero-budget/local limit (F-004 §3 — a concrete, checkable containment target).
  - Both designated kill-checks passed: Sweeney ties the bracket to nothing but order choice (never uses "holonomy"; no accumulated quantity; F-003), and Yu/Arora's impossibility is confined to local+path-oblivious rules with the history-aware branch of their own impossibility triangle explicitly unexplored (F-004).
  - Q2 degrades gracefully: no provable inequality → empirical scaling law (novel); transport fails everywhere → evidence for a cost floor (design.md H3); Q3 remains inside the paper as ablation. Q1-as-core would read as a Yu/Arora follow-up; Q3-as-core is newly riskier because Sweeney's App. D.7 names the augmented-state commutator as his own future work (F-003 §4).
- **Trade-offs**: mathematically the most ambitious option; slowest to first headline number (needs both a holonomy-measurement pipeline and a transport pipeline). Mitigated by PolyPythias as a ready testbed and the M3.1 tiny-scale pilot mandate. Obligations inherited from the deep read: cite and delimit Sweeney E.8 (bracket-control step) and Rukhovich 2501.15556; declare side-information class and compute budget for every transport claim (Yu/Arora triangle honesty); all positive results demonstrated off the NTK/lazy regime.
- **How to reverse**: superseding entry if (a) the accumulated-holonomy object proves ill-defined/unmeasurable at pilot scale (fall back to Q1-as-core with the budget-escape framing), or (b) new literature (e.g., a Sweeney augmented-state follow-up that adds cost) collapses the delta — re-run a targeted kill-scan before M3 spend.

## D-006 (2026-07-17) — Success metric is functional/generalization gap, not parameter distance
- **Problem**: Parameter-space closeness is confounded by permutation/scaling symmetries and is not the scientific target.
- **Choice**: All transport/integrability claims measured functionally (generalization behavior), with degenerate-strategy baselines mandatory.
- **Why**: Spec Phase 4 ("measure generalization rather than memorization"); instrument-skepticism principle.
- **Reverse**: superseding entry with justification if a parameter-space metric proves theoretically necessary.

## D-010 (2026-07-19) — 𝓚 discretization: transported-defect (discrete non-abelian-Stokes) form; former candidates (a),(b) absorbed as derived scalars
- **Problem**: design.md v0.2 §2 listed three candidate discretizations of the accumulated-holonomy functional 𝓚 — (a) sum of bracket norms along aligned segments, (b) target-projected ⟨g_E,·⟩ version, (c) ordered-exponential mismatch — and choosing one, with a well-definedness argument off the quadratic regime, was the first M2 deliverable.
- **Options**: (a); (b); (c) naively (endpoint mismatch of composed maps); (c-exact) — (c) expanded via the exact swap-defect telescoping identity (Lemma K-1): endpoint gap = Σ_k Δ_k, each Δ_k an exact two-step order defect δ_k transported through the actual remaining training maps, canonically indexed by inversion pairs of the permutation.
- **Choice**: **(c-exact)** as the definition (design.md §2.1), with (b) recovered as the functional projection 𝓚_E and (a) as the norm envelope 𝓚₁. Nothing discarded — the candidates were levels of one object.
- **Why**:
  1. **Well-definedness off the quadratic regime comes for free**: δ_k and Δ_k are finite compositions/differences of the actual update maps — no BCH/Taylor truncation anywhere in the definition. (a) and (b) as standalone definitions are truncated objects and inherit exactly the k-decay failure Sweeney documents (93%→65.3%); the analytic burden moves to the estimator error bound (Lemma K-2), where it belongs.
  2. **All three containment targets become structural, not analogical**: Sweeney's σ = untransported single-inversion limit; Xu's commutator defect = ‖δ_k‖ samples; Yu/Arora's (I+M_U)^tΔθ₀ = a transported defect with expansive tail Jacobians, and their off-span invariant = the Π_⊥ component of the sharpened cost bound (design.md §2.4).
  3. **The J·δ factorization is the science**: it separates where non-commutativity accrued (δ_k) from how it was amplified downstream (tail Jacobian) — precisely what H2 (early divergence), H4 (optimizer), H5 (layer) predict against.
  4. Naive (c) is circular (endpoint gap = the thing transport closes); the telescoped form escapes circularity through the decomposition, the mass 𝓚₁, and the Π_⊥ projection (objection pre-empted in §2.1).
- **Trade-offs**: attribution is gauge-dependent at second order in δ (mitigated: canonical bubble-sort schedule + Lemma K-3 first-order independence + sensitivity reporting); measurement needs JVP chains or branched replay with K = O(T²) inversions (mitigated: inversion sampling; validation gate in §2.2); first theorems restricted to 𝓒_perm with step-exchangeable schedules (non-permutation pairs deferred to the homotopy generalization, §2.5).
- **How to reverse**: if JVP transport proves numerically unstable at pilot scale (exploding tail Jacobians drowning signal) or fails the branched-replay validation gate, fall back to (b) with k-step windows and a declared locality decay, via a superseding entry. Re-run a targeted kill-scan if any new paper builds an accumulated/transported bracket before our M3 spend.

## D-011 (2026-07-19) — Side-information class for first transport pilots: 𝓘₁ = {permutation π, H₁ checkpoints at inversion positions}; ∅ as mandatory baseline rung
- **Problem**: Every transport claim must declare (𝓘, budget, ε) (design.md §5.1, Yu/Arora triangle honesty). The first pilots need a declared class, small enough to be interesting, large enough that our theory constructs an operator for it.
- **Options**: (a) 𝓘 = ∅ — θ(H₁) only; (b) 𝓘₁ = {the history delta as a permutation π, plus H₁ checkpoints at the inversion positions} — exactly the inputs Corollary K-3.1 consumes; (c) gradient statistics only (Fisher/EWC-style); (d) full H₂ trajectory — rejected outright (trivializes the question toward "you retrained").
- **Choice**: **(b) 𝓘₁ as the declared pilot class**, with (a) always run as the do-nothing/degenerate baseline rung and a planned ablation reducing checkpoint count. First transport operator = **𝓚-aware correction**: T(θ(H₁)) = θ(H₁) + Σ_{(i,j)∈Inv(π)} J̄_{ij} δ̄_{ij} (adds the predicted endpoint gap, K-3.1), budget counted in gradient-evaluation-equivalents of the extra map evaluations + JVP chains.
- **Why**: (1) 𝓘₁ is the smallest class for which the theory *constructs* an operator rather than postulating one — the invariant and the operator are the same computation, so pilot transport results directly validate or falsify the framework; (2) manifestly history-aware ⇒ outside Yu/Arora's impossibility by their own scope fence (F-004 §1); (3) budget accounting is clean and honest — if the JVP chains cost ≈ retraining, that is a *finding about the phase boundary*, not an embarrassment to hide (report per §6). (c) is deferred to the operator-ablation phase (M3.5); it lacks a constructive link to 𝓚 today.
- **Trade-offs**: 𝓘₁ presumes 𝓒_perm pairs (permutation delta known/meaningful) — acceptable at pilot scale where we construct the pairs ourselves; checkpoint storage O(K) — trivial at tiny scale, needs subsampling later.
- **How to reverse**: supersede after M3.1 if (i) the ablation shows checkpoints unnecessary (shrink 𝓘) or (ii) the K-3.1 operator fails the validation gate (then 𝓘₁ pivots from "operator input" to "measurement input" and the operator search widens per Q1 machinery).

## D-012 (2026-07-19) — Budget accounting rules for transport operators (grad-eval-equivalents; caching; HVP convention)
- **Problem**: compressed implementations (batched JVP, cached pair evaluations) change what "budget" counts. Without fixed rules, cost numbers are not comparable across experiments, and E-003's published 25–713× figures could get silently blended with numbers computed under a different instrument (comparability principle).
- **Options**: (a) wall-clock time; (b) raw uncached grad-eval counts — E-003's de facto rule (4 evals per inversion + one tail replay); (c) declared grad-eval-equivalents with explicit conventions: trajectory gradients are free (recoverable from the 𝓘₁ checkpoints as (chk[t]−chk[t+1])/η — no computation, already-declared side information); repeated evaluations may be cached and are counted once; one FD-HVP = 2 grad evals; instrument/audit evaluations (validation-gate replays) are excluded from the *operator's* budget and reported separately; retraining comparator = T grad evals.
- **Choice**: **(c) as the standard from E-005 onward, with every cost table reporting BOTH (b) and (c) side by side until M4** — no cross-rule blending; E-003's numbers remain (b)-rule and stay re-derivable from its raw JSON.
- **Why**: (a) is hardware-dependent noise; (b) punishes pure implementation dedup that involves no approximation, which would misstate the phase boundary; (c) counts the information that must actually be computed — the quantity the theory's cost bounds are about.
- **Trade-offs**: (c) is the rule most favorable to us; mitigated by mandatory dual reporting and by keeping raw counts in the results JSON so any rule can be recomputed.
- **How to reverse**: superseding entry; raw per-component counts stay in results files so past numbers can be recast under any future rule.

## D-013 (2026-07-20) — M3.2 history-pair selection: PolyPythias 160M decoupled data-seed runs are the first real history pairs; weight-seed runs are the matched control
- **Problem**: M3.2 needs *real* (not self-constructed) history pairs, cheapest-first, with maximal proximity to the theory's proved scope (𝓒_perm, permutation pairs). Which PolyPythias runs?
- **Facts** (verified 2026-07-20 against the HF model cards, `EleutherAI/pile-preshuffled-seeds`, and arXiv:2503.09543 / ICLR 2025):
  1. PolyPythias = 45 new runs: 9 seeds × 5 sizes (14M/31M/70M/160M/410M), each 300B Pile tokens, **154 checkpoints per run as git branches** (`step0` init, 10 log-spaced to 1B tokens, 143 evenly spaced 2B→300B, loadable via `revision="stepN"` — steps to 143000).
  2. **Decoupled variants exist at 160M only**: `pythia-160m-data-seed1..3` (data order varies, weight init fixed) and `pythia-160m-weight-seed1..3` (init varies, data order fixed).
  3. The data orderings are numpy index maps over the **same 147,164,160 tokenized 2048-token sequences** (`pile-preshuffled-seeds`, seeds 0–9, base seed 1234): content identical across seeds, order permuted at **sequence granularity**. A data-seed pair's history delta is therefore a genuine permutation — of sequences, not batches.
  4. Not verifiable from cards: whether the data-seed variants' fixed init equals the main run's (seed-1234) init. Checkable empirically by diffing `step0` weights.
- **Options**: (a) 14M standard-seed pairs — cheapest per eval, but init AND order both differ → outside 𝓒_perm from day one, lands on the unproven homotopy generalization; (b) 160M data-seed pairs — order-only delta, inside 𝓒_perm's closure, only size with decoupling; (c) 160M weight-seed pairs — init-only, zero-permutation-holonomy control; (d) 410M — no decoupling, most expensive.
- **Choice**: **(b) primary: P1 = (data-seed1, data-seed2)** — same init by construction, order-only delta. **(c) as matched control: C1 = (weight-seed1, weight-seed2)** — same order, init-only delta. **J1 = (seed1, seed2)** standard pair as the joint-variation reference. All 160M, same architecture/data/compute — the M3.3 controlled-pair construction already run by EleutherAI, for free.
- **Why**: order-only pairs are the only public pairs whose history delta is literally a permutation — the object K-1..K-3/𝓚 are built on; 160M is forced (only decoupled size); the P1-vs-C1 contrast directly instruments the core question (is order-induced functional divergence structurally different from init-induced divergence — H2/H4 hooks: Adam holonomy is Θ(η) per inversion per K-4b/K-4d, so order-divergence should couple to the LR schedule; init-divergence has no such η-coupling).
- **Honest limits, logged now**: (1) sequence-level reshuffling changes batch *composition*, not just batch order — the batch multiset differs across the pair; K-1 must be read at sequence granularity (batch gradient = mean of per-sequence gradients). Formal remark needed in theory.md before any 𝓚 claim on P1 (M5 item). (2) K for a full reshuffle is ~N²/4 ≈ 5×10¹⁵ sequence inversions — full attribution/transport is out of reach, and by the F-012 frontier (edge at K·η ≈ 30 at pilot scale) real pairs sit astronomically past the parameter-space validity edge. M3.2 measurements are therefore **divergence/recoverability measurements (M3.4-style), not full transport**; transport at scale waits on structured estimators (F-011's M5 item). (3) Pythia trains with Adam + warmup/cosine LR — pilot constants are SGD; only the order-of-η laws (K-4b/K-4d) carry over, not constants.
- **First measurement (E-009 candidate, to be pre-registered separately)**: checkpoint-matched divergence curves d_f(θ^A_t, θ^B_t) (cross-loss/KL on a fixed held-out Pile shard) and d_θ for P1/C1/J1 on a log-spaced ~12-checkpoint subset. Degenerate baselines: same-run adjacent-checkpoint drift (floor), cross-size gap (ceiling). Budget: ~12 ckpts × 6 runs × ~330MB ≈ 24GB staged downloads (P1 alone ≈ 8GB); forward evals only, no training.
- **Verification gates before E-009 counts**: (i) `step0(data-seed1) == step0(data-seed2)` bitwise/allclose (must hold; else D-013 is superseded and the pair moves to the homotopy track); (ii) `step0(weight-seed1) != step0(weight-seed2)`; (iii) `step0(data-seedN)` vs `step0(pythia-160m)` — classifies whether the main run joins P1's equivalence class (bonus pairs); (iv) optimizer/LR config read out of the repo configs, not assumed.
- **How to reverse**: if local hardware can't hold 160M checkpoint pairs, fall back to 14M standard pairs for tooling shakeout only (explicitly no 𝓒_perm claims); superseding entry either way.

## D-014 (2026-07-20) — ~~OPEN, pending principal sign-off~~ **CLOSED 2026-07-21 by D-015** (choice: (a)+(c) combined) — disposition of E-009's void init-only control (C1) after F-014
- **Status**: ~~OPEN~~ **CLOSED by D-015.** Recorded here so the finding and its options are durable; the choice is deliberately NOT made by the harness. Voiding a pre-registered arm and selecting its replacement are registration-level decisions (M3.2 pre-registration discipline), and picking the replacement *after seeing* the P1 curve is exactly the post-hoc freedom the E-009 registration forbids. Nothing downstream may cite C1 until this entry is closed.
- **Problem**: F-014 establishes that the `pythia-160m-weight-seed{1,2}` checkpoints at `step≥1` do not continue from their own `step0` inits — every run descends from the shared data-seed init, and continuity ratio = 1.000 at every reachable checkpoint. E-009's C1 arm therefore has no valid data beyond `step0`, and H-b/H-c have no control. What replaces it?
- **Facts constraining the options** (all from F-014 / E-009 phase 0b, none assumed):
  1. The defect is in the artifacts, not our metric — the instrument was independently verified and reproduces the recorded JSON bit-for-bit.
  2. P1 (order-only) is unaffected from `@4` onward; `@1` is void; the run is resumable.
  3. **J1 (`pythia-160m-seed1`/`-seed2`) has never been tested** and may carry the same defect. Testing it is ~2 GB read-only versus ~25 GB for a phase-1 relaunch.
  4. A true init-only control needs *same data order, different init*. Whether any public artifact supplies that is now an open question — the decoupled 160M repos were the candidate, and they do not.
  5. ws1/ws2 diverge from each other despite sharing an init from `step1`, by less than the ds pair does, contradicting D-013's record that weight-seed runs hold data order fixed. Their downstream difference is undetermined.
- **Options**:
  - (a) **Void C1 entirely; run E-009 as a single-arm order-only experiment** (P1 + J1), dropping H-b/H-c and re-registering only H-a. Cheapest, honest, but loses the order-vs-init contrast that motivated D-013's pair choice.
  - (b) **Seek a genuine init-only control elsewhere** (other PolyPythias sizes/seeds, or another public release) and re-register H-b/H-c against it. Preserves the scientific contrast; cost and even existence unknown until a survey is done.
  - (c) **Keep `C1@0` as a standalone init-distance reference point** (it is valid — three distinct inits, mutually 1.2475) and mark H-b/H-c REFUTED-as-unrunnable rather than replaced. Minimal spend, maximal honesty, weakest science.
  - (d) **Construct the control ourselves** at small scale (train two runs, same order, different init). Full control, but leaves the "real history pairs" premise of M3.2 and re-inherits pilot-scale caveats.
- **Recommendation to the principal (not yet a choice)**: gate on information before committing — run the J1 continuity audit first (cheapest-first; it may reduce the option set outright, since a J1 defect would put the whole decoupled-release premise of D-013 in question), then decide between (a) and (b). (c) is the fallback if a survey for (b) comes up empty.
- **How to reverse**: this entry is open; closing it requires a superseding decision naming the chosen option, the re-registration it implies for E-009's hypotheses, and whether D-013's pair selection itself is superseded.

## D-015 (2026-07-21) — Closes D-014: E-009 continues single-arm order-only (P1 + J1) with C1@0 kept as an init-distance reference; H-b/H-c refuted-as-unrunnable; external control pursued as a non-blocking track
- **Problem**: D-014 left open what replaces E-009's void init-only control (C1) after F-014.
- **Inputs (both gathered before this choice, per D-014's own recommendation)**:
  1. **J1 continuity audit (2026-07-21, phase 0b-J1)**: `pythia-160m-seed{1,2}` PASS G-v (cont = 5.63e-5 at step4 vs the weight-seed runs' 1.0000) and G-vi (pair distinct at every revision, rel d_θ ≈ 1.2474). The defect is confined to the decoupled weight-seed repos; J1 is usable. Quirk: `step1` ≡ `step0` in both seed repos (init copy) — `step1` already dropped from the grid.
  2. **Opus 4.8 survey (agent report, reviewed by principal)**: no public release provides a downloadable init-only pair (different init, verifiably fixed data order, early checkpoints) at LM scale. Closest: AllenAI "Signal and Noise" (arXiv:2508.13144; 10 init-varied + 10 order-varied OLMo-2 1B) — weights on internal Weka, only eval predictions on HF, late-step checkpoints only, order-fixity unverified. PolyPythias `weight-seed3` exists (full ladder) but is suspect under the same systematic fault and cannot pair with the defective ws1/ws2. Mode-connectivity/LMC releases are the inverse configuration (shared init, varied order) and vision-scale. Upstream cause best supported: **H2 — weight-seed runs launched from a shared/default init while the seed still drove the dataloader shuffle** (fits step1 identity, six distinct nonzero downstream distances, AND reconciles ws1≠ws2 despite "fixed data order"; precedent: Pythia 6.9B/12B init-config errata). Copied-upload hypotheses (H1/H3-pure) refuted by the six distinct pairwise distances.
- **Options**: as enumerated in D-014 (a)–(d).
- **Choice**: **(a)+(c) combined.** E-009 continues as a single-arm order-only experiment: arms P1 (order-only, clean from step4) and J1 (joint reference, cleared by the audit). `C1@0` is retained as a standalone init-distance reference (three distinct inits, mutually ≈1.2475 — a valid datum). **H-b and H-c as registered are REFUTED-AS-UNRUNNABLE on this artifact set** — not re-specified against seen data. C1 cells at t ≥ 1 are removed from the job list (no further spend on a void arm; recorded cells stay in the JSON). Amended registration logged in experiment_log E-009 BEFORE the resume runs; the only new gate (H-b′, J1-restricted joint dominance) is registered against J1 cells that do not exist yet, with form and tolerance frozen verbatim from the original H-b.
- **Why**: cheapest and fully honest; pre-registerable today without dependence on any unseen artifact; extracts everything the surviving arms can carry (H-a still gated on unseen floor/ceiling cells; J1 curve unseen). The order-vs-init contrast that motivated C1 is not abandoned — it is moved off E-009's critical path.
- **Parallel tracks (non-blocking, both outward-facing steps need user sign-off)**:
  - **(b)**: draft request to AllenAI for the signal-and-noise 1B seed-model weights; if ever obtained → continuity audit first, then a fresh pre-registration against its unseen curves. Watch their HF org. **Status 2026-07-21: SENT.** Drafted at docs/allenai_email_draft.md, sent by the user the same day to `davidh@allenai.org` (David Heineman, corresponding author). Availability was re-checked before sending — the paper claims release but only the eval dataset and code are public, so the ask stands. Now awaiting reply; no further action on this track until one arrives. If weights are offered: continuity-audit them (`src/e009_audit_checkpoints.py`) BEFORE any measurement, then pre-register against the unseen curves — skipping that audit is exactly the F-014 failure.
  - **Upstream report**: issue draft for EleutherAI saved at docs/upstream_issue_draft.md (includes the J1 step1≡step0 quirk and the seed-repos-clean result). Filing is a user action. **Status 2026-07-21: FILED as https://github.com/EleutherAI/pythia/issues/203 (posted as `shoyammishra`; cites open issue #198 as prior art). The HF cross-link post on `pythia-160m-weight-seed1` remains staged and unsent — deprioritized, blocked on an invalid local HF token.**
  - **(d)** (self-trained control, e.g. Pico-LM harness) held as fallback if the init-only contrast becomes load-bearing and (b) fails; would re-inherit pilot-scale caveats — deferred, M5 candidate.
- **D-013 status**: NOT superseded. The P1/J1 selection and the honest-limits analysis stand; only the C1 instantiation is voided, by an artifact defect (F-014), not by a flaw in the selection logic. `weight-seed3` is not adopted as a replacement (same suspect family).
- **Trade-offs**: loses the order-vs-init accrual contrast (H-c) at real scale for now — the K-4b/K-4d carry-over prediction stays untested there (noted in E-009; no supersession of the theory claims, which were never gated on H-c). H-b′ is strictly weaker than H-b (max over one arm instead of two); the distinction travels with any reported number.
- **How to reverse**: superseding entry if (b) lands (new control → new registration), if J1's phase-1 curve trips the pre-registered F-014 tripwire (J1 arm void → single-arm P1 only), or if EleutherAI corrects the weight-seed artifacts (C1 could be re-instantiated against fixed uploads).
