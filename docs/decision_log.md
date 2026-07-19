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
