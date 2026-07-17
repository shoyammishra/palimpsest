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
