# Palimpsest — Optimization Path Integrability in Deep Neural Networks

## Active Context
- Status: In Progress — M2 lemma ledger largely discharged (docs/theory.md): K-1, K-2, K-4b, P-1, K-5a PROVED; K-3/K-3.1 sketch; K-5b stated. Headline theory results: P-1 path-burned floor (target inequality exact for fixed-subspace access classes, F-006), momentum order-defects Θ(η) vs SGD Θ(η²) (F-005, derives Sweeney 2606.29554, seeds H4), 𝓚 measurable from a single run (Cor K-3.1).
- Current task: M2 remainder — choose minimal side-information class 𝓘 for first pilots (D-011; candidate ladder in design.md §2.5), then M3.1 pilot design (instrumented tiny run + branched-replay validation gate per §2.2); deferred to M5: K-3 constants, K-5b containment proof, Adam constant-tracking.
- Key files: docs/theory.md (proofs K-1..K-5, P-1), docs/design.md (v0.4 — §2 definitions, §2.4 exact inequality, §2.5 ledger), docs/decision_log.md (D-009, D-010), docs/findings.md (F-001..F-006), docs/research_spec.md
- Open questions: minimal 𝓘 class (D-011 pending)? Homotopy generalization of K-1 to non-permutation pairs (needed for H4)? Conic/state-dependent refinement of Π_⊥? Which PolyPythias run pairs make the cheapest first history-pair (M3.2)?

## Conventions
- Read-first order: CLAUDE.md → docs/roadmap.md → docs/design.md → whatever Active Context points at.
- docs/research_spec.md is the verbatim source-of-truth spec. Never edit it; supersede via decision_log entries.
- Hard gate: no experiment, no compute spend, before the M1 novelty verdict is logged in docs/findings.md.
- Every experiment gets a hypothesis logged in docs/experiment_log.md BEFORE it runs. One variable at a time.
- Negative/impossibility results are first-class outcomes — the project does not depend on a repair operator existing.
- Delegation: Opus 4.8 subagents for deep research/reasoning; agents return reports, principal engineer reviews and writes all files.
- Decisions go in docs/decision_log.md (options, choice, why, how to reverse). If it's not written down, it doesn't exist.
- Small commits, often. `git diff --stat` before committing. Never leave the repo broken.
- Follow FABLE5_PROJECT_PROMPT.md for the full workflow (checkpoint protocol, loop, handoff template).
