# Palimpsest — Optimization Path Integrability in Deep Neural Networks

## Active Context
- Status: In Progress — M2 underway. First M2 deliverable DONE (D-010): 𝓚 discretization fixed as the transported-defect (exact telescoping) form, well-defined off the quadratic regime by construction; TransportCost sharpened with reachable-span projection; target inequality v0.2 stated (design.md v0.3 §2).
- Current task: M2 proofs — Lemmas K-2 (JVP estimator error, segment-local), K-3 (first-order schedule-independence of attribution), K-4 (Adam augmented-state lift), K-5 (formal reduction of the v0.2 inequality to Yu/Arora Thm 3.1); then pick the minimal side-information class 𝓘 for first pilots.
- Key files: docs/design.md (v0.3 — §2.1 𝓚 definition, §2.4 target inequality, §2.5 lemma ledger, §5 binding constraints), docs/decision_log.md (D-009, D-010), docs/findings.md (F-001..F-004), docs/research_spec.md
- Open questions: minimal 𝓘 class for first transport pilots? Homotopy generalization of Lemma K-1 to non-permutation pairs (needed for H4)? Conic/nonlinear refinement of Π_⊥? Which PolyPythias run pairs make the cheapest first history-pair (M3.2)?

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
