# Palimpsest — Optimization Path Integrability in Deep Neural Networks

## Active Context
- Status: In Progress — M3.1 second rung DONE. E-003 CONFIRMED (F-008): the D-011 operator transports — functional gap ratio r = 0.145 (44 inversions) / 0.265 (218) / 0.608 (982), monotone in severity, beats all degenerate baselines; but budget = 25–713× retraining (possible ≠ cheap — first phase-boundary data point), and at full permutation the attribution overshoots in parameter space (residual 1.63) while still cutting the functional gap — K-3.1 validity edge is at K ∈ (218, 982] for η=0.05, T=64. Harnesses: src/pilot_defects.py (E-000..E-002), src/pilot_transport.py (E-003).
- Current task: E-004 — Adam-normalization η-sweep (K-4d check: does the Θ(η) momentum law survive the nonlinear state readout?); hypothesis in experiment_log BEFORE the run. Then queued: budget compression (batched forward-mode JVP O(T), inversion sampling by ‖δ̄‖) before any cost claim; probe of the K-3.1 validity edge (severity between w16 and full; r(η) at fixed π). M2 leftovers deferred to M5 (K-3 constants, K-5b proof).
- Key files: src/pilot_defects.py, src/pilot_transport.py, docs/experiment_log.md (E-000..E-003), docs/theory.md (K-1..K-5, P-1), docs/design.md (v0.4), docs/decision_log.md (D-010, D-011), docs/findings.md (F-001..F-008), docs/professor_brief.md (external-facing novelty/generalizability brief for the professor, 2026-07-19 — plain-language synthesis of F-001..F-008; citations #3/#4/#6 title-verified via arXiv abs pages)
- Open questions: does the Θ(η) momentum law survive Adam's normalization (K-4d)? Can batched JVP + inversion sampling bring transport budget under retraining (the cheap-transport question)? Homotopy generalization of K-1 to non-permutation pairs (needed for H4)? Conic/state-dependent refinement of Π_⊥? Which PolyPythias run pairs make the cheapest first history-pair (M3.2)?

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
