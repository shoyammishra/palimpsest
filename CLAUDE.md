# Palimpsest — Optimization Path Integrability in Deep Neural Networks

## Active Context
- Status: In Progress — M3.1 fourth rung DONE. E-005 CONFIRMED (F-010): batched forward-JVP chain replaces tail replays with zero measurable quality loss (Δr ≤ 0.0005 at all severities; §2.2 validation gate passed per-sample at ~0.1%) — budget scaling now O(K+T): 3.8× / 10.4× / 38.6× retraining under D-012 rules (was 25× / 143× / 713×), compression 5.3–11.3×. Still > retraining everywhere (honest clause held); pair-defect evals (~2.4/inversion) now dominate ⇒ inversion sampling is the sole remaining lever for cheap transport, and E-003's cancellation ratio 0.12–0.22 warns uniform sampling is high-variance. Full-severity param overshoot (1.631) reproduced bit-faithfully ⇒ K-3.1 validity edge is attribution error, not transport error. Prior rungs: E-004/F-009 (Adam Θ(η) law, v-channel alone suffices), E-003/F-008 (first transport), E-000..002/F-007. Harnesses: src/pilot_defects.py, src/pilot_transport.py, src/pilot_adam.py, src/pilot_transport_batched.py (E-005, exact forward-over-reverse HVP).
- Current task: E-006 — inversion sampling (importance-weighted, unbiased estimator of ΣΔ̄; design must confront the cancellation-ratio warning); hypothesis in experiment_log BEFORE the run. Then queued: K-3.1 validity-edge probe (severity between w16 and full; r(η) at fixed π). M2 leftovers deferred to M5 (K-3 constants, K-5b proof, Adam constant-tracking).
- Key files: src/pilot_defects.py, src/pilot_transport.py, src/pilot_adam.py, src/pilot_transport_batched.py, docs/experiment_log.md (E-000..E-005), docs/theory.md (K-1..K-5, P-1), docs/design.md (v0.4), docs/decision_log.md (D-010..D-012), docs/findings.md (F-001..F-010), docs/professor_brief.md (external-facing novelty/generalizability brief, plain-language synthesis of F-001..F-009 — F-010 not yet reflected there; citations #3/#4/#6 title-verified via arXiv abs pages)
- Open questions: can inversion sampling alone bring budget under retraining (chain is fixed-cost now; pair defects ~2.4K evals are the floor to beat)? Homotopy generalization of K-1 to non-permutation pairs (needed for H4)? Conic/state-dependent refinement of Π_⊥? Which PolyPythias run pairs make the cheapest first history-pair (M3.2)? Does the Θ(η) Adam law hold with tails/transport (E-004 was two-step only)?

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
