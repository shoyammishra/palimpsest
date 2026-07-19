# Palimpsest — Optimization Path Integrability in Deep Neural Networks

## Active Context
- Status: In Progress — M3.1 third rung DONE. E-004 CONFIRMED (F-009): the Θ(η) order-defect law survives Adam's nonlinear state readout — full Adam-class slope 1.0014 (frozen-θ predictor ratio 1.000046, cosine 0.9999999); normalization-only (β₁=0) slope 1.01 at small η with predictor passing — the v-channel alone makes order defects first-order (derived leading term added to theory.md K-4d, status stays SKETCH). H4 chain SGD Θ(η²) → EMA Θ(η) → Adam Θ(η) now empirical at pilot scale. Prior rung E-003/F-008 (transport possible, 25–713× retraining; K-3.1 validity edge K ∈ (218, 982] at η=0.05, T=64) unchanged. Harnesses: src/pilot_defects.py (E-000..E-002), src/pilot_transport.py (E-003), src/pilot_adam.py (E-004).
- Current task: budget compression (batched forward-mode JVP O(T) instead of O(KT) tail replays; inversion sampling by ‖δ̄‖) — required before any cost claim; hypothesis in experiment_log BEFORE the run (E-005). Then queued: K-3.1 validity-edge probe (severity between w16 and full; r(η) at fixed π). M2 leftovers deferred to M5 (K-3 constants, K-5b proof, Adam constant-tracking).
- Key files: src/pilot_defects.py, src/pilot_transport.py, src/pilot_adam.py, docs/experiment_log.md (E-000..E-004), docs/theory.md (K-1..K-5, P-1), docs/design.md (v0.4), docs/decision_log.md (D-010, D-011), docs/findings.md (F-001..F-009), docs/professor_brief.md (external-facing novelty/generalizability brief for the professor, 2026-07-19 — plain-language synthesis of F-001..F-009, updated 2026-07-19 to include F-009; citations #3/#4/#6 title-verified via arXiv abs pages)
- Open questions: can batched JVP + inversion sampling bring transport budget under retraining (the cheap-transport question)? Homotopy generalization of K-1 to non-permutation pairs (needed for H4)? Conic/state-dependent refinement of Π_⊥? Which PolyPythias run pairs make the cheapest first history-pair (M3.2)? Does the Θ(η) Adam law hold with tails/transport (E-004 was two-step only)?

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
