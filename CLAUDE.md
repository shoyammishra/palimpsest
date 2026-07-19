# Palimpsest — Optimization Path Integrability in Deep Neural Networks

## Active Context
- Status: In Progress — M3.1 pilot ladder COMPLETE through the validity edge (E-000..E-007, F-007..F-012). Latest: E-007 CONFIRMED (F-012): the K-3.1 attribution overshoot is a **(K, η) frontier, not a K-wall** — ρ monotone in K with edge crossing K_c ≈ 606 at η = 0.05 (prefix family), and Θ(η)-suppressed at fixed full π (ρ = 1.63 → 0.88 → 0.49 as η halves twice); r < 1 everywhere (functional transport survives past the parameter edge — the E-003 dissociation is now a curve). Post-hoc candidate law ρ ≈ 0.033·K·η (±12%, 8 cells) — MUST be pre-registered (vary T / third η / other family) before any claim; inversion *structure* matters at matched K (prefix ρ = 0.32 vs block-shuffle 0.47). Prior rungs: E-006/F-011 (unbiased sampling closed at pilot scale; α ≈ 0.19 coherence exponent; truncation beats sampling), E-005/F-010 (O(K+T) chain), E-004/F-009 (Adam Θ(η)), E-003/F-008 (first transport), E-000..002/F-007. Harnesses: src/pilot_defects.py, src/pilot_transport.py, src/pilot_adam.py, src/pilot_transport_batched.py, src/pilot_sampling.py, src/pilot_edge.py.
- Current task: M3.2 scoping — PolyPythias pair selection (which run pairs make the cheapest first history-pair). Next experiment must pre-register the ρ = c·K·η edge law as a gated clause (vary T or third η). M2 leftovers deferred to M5 (K-3 constants, K-5b proof, Adam constant-tracking, structured estimators per F-011).
- Key files: src/pilot_defects.py, src/pilot_transport.py, src/pilot_adam.py, src/pilot_transport_batched.py, src/pilot_sampling.py, src/pilot_edge.py, docs/experiment_log.md (E-000..E-007), docs/theory.md (K-1..K-5, P-1), docs/design.md (v0.4), docs/decision_log.md (D-010..D-012), docs/findings.md (F-001..F-012), docs/professor_brief.md (external-facing brief, synthesis of F-001..F-012, refreshed 2026-07-19; citations #3/#4/#6 title-verified via arXiv abs pages)
- Open questions: does ρ ≈ c·K·η survive pre-registration at different T/η/family (the quantitative phase-boundary frontier)? What inversion-set geometry sets the family prefactor (Π_⊥ hook, M5)? Can structured estimators (truncation + control variate) beat enumeration where unbiased sampling failed (F-011)? Does α ≈ 0.19 coherence scaling hold at larger T? Homotopy generalization of K-1 to non-permutation pairs (needed for H4)? Which PolyPythias run pairs make the cheapest first history-pair (M3.2)?

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
