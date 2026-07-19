# Palimpsest — Optimization Path Integrability in Deep Neural Networks

## Active Context
- Status: In Progress — M3.1 fifth rung DONE. E-006 CONFIRMED (F-011, pre-registered negative): unbiased inversion sampling is CLOSED as the cheap-transport lever at pilot scale — the exact variance law relerr = A_p/√M (A² = Σ‖Δ̄‖²/p /‖ΣΔ̄‖² − 1) verified to ~1% median over 84 cells; budget-honest schemes need M*/K = 333/36/1.4 (uniform) and 359/31/1.1 (free proxy) at w4/w16/full, i.e. enumeration wins everywhere; at w4 every sampled M is worse than doing nothing. Trend quantified: coherence exponent c ~ K^−0.19, M*/K falls ~2 orders per severity step, oracle diagnostic dips under K at full severity (M*/K=0.81) ⇒ sampling reopens at scale, not here. Surprise (gate-free comparator): biased top-M truncation beats unbiased sampling at almost every M ⇒ structured estimators (truncate+control-variate) are the M5 candidate. Prior rungs: E-005/F-010 (O(K+T) chain), E-004/F-009 (Adam Θ(η)), E-003/F-008 (first transport, cancellation 0.12–0.22), E-000..002/F-007. Harnesses: src/pilot_defects.py, src/pilot_transport.py, src/pilot_adam.py, src/pilot_transport_batched.py, src/pilot_sampling.py.
- Current task: E-007 — K-3.1 validity-edge probe (severity between w16 and full at fixed η; and r(η) at fixed π — is the full-severity attribution overshoot an η-artifact or a K-wall?); hypothesis in experiment_log BEFORE the run. Then: M3.1 wrap-up synthesis + professor_brief refresh (F-010/F-011 not yet reflected there). M2 leftovers deferred to M5 (K-3 constants, K-5b proof, Adam constant-tracking, structured estimators per F-011).
- Key files: src/pilot_defects.py, src/pilot_transport.py, src/pilot_adam.py, src/pilot_transport_batched.py, src/pilot_sampling.py, docs/experiment_log.md (E-000..E-006), docs/theory.md (K-1..K-5, P-1), docs/design.md (v0.4), docs/decision_log.md (D-010..D-012), docs/findings.md (F-001..F-011), docs/professor_brief.md (external-facing brief, synthesis of F-001..F-009 — F-010/F-011 not yet reflected; citations #3/#4/#6 title-verified via arXiv abs pages)
- Open questions: where exactly is the K-3.1 validity edge (K ∈ (218, 982] at η=0.05 — and does it move with η)? Can structured estimators (truncation + control variate) beat enumeration where unbiased sampling failed (F-011)? Does α ≈ 0.19 coherence scaling hold at larger T (decides sampling-at-scale)? Homotopy generalization of K-1 to non-permutation pairs (needed for H4)? Which PolyPythias run pairs make the cheapest first history-pair (M3.2)?

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
