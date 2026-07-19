# Palimpsest — Optimization Path Integrability in Deep Neural Networks

## Active Context
- Status: In Progress — M3.1 first rung DONE. 𝓘 for pilots fixed (D-011: 𝓘₁ = {π, H₁ checkpoints}; 𝓚-aware correction as first operator). E-000/E-001/E-002 run and CONFIRMED (F-007): K-4b verified quantitatively (slope 2.00 SGD vs 1.00 momentum; constant ratio 1.0001, cosine ≈1), linear transport valid at pilot scale, K-1 verified in code to 3e-17. Harness: src/pilot_defects.py (numpy, manual backprop, self-checking).
- Current task: E-003 — first 𝓚-aware transport attempt (D-011 operator T(θ)=θ+ΣJ̄δ̄ via K-3.1) vs degenerate baselines (do-nothing / average / light fine-tune), functional metric per D-006; hypothesis to be logged in experiment_log BEFORE the run. Also queued: Adam-normalization η-sweep (K-4d check); M2 leftovers deferred to M5 (K-3 constants, K-5b proof).
- Key files: src/pilot_defects.py, docs/experiment_log.md (E-000..E-002), docs/theory.md (K-1..K-5, P-1), docs/design.md (v0.4), docs/decision_log.md (D-010, D-011), docs/findings.md (F-001..F-007)
- Open questions: does the Θ(η) momentum law survive Adam's normalization (K-4d)? Homotopy generalization of K-1 to non-permutation pairs (needed for H4)? Conic/state-dependent refinement of Π_⊥? Which PolyPythias run pairs make the cheapest first history-pair (M3.2)?

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
