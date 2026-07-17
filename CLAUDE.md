# Palimpsest — Optimization Path Integrability in Deep Neural Networks

## Active Context
- Status: In Progress — M1 gate CLOSED (D-009: REFORMULATE final, core = Q2 holonomy→transport-cost invariant; Q1 = machinery, Q3 = ablation axis). Both gating papers read end-to-end by principal (F-003, F-004); both kill-checks passed.
- Current task: M2 formal framework — first deliverable: choose and justify the discretization of the accumulated-holonomy functional 𝓚 (candidates (a)–(c) in design.md §2) and prove/argue it is well-defined off the quadratic regime; then sharpen TransportCost and the target inequality.
- Key files: docs/design.md (v0.2 — core question, definitions, binding positioning constraints §5), docs/findings.md (F-001..F-004), docs/decision_log.md (D-009), docs/research_spec.md
- Open questions: which 𝓚 discretization (bracket-norm sum vs target-projected vs ordered-exponential mismatch)? What is the minimal side-information class 𝓘 for the first transport pilots? Which PolyPythias run pairs make the cheapest first history-pair (M3.2)?

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
