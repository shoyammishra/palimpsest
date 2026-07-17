# Palimpsest — Optimization Path Integrability in Deep Neural Networks

## Active Context
- Status: In Progress
- Current task: M1.3 gate finalization — deep-read the two gating papers (Sweeney arXiv:2606.24993 ICML 2026; Yu/Arora arXiv:2510.16629), log deltas, then finalize the provisional REFORMULATE verdict (F-001, D-008) and pick among the three surviving questions
- Key files: docs/findings.md (F-001 verdict + surviving questions), docs/decision_log.md (D-008), docs/design.md (needs revision after gate finalizes), docs/research_spec.md
- Open questions: Does Yu/Arora's impossibility leave room for a history-aware transport operator? Does Sweeney's Lie-bracket work leave the holonomy→transport-cost invariant open? Which of the three surviving questions (F-001) becomes the core?

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
