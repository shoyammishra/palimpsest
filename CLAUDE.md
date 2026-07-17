# Fable — Optimization Path Integrability in Deep Neural Networks

## Active Context
- Status: In Progress
- Current task: M1 novelty verification — kill-scan literature review (delegated to Opus, in flight); on return, write verdict to docs/findings.md and decide proceed/reformulate/abandon
- Key files: docs/research_spec.md, docs/roadmap.md, docs/design.md, docs/decision_log.md, docs/findings.md
- Open questions: Has "rewriting optimization history without data replay" / "SGD path-integrability" already been answered in the literature? (M1 gate — no experiments until resolved)

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
