# Decision Log

Format per entry: problem, options, choice, why, trade-offs, how to reverse.

---

## D-001 (2026-07-17) — Adopt FABLE5 workflow and docs structure
- **Problem**: New project needs a state/memory system a cold session can resume from.
- **Options**: (a) ad-hoc notes; (b) FABLE5_PROJECT_PROMPT.md structure (CLAUDE.md dashboard + docs/ + results/).
- **Choice**: (b).
- **Why**: User-mandated; makes the repo self-sufficient without conversation history.
- **Reverse**: restructure docs/, update CLAUDE.md pointers.

## D-002 (2026-07-17) — Novelty verification is a hard gate before any compute
- **Problem**: Spec's biggest risk is that the core question is already answered; experiments before that check waste compute and bias us toward defending the idea.
- **Options**: (a) start pilots in parallel with lit review; (b) hard gate — zero experiments until M1 verdict.
- **Choice**: (b).
- **Why**: Spec makes novelty "highest priority"; cheapest-first principle — a literature search is the cheapest possible information purchase; parallel pilots create sunk-cost pressure.
- **Trade-off**: slower start if the idea survives. Acceptable.
- **Reverse**: log a superseding decision if a pilot becomes necessary to *inform* the reformulation.

## D-003 (2026-07-17) — Preserve the original spec verbatim
- **Problem**: The spec is the contract; paraphrases drift.
- **Choice**: docs/research_spec.md holds it verbatim, never edited; changes happen via decision-log supersession.
- **Reverse**: n/a (append-only by convention).

## D-004 (2026-07-17) — Kill-scan delegated to Opus 4.8 subagent
- **Problem**: The M1.1 kill-scan is deep parallel research; burning principal context on raw searching is wasteful.
- **Options**: (a) principal does all searches inline; (b) delegate scan to an Opus subagent with full context, principal reviews and writes verdict.
- **Choice**: (b), per the standing delegation policy (Opus = default engineer for judgment-bearing research; agent returns a report, principal writes files).
- **Trade-off**: agent starts cold — mitigated by passing spec + framing in the task prompt.
- **Reverse**: rerun searches inline if the agent's report fails review.

## D-005 (2026-07-17) — Initialize git at project start
- **Problem**: "Small commits, often" requires a repo from day one.
- **Choice**: `git init` + scaffold commit at M0.
- **Why**: Never leave state uncommitted; enables honest history of the research process.
- **Reverse**: n/a.

## D-006 (2026-07-17) — Success metric is functional/generalization gap, not parameter distance
- **Problem**: Parameter-space closeness is confounded by permutation/scaling symmetries and is not the scientific target.
- **Choice**: All transport/integrability claims measured functionally (generalization behavior), with degenerate-strategy baselines mandatory.
- **Why**: Spec Phase 4 ("measure generalization rather than memorization"); instrument-skepticism principle.
- **Reverse**: superseding entry with justification if a parameter-space metric proves theoretically necessary.
