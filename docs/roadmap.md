# Roadmap

Project: Optimization Path Integrability in Deep Neural Networks ("Palimpsest")
Started: 2026-07-17. Target venue: NeurIPS / ICML / ICLR (or explicit, well-argued abandonment — a negative verdict is a valid deliverable per the spec).

## M0 — Project scaffold ✅ 2026-07-17
Repo structure, CLAUDE.md, docs/, verbatim spec preserved, git initialized.

## M1 — Novelty verification (HARD GATE — highest priority, in progress)
No experiments and no compute until this gate resolves.

- M1.1 **Kill-scan** (in flight, delegated to Opus 4.8): targeted search for papers that already answer the core question — "rewrite optimization history without data replay," "SGD path-integrability," holonomy/curvature/non-commutativity of SGD updates, weight-space transport between training runs. Output: closest-work list + three probability estimates (already answered / seen as incremental / hidden overlap) + verdict.
- M1.2 **Exhaustive adjacent-literature map** (only if kill-scan verdict is "survives"): map all ~35 literatures named in the spec; per cluster: closest 3–5 papers, what they claim, precise delta to our question.
- M1.3 **Gate decision** logged in docs/decision_log.md: PROCEED / REFORMULATE (find the deeper surviving question) / ABANDON.

Deliverable: novelty section of the final report (items 1–4 of spec's Expected Output), written to docs/findings.md.

## M2 — Formal framework (after M1 gate passes)
- Formal problem statement: history space 𝓗, training map Φ, equivalence classes, transport operator T, budget constraints (no data replay, compute ≪ retraining), functional (generalization-based) success metric.
- Candidate theory: non-commutativity as Lie-bracket/holonomy obstruction; conservative-field test for SGD/momentum/Adam; Fisher/information-geometric transport; mode-connectivity equivalence.
- 3–5 competing hypotheses with distinguishing predictions (spec items 5–8).

## M3 — Experimental program (spec Phases 1–5, cheapest-first)
- M3.1 Pilot on tiny scale (TinyStories-class or small MLP/CNN) — exact pipeline smoke-tested before any larger run.
- M3.2 Existing checkpoint suites: PolyPythias / Pythia / OLMo. No pretraining from scratch.
- M3.3 Construct controlled history pairs (curriculum, optimizer, LR schedule, replay, augmentation) holding architecture/data/compute constant.
- M3.4 Trajectory-information measurements (what is recoverable from θ_T / checkpoint sequences).
- M3.5 Intervention: attempt history-to-history transport; measure generalization, not memorization.
- M3.6 If transport fails: characterize why; hunt impossibility results / necessary conditions.

## M4 — Ablations & analysis
Layer, depth, optimizer, initialization, curriculum, model size, parameter type, LR schedule. Key question: are some layers integrable while others are path-burned?

## M5 — Writing & adversarial review
- Full research report (all 15 spec output items) → docs/draft.md.
- Reviewer attack simulation (NeurIPS/ICML/ICLR), acceptance-probability estimate, highest-risk assumptions, strengthening recommendations.
- Non-technical summary → docs/report.md.

## Kill criteria (from spec, binding)
- M1 finds the core question already answered → do not continue; reformulate or abandon.
- Transport operator memorizes / needs retraining / toy-only → stop building editors; pivot to characterizing why repair fails.
