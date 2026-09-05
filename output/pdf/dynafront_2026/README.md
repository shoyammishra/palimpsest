# DynaFront 2026 workshop draft

Anonymous workshop paper: four pages of main content, with references beginning on page 5. The supplied NeurIPS 2026 style is copied without modification; no font-size or margin overrides are used. `paper.tex` is the main file for Overleaf or local LaTeX.

Venue checked 2026-09-05: https://sites.google.com/view/dynafrontneurips26/call-for-papers recommends 4–5 pages **excluding references and supplementary material** and requires double-blind review. This supersedes the initial working assumption of four pages including references. The workshop title is set with the supplied style's `workshoptitle` command; the style itself prints a generic conference footer in review mode.

The supplied `checklist.tex` is retained verbatim as `checklist_template.tex`, a reference template, and is **not included in the paper**. The workshop CfP/FAQ reviewed does not explicitly require it. Its main-conference instruction text is not treated as a new user request. If OpenReview requires a checklist, the author must complete the original questions before appending it; the untouched TODO template must not be submitted as a completed checklist.

## Build

From this directory, run `pdflatex -interaction=nonstopmode -halt-on-error paper.tex` twice. The prebuilt vector figure is included, so Python is unnecessary for compiling the paper. `build_figure.py` regenerates it from the repository's recorded JSONs using matplotlib. It only reads existing measurements and runs no experiment.

## Evidence map (internal; not part of the anonymous paper)

- Section 2: `docs/design.md`, K-1/P-1 in `docs/theory.md`, and `src/e009_divergence.py` for the exact shard and metric. The floor is explicitly conditional on a fixed quadratic metric and fixed subspace; no global nonlinear or computational lower bound is claimed.
- Section 3, Figure 1, Table 1: F-015/F-016/F-017 in `docs/findings.md`; `results/raw/e009_divergence_2026-07-20.json`, `e010_divergence_2026-07-22.json`, and `e011a_direction_2026-07-22.json`. Step 1 is omitted. All curve points are read directly, not refitted or remeasured.
- Section 4, grafts: F-018; `results/raw/e011b_graft_2026-07-22.json`. The formal verdict remains inconclusive for spatial recoverability.
- Section 4, Table 2: F-019; `results/raw/e012_recoverability_2026-07-22.json`. No estimator certified and no recovery ladder was run. Small CE changes are not presented as proof that higher-order terms vanish. The predictive Fisher is not equated with the observed-label loss Hessian.
- External bibliography metadata checked against the linked arXiv abstract pages on 2026-09-05. This is targeted citation verification, not an exhaustive new novelty review.

## Scope and remaining submission work

This is a workshop manuscript drawn from completed measurements, not the unfinished full M5 report. No experiments or measured-number code paths changed. The professor brief remains a separate artifact. Before external submission, the author should confirm the live OpenReview form's supplementary/checklist requirements and provide any required anonymous reproducibility attachment; no public code-release link is claimed here. Nothing has been submitted or published.

Validation: compiled PDF has five pages, main text ends on page 4 and references begin on page 5; final LaTeX references resolve; all pages rendered for visual inspection; style copy hash checked; source reviewed for identifying information and unsupported recovery claims. Research tests were not rerun because this change only writes a manuscript and plots already recorded results.
