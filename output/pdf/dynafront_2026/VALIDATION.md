# Revised manuscript validation — 2026-09-05

## Numerical evidence

Read-only assertions recalculated the following from the original JSONs, checked rounding at the displayed precision, and checked the underlying pass/fail counts. No model evaluation, new seed, experiment, or change to a recorded output was involved.

| Manuscript quantity | Recorded source / check |
|---|---|
| Endpoint divergences 0.290461 / 0.290638 / 0.293055 / 0.306112 | E-009 and E-010 `cells` for P1/P2/P3/J1 at 143000 |
| Endpoint ratio column 0.9489 / 0.9494 / 0.9573 / 1.0000 | Recomputed against the single J1 denominator |
| Endpoint drift ratios 3.81 / 3.81 / 3.84 / 4.01 | Recomputed against E-009 `floor:142000-143000` |
| Maximum/minimum 1.009 and comparison threshold 2.0 | Endpoint readouts and E-010 registration |
| Drift 0.076275, 0.680225, 0.140831; size comparison 0.460360 | E-009 drift-anchor and ceiling cells |
| P1 at 16k: KL 0.312597, ratio 0.932, cosine 0.294 | E-009 `P1@16000`, `J1@16000`; E-011a P1 curve |
| P1 at 64k: ratio 0.943, cosine 0.877 | E-009 and E-011a |
| P1 ratio 0.993 at 128k, 0.949 at endpoint | E-009 ratios, recomputed |
| P2/P3 16k cosines 0.441 / 0.466 | E-011a pair curves |
| First sampled cosine ≥0.9: P1 128k, P2/P3 64k | Recomputed from complete E-011a curves, matches recorded `t_star`; descriptive threshold from registration |
| 48 × 2048 = 98304 positions; 12 usable divergence-grid checkpoints | E-009 metadata, excluding void step 1; zero direction at step 0 is omitted from cosine plot |
| Graft α = 0.25 / 0.5 / 1; parent +0.5 band = 3.001560 | E-011b frozen constants and JSON thresholds |
| 7/8 complete swaps invalid | E-011b T1 cells at α=1 |
| 23/35 evaluated graft readouts invalid | All JSON cells with `ce_h`; includes controls and second-anchor readouts, not 35 distinct hybrids |
| QKV share ≈87%, invalid at all tested nonzero strengths | E-011b `dtheta2_share`, QKV cells |
| LayerNorm full swap valid but degraded and distant from both parents | E-011b T1/T1r ln cells and flags |
| Table 2's nine relative errors and pass counts 0/3, 0/3, 1/3 | E-012 `phase0`, independently recomputed as absolute prediction error divided by measured KL |
| CE increases 0.0050 / 0.0337 / 0.0087 | E-012 `probe_scales` |
| ≤0.25 certification on all three directions; no family selected | E-012 thresholds and `I_Sigma` gate |
| ≈13% endpoint-gap error; ≈5.3× update underprediction | E-012 diagonal predictive-Fisher errors and measured/predicted ratio |
| Reproduction errors 1.4×10^-10 / 1.42×10^-10 | E-011b I3 / E-012 I_df; comparison is against the rounded recorded reference, not an independent scientific replicate |
| Full empirical diagnostic 0/3; Phase 1 unrun | E-012 diagnostic and empty `phase1` |

The 160M/410M names, nominal 300B training tokens, and 143000-step endpoint follow the recorded experiment design and Pythia assets. All plotted curve values are read directly from E-009/E-011a by `build_figure.py`. Shared-run structure comes from E-010's continuity and shared-initialization audit. The second Fisher-build pattern is documented in the E-012 experiment-log audit; it is not counted as another model replicate.

## Citation support

No references were invented or added. Existing bibliography metadata and the specific associated claims were checked on primary sources:

- [Biderman et al., Pythia](https://arxiv.org/abs/2304.01373): public trajectories/checkpoints across training and model sizes.
- [van der Wal et al., PolyPythias](https://arxiv.org/abs/2503.09543): seed, initialization/data-order, and trajectory-stability comparisons. The manuscript does not claim this earlier work discovered the present ratio/cosine result.
- [Kunstner et al., Limitations of the Empirical Fisher Approximation](https://arxiv.org/abs/1905.12558): empirical Fisher need not approximate the predictive Fisher or capture its second-order information.
- [Frankle et al., Linear Mode Connectivity and the Lottery Ticket Hypothesis](https://arxiv.org/abs/1912.05671): instability/stability to SGD noise, including data order; not a claim about the present LM cosine curves.
- [Rukhovich et al., Commute Your Domains](https://arxiv.org/html/2501.15556v1): local commutator-based training-order criteria, including bilingual LM training.
- [Sweeney, The Geometry of Sequential Learning](https://arxiv.org/html/2606.24993v1): local Lie-bracket order prediction for sequential learning; not endpoint recovery-cost estimation.

The telescoping and fixed-subspace minimization arguments were checked against K-1/P-1 in `docs/theory.md`. Their conditions remain explicit. No broad novelty/“first” claim is made and no exhaustive new literature search is represented as complete.

## Document checks

- LaTeX compilation: five total pages; main text ends on page 4, references start on page 5. No unresolved references/citations or overfull boxes in the final log.
- Visual QA: all five final pages rendered with Poppler and inspected; figure labels, annotations, table columns, footer, and text boundaries checked. Figure is vector PDF at 5.5-inch print width, with 7–8-point labels and separate ratio/cosine panels.
- Anonymity: source and extracted PDF text, PDF metadata, links, and attachments checked for repository/user names, local workspace paths, and credentials. The generic anonymous author block is produced by the template. Public cited authors and dataset/provider names are not submission-author disclosures.
- Venue: [DynaFront CfP](https://sites.google.com/view/dynafrontneurips26/call-for-papers) supports anonymous review and 4–5 pages excluding references/supplement. Footer names the workshop; supplied `.sty` and checklist template remain unchanged. Main-conference checklist instruction text has not been treated as an extra user command.
- Comparability: `src/` and `results/` remain unchanged against the committed baseline. The figure is a presentation-only change; no measured-number instrument or result version changes.

Research tests and training were not rerun because this is manuscript editing. Author review and live OpenReview requirements remain outside these document checks. No manuscript submission was performed.
