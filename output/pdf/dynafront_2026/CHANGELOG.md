# Manuscript revision — 2026-09-05

## Narrative and title

- Adopted the requested title, **Persistent Data-Order Effects Precede Directional Alignment in Language Models**. Of the alternatives, A makes the measured timing contrast most visible. B's “rotating” could suggest a rotation mechanism that was not measured; C hides the key result; D is precise but longer. The abstract explicitly bounds “directional alignment” to raw parameter space and retains the probe results.
- Made temporal dissociation the first contribution and the main result: P1 at step 16,000 has output divergence 0.312597 nats, ratio 0.932, and raw endpoint-direction cosine 0.294. The absolute divergence supports the magnitude claim without depending on the single joint-pair denominator.
- Retained step-64k values, the descriptive first sampled 0.9 crossing at 128k, and the correlated P2/P3 timing differences. Distinguished sampled threshold crossings from precise stabilization times and from functional-direction lock-in.
- Reframed Section 4 around probe validity: neither graftability nor agreement in one selected perturbation direction automatically establishes functional recoverability.

## Removed or softened claims

- Replaced abstract's “95% of joint divergence” with the direct endpoint comparison, 0.290–0.293 versus 0.306 nats. Ratios remain descriptive, never a causal variance decomposition.
- Kept training drift distinct from numerical noise, the three pairs distinct from independent replicates, and initialization effects distinct from interactions.
- Corrected “23 of 35 interpolation cells” to the recorded tally of graft **readouts**, which includes controls and second-anchor readouts of the same hybrids. No data were changed and no new experiment was run.
- Limited the repeated Fisher-build statement to the documented order-gap pass/random fail pattern. Missing cross-parameter curvature remains a compatible explanation, not an identified cause.
- Retained that small CE changes do not certify the quadratic regime; no general Fisher failure, nonlinear irreversibility, successful transport, or computational lower bound is claimed.

## Space and figure

- Compressed the fixed-subspace proposition into the defining equation and a short minimization argument with all three restrictions: fixed subspace, fixed quadratic metric, no state-dependent reachability claim.
- Kept the swap-defect expansion and telescoping identity, with explicit attenuation/rotation/cancellation and the optimizer-state caveat. Added compact positioning against the existing six verified references.
- Rebuilt Figure 1 at its actual 5.5-inch print width. Left retains the recorded functional-divergence curves and drift-window markers; right uses separate aligned ratio and cosine panels. Labels at 16k come directly from JSON. There is no shared ratio/cosine ordinate and no new data, fit, or smoothing.
- Corrected the review notice in the manuscript preamble to identify DynaFront 2 at NeurIPS 2026. The supplied style file, margins, fonts, anonymity, and line numbering remain unchanged.

## Unresolved weaknesses

- One size, one shared initialization for order-only pairs, one joint pair, no valid initialization-only arm, and no held-out downstream evaluation.
- Raw alignment is coordinate-dependent and not functional-direction alignment. Its apparent timing depends on a sparse checkpoint grid and the descriptive 0.9 threshold.
- The theory is a framework, not an empirical attribution of Pythia's order defects or a validated recovery bound. Neither recovery probe certifies a positive recovery measurement.
- The submission still needs author review and any live OpenReview reproducibility/checklist requirements. The provided checklist template is not a completed checklist. See `REVIEWER_CRITIQUE.md` for prioritized future experiments.

## Quality control

Compiled main text remains four pages, with references on page 5. Final checks cover numeric readouts against the recorded JSONs, citation support, resolved references, overflow, vector-figure readability, PDF/source anonymity, and untouched experimental outputs. Details are in `VALIDATION.md`.
