# Reviewer-style self-critique

## Strongest contribution

The paper distinguishes the development of output-divergence magnitude from alignment of the raw parameter-space endpoint displacement in completed language-model histories. P1's step-16k divergence is already 0.312597 nats while its final-direction cosine is 0.294; this supports the contrast independently of the joint-pair denominator. The same qualitative pattern occurs in the three correlated comparisons, with different timing. Explicit probe gates then prevent two invalid readouts from being presented as evidence for or against recoverability. This is a focused empirical/methodological workshop contribution, not a new general theory of irreversibility.

## Likely objection 1: Is the temporal dissociation informative beyond coordinate geometry?

The strongest objection is that a scalar output divergence and a raw Euclidean cosine need not agree even in an ordinary, well-behaved optimization trajectory. Neither a stable output-space direction nor the generation time of individual order defects was measured. Late alignment can partly follow from approaching the endpoint, and the checkpoint grid only coarsely localizes the chosen 0.9 crossing. The current results describe a dissociation; they do not identify its mechanism or establish a distinct dynamical phase.

The revision addresses this by making the absolute KL evidence visible, qualifying raw directions throughout, and stating the grid and endpoint-identity caveats. This improves the claim's precision but does not resolve the mechanism. A stronger later paper needs a certified function-sensitive alignment readout, robustness to coordinate/parameter-block weighting, and an account of how the observed timing relates to the learning-rate schedule and update dynamics.

## Likely objection 2: Are the sample and probe failures sufficient for the broader motivation?

Three pairwise comparisons from three runs sharing one initialization are correlated. The joint arm is one pair and the initialization-only control is unavailable. One training-distribution shard cannot establish downstream relevance or broad statistical generalization. Block grafts may simply disrupt co-adapted parameters, and the tested diagonal/empirical approximations are restrictive. Their negative results need not imply difficulty for other correction classes. The conditional subspace statement is elementary and lacks a certified metric application here.

The revision preserves every restriction, reports the formal graft outcome as inconclusive, and makes estimator certification the methodological finding. It cannot honestly convert this evidence into a universal recovery-cost claim. An anonymous reproducibility package with the frozen registrations, evaluation recipe, and raw readouts would also improve reviewability; the LaTeX ZIP alone is not that package.

## Experiments that would most strengthen a later main-track version

These are proposals, not measurements in this paper. Each requires its own frozen design before any new result is used.

1. **Certify functional geometry first.** Compare a metric carrying cross-parameter structure with direct predictive KL or another direct function-space readout across predeclared random, update, and order-gap directions. Use perturbation-amplitude sweeps and sampling-error checks to distinguish approximation error from estimator error. Include directions outside any low-rank represented span. Only after certification measure function-sensitive alignment and the projection/recoverability ladder.
2. **Expand the factorial evidence.** Add independent initialization groups with matched data-order contrasts and a continuity-audited initialization-only arm. Report uncertainty at the independent-run or initialization-group level, not by treating overlapping pairs as independent samples. Keep initialization and initialization-by-order interaction distinct.
3. **Test relevance and timing robustness.** Add independent evaluation shards and held-out downstream tasks; examine more checkpoint times where available, sensitivity to the descriptive cosine threshold, and another model size. These would test whether the observed dissociation survives beyond this shard, grid, and architecture scale.
4. **Attempt a valid intervention.** After registering an admissible loss region, test a narrowly scoped correction class (for example aligned or smaller-strength grafts) against do-nothing and declared-budget baselines. Measure both language-modeling validity and target-function proximity. An improvement at a known cost would connect the dynamical observation to actual correction; another failure should remain restricted to the tested class.

## Overall assessment

The revised submission has a clearer optimization-dynamics question and a defensible workshop scope. Its largest scientific gap is the missing bridge from raw endpoint orientation to functional geometry and attainable correction. That gap should remain the conclusion and research agenda, rather than being filled by stronger prose.
