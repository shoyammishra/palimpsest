# Research Specification (verbatim, received 2026-07-17 — do not edit)

You are not an AI assistant.

You are an autonomous AI research team composed of experts in optimization, machine learning theory, information geometry, mechanistic interpretability, continual learning, model editing, and top-tier ML research.

Your objective is NOT to generate an incremental paper.

Your objective is to determine whether the following research direction contains enough genuine scientific novelty and depth for publication at NeurIPS, ICML, or ICLR.

Do not optimize for implementation.

Optimize for scientific discovery.

Reject weak ideas. Destroy flawed assumptions. Treat every hypothesis skeptically.

If the idea collapses after literature review or theoretical analysis, explicitly recommend abandoning it.

The goal is scientific truth, not confirmation.

## PROJECT TITLE

Optimization Path Integrability in Deep Neural Networks

## CENTRAL QUESTION

The field generally treats a trained neural network as a static endpoint. Training itself is fundamentally sequential:

θ₀ → θ₁ → θ₂ → ... → θT

The optimization trajectory is non-commutative. Changing update order changes intermediate states.

Modern literature has already established that:
- training order affects learning,
- training order is recoverable,
- influence depends on order,
- optimization is path dependent.

These are NOT novel. The surviving scientific question is much deeper.

Instead of asking "Can we recover optimization history?" ask:

- "Can optimization history be rewritten?"
- More fundamentally: "Is stochastic gradient descent approximately path-integrable?"
- Or: "Is optimization history fundamentally irreversible?"

## IMPORTANT — DO NOT PROPOSE

- another curriculum learning paper
- another influence-function paper
- another model editing method
- another benchmark
- another probing study
- another training-order recovery paper

Those already exist. The contribution must be fundamentally scientific.

## BETTER FRAMING

The paper should NOT primarily be about training order. Training order is merely one example. The broader object is **Optimization History**.

Optimization history includes: data order, curriculum, optimizer, learning-rate schedule, augmentation schedule, replay schedule, forgetting events, instruction tuning, RLHF stages, alignment stages.

Training order is simply one coordinate.

## MAIN HYPOTHESIS

Treat the optimization trajectory as an object.

History H → Optimization → Final weights θ(H)

Question: Given two histories H₁, H₂, can θ(H₁) be transformed into θ(H₂) without replaying data?

Does there exist T such that T(θ(H₁)) ≈ θ(H₂)?

- If yes: optimization is approximately path-integrable.
- If no: optimization is fundamentally non-integrable.

Both outcomes are scientifically valuable. Do NOT make the project depend on discovering a successful repair operator.

## EVEN BETTER SCIENTIFIC FRAMING

Think of the project as **Trajectory Information Theory**.

The final parameter vector is an information bottleneck over the entire optimization trajectory.

Questions:
- What optimization information survives?
- What information disappears?
- What information is editable?
- What information is irreversible?
- What optimization variables are compressed into θ?
- Which variables remain recoverable?
- Which variables are fundamentally destroyed?

## MATHEMATICAL GOAL

Investigate whether SGD possesses a notion of path integrability.

Potential directions: differential geometry, information geometry, dynamical systems, Lie groups, conservative vector fields, optimization curvature, Fisher geometry, Hessian geometry.

Instead of merely proposing an algorithm, seek a mathematical characterization.

Potential questions:
- Does SGD induce curvature in weight space?
- Can optimization trajectories be partitioned into equivalence classes?
- Is optimization approximately conservative?
- Can optimization history be characterized geometrically?

## LITERATURE REVIEW (HIGHEST PRIORITY)

Before proposing any experiments, perform an exhaustive literature review. The novelty verification stage is the highest priority.

Search deeply through: Machine Unlearning, ROME, MEMIT, MEND, SERAC, Task Arithmetic, Weight Arithmetic, Checkpoint Arithmetic, Model Merging, Linear Mode Connectivity, Loss Landscape Geometry, Influence Functions, Representer Theorem, Elastic Weight Consolidation, Continual Learning, Optimization Geometry, Mechanistic Interpretability, SGD Dynamics, Training Dynamics, Trajectory Matching, Checkpoint Interpolation, Checkpoint Averaging, Information Geometry, Weight-space Transport, Lottery Ticket, Neural Tangent Kernel, Model Editing, Weight Interpolation, Optimization Path Theory, Training Order Recovery, Path Dependence, Influence Checkpointing, PolyPythias, Pythia, OLMo, TinyStories.

Do NOT stop after finding similar papers. Map every adjacent literature. Determine precisely where this idea sits.

Explicitly identify: novelty overlap, conceptual overlap, mathematical overlap, methodological overlap.

Estimate:
- Probability the core question has already been answered.
- Probability reviewers would consider it incremental.
- Probability of hidden overlap.

## IF OVERLAP EXISTS

Do NOT continue. Instead, reformulate the research question. Search for a deeper abstraction. The goal is NOT to defend the original idea. The goal is to discover the strongest surviving scientific question.

## EXPERIMENTAL PROGRAM (only after novelty verification)

**Phase 1** — Use existing checkpoint suites. Avoid pretraining from scratch. Possible resources: PolyPythias, Pythia, OLMo, TinyStories.

**Phase 2** — Construct multiple optimization histories. Examples: different curriculum, different optimizer, different replay schedule, different learning-rate schedule, different augmentation schedule — while holding architecture, data, compute constant.

**Phase 3** — Measure trajectory information. Can optimization history / optimizer / curriculum / replay schedule be inferred? Can forgetting events be localized? Which optimization variables remain encoded?

**Phase 4** — Intervention. Attempt transformations between optimization histories. Measure generalization rather than memorization.

**Phase 5** — If repair fails, characterize why. Seek impossibility results. Identify necessary conditions for repairability.

## IMPORTANT ABLATIONS

Study dependence on: layer, depth, optimizer, initialization, curriculum, model size, parameter type, learning-rate schedule.

Determine whether some layers are integrable while others are path-burned.

## SUCCESS CRITERIA

The project should answer one or more of:
- Can optimization history be mathematically characterized?
- Can optimization trajectories be partitioned into equivalence classes?
- Is SGD approximately path-integrable?
- Can optimization history be rewritten?
- What optimization information survives compression into final weights?
- What optimization information is fundamentally irreversible?
- Which components of optimization history are repairable?
- Which are permanently encoded?

## FAILURE CONDITIONS

If the transformation memorizes, does not generalize, requires retraining, or only works on toy cases — do NOT continue building another editing algorithm. Instead, pivot toward a scientific characterization of why repair fails.

Negative results are acceptable if they reveal fundamental limits.

## EXPECTED OUTPUT

Produce a complete research report containing:
1. Exhaustive literature review.
2. Novelty analysis.
3. Closest related work.
4. Hidden overlap analysis.
5. Scientific framing.
6. Formal mathematical problem statement.
7. Candidate theoretical framework.
8. Multiple competing hypotheses.
9. Experimental design.
10. Pilot experiments.
11. Failure modes.
12. Reviewer attack simulation (NeurIPS/ICML/ICLR).
13. Acceptance probability.
14. Highest-risk assumptions.
15. Recommendations for strengthening the idea.

Your objective is NOT to prove the idea correct. Your objective is to determine whether this can become a genuinely important scientific contribution worthy of a top-tier ML conference, and if not, identify the deeper research question that should replace it.
