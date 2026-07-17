# Notes (scratch pad)

## Binding constraints from the spec (quick reference)
- Forbidden framings: curriculum-learning paper, influence-function paper, model-editing method, benchmark, probing study, training-order-recovery paper.
- Known-not-novel (do not present as contributions): order affects learning; order is recoverable; influence is order-dependent; optimization is path-dependent.
- The project must NOT depend on a successful repair operator existing — non-integrability is an equally valid outcome.
- If transport memorizes / needs retraining / toy-only → pivot to characterizing why repair fails.

## Open threads
- Which side-information budgets for T are scientifically interesting? (nothing vs. a few checkpoints vs. gradient statistics) — decide after M1.
- PolyPythias: multiple seeds of Pythia training — check exactly which history coordinates vary across the suite (seed only? order?) before leaning on it for Phase 2.
- Residual stochasticity ω vs. history H: where to draw the line (is the data-order RNG seed part of H or ω?) — definitional choice with real consequences for the integrability claim.
- Possible cheap theory win: momentum/Adam as non-gradient flows — is there existing work formalizing "no potential exists" for Adam? Check during M1.
