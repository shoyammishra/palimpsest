# DRAFT — request to Ai2 for the signal-and-noise 1B seed-model weights (NOT SENT)

Status: draft only. Sending is an outward-facing action and I have no mail capability
in this environment — the user sends it. D-015 option-(b) track.

## Recipient

`davidh@allenai.org` — David Heineman, corresponding author on *Signal and Noise: A
Framework for Reducing Uncertainty in Language Model Evaluation*
([arXiv 2508.13144](https://arxiv.org/abs/2508.13144)).

Availability check done 2026-07-21: the paper says they "train and release 20 1B
models, with 10 models trained varying the data order initialization and 10 varying the
random seed initialization," but the released artifacts we can find are the
[dataset](https://huggingface.co/datasets/allenai/signal-and-noise) and the
[code](https://github.com/allenai/signal-and-noise) — no public HF repo for those 20
seed models, and `snr/constants/models.py` names only the public OLMo-2 runs. So the
ask is real, not a search failure on our side. Consistent with the earlier Opus survey
(D-015 inputs).

Why these specifically: **10 runs varying only the random init** is exactly the
initialization-only control that does not otherwise exist publicly — the PolyPythias
`weight-seed` pair was our candidate and it is defective (F-014, filed upstream as
[EleutherAI/pythia#203](https://github.com/EleutherAI/pythia/issues/203)).

## Draft email

**Subject:** Request: 1B seed-model weights from Signal and Noise (init-seed arm)

Dear Dr. Heineman,

I'm an undergraduate researcher at BITS Pilani working on how much of a trained
model's final behaviour is determined by the *order* its data arrived in versus its
*initialization*. I'm writing to ask whether the 20 1B-5xC seed models from Signal and
Noise — specifically the 10 trained varying only the random seed initialization — are
available in any form, or could be shared.

The paper states these were trained and released, but I can only find the evaluation
dataset and the code released publicly; I may well have missed the model repository, in
which case a pointer is all I need.

Why I'm asking. I'm measuring functional divergence (symmetric KL between next-token
distributions) between pairs of runs that differ in exactly one factor, across training.
For the order-only arm I'm using the PolyPythias 160M `data-seed` pair, and the result
so far is that the order-only difference grows to roughly 95% of the divergence of a
pair differing in both order and initialization by mid-training — the initialization's
functional imprint appears to wash out while the order's persists. That claim needs an
initialization-only control to be worth much, and I don't have one: the PolyPythias
`weight-seed` pair, which would have been it, does not descend from its own published
`step0` init (audit filed upstream at https://github.com/EleutherAI/pythia/issues/203).
Your init-seed arm is the only set I know of that isolates that variable.

What would be most useful, in decreasing order: intermediate checkpoints for two or
more of the init-seed runs (the time course is the measurement); final checkpoints for
two or more of them (still gives one usable point); or simply a note on whether release
is planned, so I can decide whether to wait. Two runs is enough to start — I don't need
all ten.

I'd of course cite the paper and credit the models, and I'm happy to share the
divergence code and results back, or to run anything you'd find useful on them. I'd
also be glad to hear if you think the comparison is confounded in a way I haven't seen.

Thank you for the work and for how much of it you put in the open.

Best regards,
Shoyam Mishra
BITS Pilani
f20240176@pilani.bits-pilani.ac.in

## Notes before sending

- Check the arXiv page for a newer version / release note first — if the models went up
  after 2026-07-21, this email is unnecessary.
- Claims made in this email that must stay true: the ~0.95 order-only fraction (F-015,
  n=1 pair, train-distribution eval shard) and the F-014 audit. Both are logged.
- If a reply arrives with weights: continuity-audit them first
  (`src/e009_audit_checkpoints.py`), then pre-register against unseen curves before
  measuring. Do not skip the audit — that is exactly what F-014 was.
