# Optimization Path Integrability in Deep Neural Networks
### A novelty and generalizability brief

*Everything below is in plain language; every technical term is defined in one sentence at first use, and the strongest analogy for the whole project is in the first paragraph of the summary.*

---

## 1. Executive summary (one page, plain English)

Think of two cooks who are given the exact same ingredients and told to make the same dish. One adds the salt first and the acid last; the other reverses it. Even with identical ingredients and identical total effort, the two finished dishes are not the same — the *order* of the steps left a permanent mark. Now ask a harder question: given only the two finished dishes, can you turn one into the other by adding a small correction — a pinch of this, a squeeze of that — **without cooking the whole thing again from scratch?** And is there some way to *measure*, from the recipe alone, how hard that correction will be — or whether it is flat-out impossible because some flavor got "cooked in" and can no longer be separated out?

Training a neural network is exactly this kind of ordered process. A model is not built in one shot; it is nudged, step by step, by the data it sees, in the order it sees it, with a particular set of training choices (which optimizer, what learning-rate schedule, what curriculum, and so on). Change the order or the choices and you get a different final model, even with the same data and the same total compute. This is well known. Our project asks the question that the field has *not* answered:

> **Given two models trained from the same ingredients but with different histories, what is the minimum amount of computation needed to convert one into the other without retraining — and is there a measurable quantity, computed from the training path itself, that predicts that cost (including predicting when the cost is infinite, i.e., some information is permanently "burned in")?**

This is a question about a *law* and a *limit*, not about yet another editing trick. Two outcomes are both scientifically valuable and both publishable: either such a law exists (training is, in a precise sense, partly "rewritable"), or it provably does not past a certain point (training permanently destroys some information — an impossibility result).

**What is genuinely new.** The two closest published works stop one step short of our question, in opposite directions. One (Sweeney, ICML 2026) predicts *which order is better* before you train, using a purely *local* calculation — it never looks at two finished models and never talks about the cost of converting between them. The other (Yu, He, Goyal, and Arora, 2025) proves that a certain *restricted class* of conversion methods is impossible — but explicitly only for methods that ignore the training history, and the authors openly name the history-aware branch we take as unexplored. We sit in the gap between "predict the order" and "prove a narrow impossibility," and our central quantity is designed to *contain both of those results as special cases*.

**What we have shown so far, stated honestly.** We have proved two mathematical results (a law about how one popular training choice makes order-effects larger, and an impossibility floor that mathematically contains the prior impossibility result as a special case), confirmed the law numerically to four decimal places — including for the most popular modern optimizer family, where a plausible cancellation could have killed it and does not — and built a first working converter that closes 39%–85% of the gap between two models and beats every simple shortcut we compared against. The honest caveat: everything so far is at *tiny scale* (a network with a few hundred parameters), and the converter currently costs 25×–713× more than just retraining. We report that cost openly as the first measured point on the cost curve — not as a success, not hidden. Making the converter cheaper is a declared next step, not a solved one.

**Why it is broad.** "Training history" is not a niche. Data order is just one coordinate; the same framework covers curriculum design, choice of optimizer, learning-rate schedules, data replay, augmentation schedules, and the staged fine-tuning used in instruction-tuning, RLHF, and alignment. The downstream stakes touch machine unlearning and its regulatory verification ("can we prove data was really deleted?"), reproducibility, model merging and editing, continual learning, and using a trained model as an audit trail of how it was made.

---

## 2. The question in plain words

A trained model is usually treated as a finished object — a fixed list of numbers (its "weights," the internal dials the network has learned). But it was produced by a long sequence of tiny adjustments: *start → adjust → adjust → … → finish.* Two facts make the *sequence* matter. First, the adjustments do not commute: doing step A then step B lands you somewhere different from B then A. Second, because of that, two runs with the same data but a different order — or a different optimizer, schedule, or curriculum — produce genuinely different final models.

The field already knows all of this: that training order affects the result, that order can sometimes be inferred after the fact, and that a data point's "influence" depends on order. Those are established, and our project deliberately does **not** re-ask any of them. Instead it asks three tightly linked questions about the *finished* models:

1. **Conversion.** Can we take one finished model and edit it into the other with a small, targeted correction, *without replaying the training data* and *without spending anything close to the cost of retraining*?
2. **A cost law.** Is there a single quantity — measurable from the training path — that predicts how expensive that conversion is? We call this quantity, informally, the **accumulated order-sensitivity** of the path: how much the "which order" effects piled up along the way. (Its technical name in geometry is a mouthful; the plain idea is "how much the path curved," like the total drift you accumulate if you walk a closed loop on a curved surface and don't end up exactly where you started.)
3. **A hard limit.** Is some of the difference between the two models *permanently burned in* — impossible to remove by any correction of the allowed kind, at any budget? We call such information **path-burned**.

The scientific prize is a *quantitative relationship*: conversion cost is at least some function of the accumulated order-sensitivity, with a sharp boundary where cost becomes infinite (the path-burned regime). Whether that relationship holds, breaks, or holds only in some regimes — all of these are real findings.

---

## 3. What already exists — and what nobody has done

This is the heart of the novelty case. We compare, work by work, against the closest published research. For each we state precisely what it *did* and, just as importantly, what it *did not* do, so the delta is unambiguous.

### 3.1 The two works that come closest

**(a) Sweeney, "The Geometry of Sequential Learning" (ICML 2026), arXiv:2606.24993 — predicts the better order; never converts between finished models; never talks cost.**

This is an excellent and directly relevant paper, and we build on it. What it does: *before* you train, it computes a small local quantity at the starting point — essentially, "if I took one step of task A and one step of task B, how much does the order flip the result?" — and uses that to predict which training order will give a better final model, and to rank many possible curricula. It validates this across several settings (instruction tuning, preference tuning, language-model domains, and diffusion models).

What it does **not** do, verified by reading the full paper including its appendices:
- It is *local and prospective*: every quantity is computed at (or extremely near) the single starting point, and used to *plan* an order before training. It never takes two *already-finished* models and relates them.
- It never accumulates its quantity along the whole training path. In fact the paper's own experiments show its local quantity *loses accuracy* over long training (from 93% down to 65% predictive accuracy as the horizon grows) — which is direct evidence that the accumulated, path-long object we build is the missing piece, not a redundant one.
- The words "conversion," "cost of converting between endpoints," "reversibility," and "undo" never appear. Its only mentions of "cost" concern how expensive its *own planning calculation* is.
- **One passage must be named precisely so a reviewer cannot say we missed it.** In an appendix (E.8, "bracket-control"), Sweeney takes a *single* tiny correction step in the direction of its local quantity and shows it helps in a majority of a specific two-step test. This is the paper's one gesture toward "use this direction to fix things." It is still a single infinitesimal correction at *planning* time in a two-step toy setting — not conversion between two full, finished training histories. We cite it and state the difference explicitly.

*Our delta:* we accumulate the order-sensitivity along the *entire* path, we operate on *finished* models, and we tie the accumulated quantity to a *conversion cost* — three things this work does not attempt.

**(b) Yu, He, Goyal, and Arora, "On the Impossibility of Retrain Equivalence in Machine Unlearning" (2025), arXiv:2510.16629 — proves an impossibility, but only for history-blind methods, and names our branch as unexplored.**

This is a strong theory paper and the one most likely to make a reviewer ask "hasn't the impossibility already been proven?" We read it end to end, including its proof appendix. What it proves: if you have two models trained on the same data in different orders, and you try to "unlearn" (remove) some data using a rule that is **local** (the rule looks only at gradients of the data being removed) and **history-blind** (the same fixed rule is applied to both models regardless of their different histories), then the two models drift *further apart* the more you apply the rule — so no single such rule can reconcile both to what full retraining would have produced. They confirm this empirically on real models from 1 to 14 billion parameters.

Why our work is not blocked by this — verified line by line against their text:
- **They fence out our case by name.** The paper explicitly states it does *not* address methods that use extra information about the retained data, that modify training to enable later editing, or that assume stronger access. Our methods are *history-aware by construction* — they are told the history difference and use it — which the authors place outside their theorem.
- **They name our exact branch and leave it unexplored.** The paper frames its result as a "pick at most two of three" trade-off among (i) history-independence, (ii) matching the retrained model, and (iii) using only local rules. They give up (ii). We give up (i) — we build a rule that *knows and uses* the history. The authors state this branch and do not pursue it. Our central question is literally "how much budget makes that third option affordable."
- **Their result becomes a special case of ours.** In their proof, the part of the model difference that lies "outside the reach" of a history-blind rule — that is, outside the fixed set of directions the rule's updates are able to move the model in — can never be corrected — a clean, provable instance of information being permanently stuck. Our own impossibility result (Section 4) is built to *contain* exactly this as one instance, by generalizing "outside the reach of the rule" to "outside the declared information and budget the method is allowed to use." Their divergence is driven specifically by *un*learning (which pushes models apart); a conversion method is not doing that, so nothing in their machinery forbids it — provided we honestly declare our extra information and show our compute is far below retraining.
- **We use their yardstick.** They measure model difference by *behavior on test inputs*, not by comparing raw weights. We use the same behavioral yardstick — comparing what models *do*, not their internal numbers — because raw weights can differ for meaningless reasons (a network can be internally rearranged without changing its behavior at all).

*Our delta:* they prove a limit for history-*blind* methods; we ask, and begin to answer with both a positive construction and a matching lower-bound floor, what history-*aware* methods can achieve and at what cost — the vertex of their own trade-off they left open.

### 3.2 Other works in the neighborhood

**Xu (2026), "Low-Dimensional Execution Manifolds in Transformer Learning Dynamics," arXiv:2602.10496 — describes the geometry of training paths on toy models; no conversion, no cost.** This work studies the geometry of training trajectories of small toy transformers on modular-arithmetic tasks, including measuring how much update order matters, and borrows the language of "integrable vs. non-integrable" dynamics. It is descriptive: it measures order effects, but builds no conversion method and states no cost law. In our framework its measured per-step order effect is one ingredient; we accumulate those ingredients along the path and tie the total to conversion cost.

**Sweeney (2026), "Optimizer Memory Makes Shuffle Order a First-Order Source of Fine-Tuning Noise," arXiv:2606.29554 — observes empirically that a common optimizer makes order effects bigger; we derive it.** This separate short work reports, empirically, that when you train with a modern optimizer that keeps a running "momentum" (a memory of recent adjustments), the order in which data arrives becomes a larger source of variation than with plain training. This is exactly one of our predictions — and we *derive* it from first principles as a two-line consequence of our framework, with the precise size of the effect, and we then confirmed that derived size numerically to four decimal places — first for pure momentum, and subsequently for the full Adam-style optimizer including its normalization step, which is the actual setting of this work's observation. When a framework reproduces someone else's empirical observation as a clean corollary and pins down its magnitude, that is evidence the framework is carving the problem at a real joint.

**Goodbrake (2023), "Unnatural Algorithms," arXiv:2312.04739 — a different classification of optimizers.** This work sorts optimizers by a different mathematical property (roughly, how gracefully they behave when you rescale the problem). It is a neighbor because it also classifies optimizers, but along a different axis than ours (reversibility / order-sensitivity). We distinguish the two explicitly rather than conflating them.

**Rukhovich, Podolskiy, and Piontkovskaya (2025), "Commute Your Domains: Trajectory Optimality Criterion for Multi-Domain Learning," arXiv:2501.15556 — an earlier version of the local order quantity.** This work introduced the same local order-quantity that Sweeney later used, as a descriptive criterion. We cite it as the earlier appearance of that local building block, to be transparent about lineage — but it, too, is local and prospective, with no accumulation, no conversion, and no cost.

**PolyPythias (ICLR 2025), arXiv:2503.09543 — a ready-made testbed, not a method.** This is a public collection of 45 training runs that vary the random seed and the data order over a fixed dataset. It is perfect *infrastructure* for us — a set of same-data, different-history model pairs to test on at real scale — but it only studies stability; it builds no conversion method and states no cost law. We plan to use it as our scale-up testbed.

**TracIn (an influence-tracing method) and the NTK / "lazy" training regime — the assumptions we deliberately step outside.** TracIn, a popular method for tracing a data point's influence, *assumes* that order does not matter (that steps commute); we treat that assumption as a hypothesis to stress-test rather than accept. The "lazy" or NTK regime is a special training regime in which order genuinely does not matter — but there the whole question is trivial. So any positive conversion result we claim must be demonstrated *outside* that easy regime, or it is folklore rather than a finding. This is a discipline we impose on ourselves.

### 3.3 Adjacent fields, and why we are not one of them (one line each)

- **Machine unlearning** removes specific data from a model — one special case of our more general "convert any history into another" (and the setting of the impossibility paper above).
- **Model editing (e.g., ROME, MEMIT)** surgically changes specific facts a model knows. It targets facts; we target *histories*, and ask about a cost law and a hard limit, not a single edit recipe.
- **Task arithmetic / model merging / "model soups"** add or average whole models to combine skills — heuristics for combining endpoints that never ask what conversion *costs*, when it is impossible, or use the training path.
- **Linear mode connectivity** studies whether two trained models can be joined by a low-loss path in weight space — the *geometry between endpoints*, not the *cost and limits of moving between them using the history*.
- **Influence functions** estimate how a single data point changed the model — a per-point sensitivity, not conversion between full ordered histories.
- **Continual learning (e.g., EWC)** is a *training-time* remedy against forgetting; we ask, *after the fact*, what got permanently burned in and what can still be corrected.
- **Curriculum learning** chooses a good data order up front; we convert between the *results* of different orders. And **training-order recovery** infers what order was used — a known-solved "what was the order," not our "can the result be rewritten, and at what cost."

### 3.4 Summary comparison table

Columns are the specific capabilities and questions that define our contribution. "Local only" means the method computes at a single point rather than accumulating along the whole path. "Finished-model conversion" means it takes two already-trained models and edits one toward the other. "Cost law" means it relates a measured path quantity to the compute needed to convert. "Impossibility / hard limit" means it identifies when conversion is impossible at any budget. "History-aware" means the method is allowed to use the training history difference. "Beyond toy scale" means demonstrated past small toy models.

| Work | Accumulates along the *whole* path | Finished-model conversion | Cost law (path quantity → conversion cost) | Impossibility / hard limit | History-aware method | Beyond toy scale |
|---|---|---|---|---|---|---|
| Sweeney, ICML 2026 (2606.24993) — order prediction | No (local, prospective) | No | No | No | (planning only) | Yes (large models) |
| Sweeney 2606.29554 — optimizer-memory noise | No | No | No | No | n/a (observation) | Yes |
| Xu 2602.10496 — trajectory geometry | No | No | No | No | No | No (toy) |
| Yu/He/Goyal/Arora 2510.16629 — unlearning impossibility | No | No | No | **Yes, but only for history-blind rules** | **No (explicitly)** | Yes (1–14B) |
| Rukhovich 2501.15556 — local order criterion | No | No | No | No | No | Partial |
| PolyPythias 2503.09543 — testbed | n/a (dataset) | No | No | No | n/a | Yes (dataset) |
| Model merging / editing / continual learning | No | Edits endpoints, not histories | No | No | No | Yes |
| **This project** | **Yes** | **Yes (first operator built)** | **Yes (the central claim; first data point measured)** | **Yes (proved floor; contains the row above as a special case)** | **Yes (by construction)** | **Not yet — tiny-scale pilot; scale-up is the declared next step** |

The pattern is the point: individual capabilities exist scattered across the field, but **no prior work occupies the combination** — accumulate along the path, convert between finished models, tie a measured path quantity to conversion cost, and pin down the hard limit — and the two nearest works explicitly stop just short of it, one on each side.

---

## 4. What we have already shown (proved and measured, with scale caveats)

We separate what is *mathematically proved* from what is *measured in code*, and we attach the scale caveat to every number.

### 4.1 Proved on paper

**A law for how one common training choice enlarges order-effects.** Plain optimizers ("plain gradient descent") produce order-effects that shrink quadratically as you take smaller steps — halve the step size and the order-effect drops to roughly a quarter. We proved that optimizers with *momentum* (a running memory of recent updates — the default in most modern training) produce order-effects that shrink only *linearly* — halve the step size and the effect only halves. In plain terms: **momentum makes training order matter more, by a predictable amount, and the memory of order lives in the optimizer's internal state and leaks into the model over time.** This exactly reproduces, from first principles, the empirical observation reported by the separate 2026 work above — and gives its precise size. We also argued (at sketch level, not yet a full proof) that the same conclusion survives for the most popular modern optimizer, which adds a further normalization step — and that argument now has direct numerical confirmation (next section), including its sharpest consequence: the normalization memory *alone*, with momentum switched off, is enough to produce the stronger order-sensitivity.

**An impossibility floor that contains the prior impossibility result.** We proved that for any conversion method restricted to move the model only within a *fixed set of directions* (think of a method allowed to push only north–south, never east–west), a portion of the difference between two models can *never* be removed, at *any* budget — and we identified the correct measure of that permanent residual (subtly, a projection *distance*, because allowed motion can partly compensate for effects in forbidden directions, so the naive answer is wrong). Crucially, the Yu/He/Goyal/Arora impossibility comes out as *one special case* of ours, when the fixed direction-set is the one their history-blind rules are confined to — making a published theorem fall out as a special case is exactly the "subsumes prior work" result reviewers reward. Honest limit: this is proved for *fixed*-direction methods; realistic methods (fine-tuning a nonlinear network) can change their reachable directions as they go, and extending the floor to that case is open.

**The conversion quantity is measurable from a single run.** We showed (at sketch-level rigor, with the constants deferred to the full write-up) that the quantity predicting the conversion gap can be computed from *one* model's training checkpoints alone (checkpoints are saved snapshots of the model taken at intervals during training) — no need to retrain intermediate versions. This is what makes the whole program a practical instrument rather than a thought experiment.

### 4.2 Measured in code (tiny scale — stated up front)

All experiments below are on a deliberately tiny network (a few hundred parameters), in high-precision arithmetic, with fixed random seeds, as a *smoke test of the exact pipeline* before any expensive run. None of these are large-scale claims.

- **The momentum law held to four decimal places.** The measured shrink-rates matched the theory (2.00 for plain, 1.00 for momentum, where theory predicts exactly 2 and 1), and the *direction and size* of the effect matched the derived formula almost exactly (agreement to better than one part in a thousand). The framework's sharpest prediction survived first contact intact.
- **The law survives the most popular modern optimizer.** The Adam family adds a normalization step that could, in principle, have cancelled the momentum effect; we tested this directly. It does not cancel: measured shrink-rate 1.0014 where theory predicts exactly 1, with size and direction matching an independently computed prediction to better than one part in ten thousand. Sharper still: with momentum switched *off* entirely — a configuration where plain theory says the effect should vanish to second order — the optimizer's *other* internal memory (the running record of gradient sizes used for normalization) by itself reproduced the first-order sensitivity, exactly as our sketch argued. Every optimizer class we have tested (plain, momentum, Adam-family) now lands where the theory says it should.
- **A bookkeeping identity held to machine precision.** An exact accounting identity the whole method rests on matched the true value to roundoff error (about 15 correct digits) over a 21-swap reordering.
- **A first working converter.** Using only a declared, limited set of side-information (the history difference plus one run's checkpoints), our converter reduced the behavioral gap between two models to **14.5% of the original gap for a mild reordering, 26.5% for a moderate one, and 60.8% for a full scramble** — i.e., it closed 39%–85% of the gap. The gap-reduction got monotonically harder as the reordering got more severe, exactly as the theory predicts. It beat *every* simple shortcut we compared against (for example, just averaging checkpoints, or blindly fine-tuning) at every severity level.

### 4.3 The honest cost caveat (reported, not hidden)

The converter above currently **costs 25× to 713× more than simply retraining** the second model. We report this openly as the *first measured point on the cost curve*, not as a success: at this scale, our converter does not yet beat retraining on cost for anything. We already know *why* (the current implementation does far more repeated work than necessary) and we have declared the specific efficiency improvements to try *before* making any cost claim. We also observed a scientifically useful failure at the most severe reordering: the converter overshot in raw-weight terms while still closing the behavioral gap — direct confirmation that raw-weight distance is the wrong yardstick and behavior is the right one, and a precise localization of where our current approximation starts to break down.

**Framing discipline we hold ourselves to:** *possible is not the same as cheap*; a cost that exceeds retraining is a data point about the phase boundary, not something to bury; and an impossibility result is a first-class outcome, not a failure.

---

## 5. Why this is broad, not narrow

A reviewer's fair worry is "this is a paper about data ordering." It is not. Data order is one coordinate of a much larger object — the **optimization history** — and the framework is built to be generic in three independent ways.

**Broad in what counts as "history."** The same machinery applies to: the order data is seen; the curriculum (easy-to-hard vs. random); the choice of optimizer; the learning-rate schedule; data replay schedules; augmentation schedules; and the multi-stage pipelines used in modern instruction-tuning, RLHF, and alignment (pre-train, then fine-tune, then align — each a stage whose order and content are a "history"). Any two runs that share ingredients but differ in *how* they were combined are a valid pair for our question.

**Broad in optimizer and architecture.** The theory is not tied to a specific network shape or a specific optimizer. In fact, one of our deliverables is precisely that results *stratify by optimizer class* — the momentum law above is the first proved instance of an optimizer-dependent order-sensitivity, and the plain → momentum → Adam-family chain is now numerically confirmed end to end at pilot scale. "Which optimizers make history more reversible, and which burn it in permanently" is itself a general, testable classification, not a side note.

**Broad in downstream stakes.** Each of these is a real problem the framework speaks to directly:

- **Machine unlearning and regulatory verification.** When a regulation requires that a user's data be deleted, can deletion be *verified*, and is the data's influence actually *erasable* — or is some of it path-burned? Our impossibility floor is a direct tool for answering "this cannot be removed by methods of this class at any budget."
- **Reproducibility.** How much does run-to-run history variation actually change a model's behavior, and can that variation be corrected *after the fact* instead of by re-running? Our converter and cost law address exactly this.
- **Model merging and editing.** Why do some model edits and merges work while others fail? A cost/impossibility law over histories offers a principled explanation of which combinations are reachable and which are not.
- **Continual learning.** When a model learns tasks in sequence, what gets permanently burned in versus what remains correctable? This is the path-burned question applied to sequential tasks.
- **Training as an audit trail.** What does a finished model reveal — or provably fail to reveal — about how it was made? This connects the work to provenance and accountability of deployed models.

In short: the *object* (optimization history) is universal to how every modern model is made, the *method* is optimizer- and architecture-generic, and the *stakes* span several active subfields. Data order is the entry point, not the ceiling.

---

## 6. Honest risks, and how we are managing them

Professors and reviewers respect a candid risk section, so here is ours, with mitigations.

- **Risk: reviewers call it incremental.** Our own internal novelty audit put this as the single largest risk for the *original* framing (the one about training order alone). We addressed it *structurally*, not cosmetically, by reformulating the core around the cost-and-limit law — the one thing that provably subsumes the nearest prior works as special cases. An incremental paper cannot contain a competitor's theorem as a special case; ours is built to.
- **Risk: the core question is secretly already answered.** We estimated this as low but nonzero and treated the novelty check as a hard gate before spending any compute: we read the two gating papers end to end and logged exactly what each does and does not do (Section 3). We will re-run a targeted check for new preprints before the large-scale phase, since a fast-moving follow-up could close the gap.
- **Risk: hidden overlap we haven't found.** Real and nonzero. Mitigation: the end-to-end reads above, an explicit map of the adjacent fields (Section 3.3), and preemptive citation of the earlier local-quantity work so lineage is transparent.
- **Risk: results are tiny-scale.** True today. Everything measured is on a few-hundred-parameter network. Mitigation: the tiny scale was a deliberate cheapest-first smoke test of the exact pipeline; the scale-up path is defined and uses the public PolyPythias runs so we do not pretrain from scratch.
- **Risk: the converter is currently far more expensive than retraining.** True and reported openly (Section 4.3). Mitigation: we have identified the specific inefficiencies and declared the improvements to make *before* any cost claim; and even if conversion stays expensive, an impossibility/cost-floor result is itself a valid contribution.
- **Risk: the impossibility floor is proved only for a restricted method class.** True. Extending it from fixed-direction methods to methods that adapt their reachable directions is open. Mitigation: it is stated as open, and the restricted case already contains a published theorem as a special case, which is a meaningful result on its own.

---

## 7. Why this fits a top-tier venue

Three features match what NeurIPS, ICML, and ICLR reward:

1. **A candidate new quantitative law**, not a new gadget: a relationship between a measurable property of the training path and the cost of converting between finished models — designed to contain two recent results as special cases.
2. **Impossibility results as a first-class outcome.** The project does not live or die by a working converter. A proved limit ("this information is permanently burned in relative to methods of this class") is publishable, and top venues actively value clean impossibility results.
3. **An optimizer classification tied to reversibility** — a general, testable dichotomy (which training choices make history correctable vs. permanent), with the first instance already proved and numerically confirmed.

The area is demonstrably active: our two nearest neighbors are an ICML 2026 paper and a 2025 theory paper, and we occupy a branch that one of them *names as unexplored*. That is close to the ideal position for a top-tier submission — adjacent to hot work, clearly distinct from it, and with both a positive law and an impossibility result as possible outcomes, either of which stands on its own.

---

## 8. References

1. Sweeney, "The Geometry of Sequential Learning: Lie-Bracket Prediction of Transfer Order," ICML 2026. arXiv:2606.24993.
2. Yu, He, Goyal, Arora, "On the Impossibility of Retrain Equivalence in Machine Unlearning," 2025. arXiv:2510.16629.
3. Xu, "Low-Dimensional Execution Manifolds in Transformer Learning Dynamics: Evidence from Modular Arithmetic Tasks," 2026. arXiv:2602.10496.
4. Sweeney, "Optimizer Memory Makes Shuffle Order a First-Order Source of Fine-Tuning Noise," 2026. arXiv:2606.29554.
5. Goodbrake, "Unnatural Algorithms," 2023. arXiv:2312.04739.
6. Rukhovich, Podolskiy, Piontkovskaya, "Commute Your Domains: Trajectory Optimality Criterion for Multi-Domain Learning," 2025. arXiv:2501.15556.
7. PolyPythias, ICLR 2025. arXiv:2503.09543. *(A public suite of 45 training runs varying seed and data order over fixed Pythia data.)*
8. TracIn (influence-tracing method that assumes order-independence), NTK / "lazy" training regime, and the adjacent fields named in Section 3.3 (machine unlearning; model editing incl. ROME/MEMIT; task arithmetic / model merging / model soups; linear mode connectivity; influence functions; continual learning incl. EWC; curriculum learning) — cited here as neighboring literatures we delimit against; specific canonical references to be attached in the full paper.
