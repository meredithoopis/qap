# Pre-registered hypotheses (frozen at the end of Week 1)

Copied from the proposal, Section 6 and Table 3. They are fixed before any evaluation is
looked at, together with `frozen_tasks.json` (task list + screening rule) and
`splits.json` (train / calibration / test by task).

The experiments test five claims: QAP improves functional substitution without hurting
unsubstituted imitation (H1); drop moves help exactly when part of a demonstration has no
counterpart (H2); asymmetry matters on irreversible steps (H3); functional costs are
necessary under substitution (H4); and labels and abstentions behave as the analysis
predicts (H5).

| | Hypothesis | Falsified if |
|---|---|---|
| H1 | QAP best at G2–G3 and multi-step; tied at G0–G1 | VLM chooser with the same executor matches QAP at G3 |
| H2 | drop moves help only with missing counterparts | gain at G0–G1, or none at G3–G4 |
| H3 | symmetric cost fails on irreversible steps | degradation not concentrated there |
| H4 | visual-cost methods degrade with grade | flat profile across G0–G3 |
| H5 | label errors lie in low-margin steps; false abstention ≤ α | errors uniform in margin; rate > α |

## Ablations and the claim each tests (proposal Table 2)

| Ablation | Claim | Prediction |
|---|---|---|
| IQE → symmetric (HILP ℓ2) | Prop. 5 | worse on irreversible steps only |
| d_θ → visual cosine | Prop. 6 | degrades with grade |
| Drop moves on/off | H2 | no gain G0–G1; gain G3–G4 |
| Calibrated τ → percentile | Thm. 4 | more false drops |
| Ordered → unordered | Def. 1 | errors on coupled steps |
| λ ∈ {0, λ̂, ∞} | Def. 1 | faithfulness–success trade-off |
| Capacities on/off | Prop. 1 | errors on consumables |
| QRL vs. TMD; IQE vs. MRN | Sec. 4.2 | ranking quality of d_θ |
| Contexts per task: 1, 2, 3 | Thm. 7 | label accuracy rises |
| Ensemble K = 1 vs. 5 | Sec. 4.2 | fewer false substitutions |
| Play budget | Sec. 4.3 | success saturates |
| Policy vs. planner executor | Sec. 4.4 | isolates control error |

## Grades (IG-Sub, single step)

G0 same instance · G1 same category, new instance · G2 different category, same affordance ·
G3 different affordance requiring a different strategy · G4 no feasible substitute
(correct behaviour: abstain). G0–G1 coincide with IG's L0–L2 and serve as controls.

## Week-2 go/no-go (execution plan)

Quasimetric PR-AUC clearly above visual cosine; learned policy reaches oracle subgoals in a
majority of trials. Fallback: supervised regression of planner path lengths into the IQE
head (distance distillation, `--objective distill`).
