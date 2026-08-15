---
name: model-improvement-roadmap
description: Technical roadmap for improving TinySepsis, grounded in a review of what other 2024-2025 sepsis-prediction architectures do differently, prioritized by expected impact vs. engineering cost.
---

# Model Improvement Roadmap: TinySepsis

This is a competitive/technical analysis, not a promise. Every item below is prioritized by **expected impact on the paper's own evaluation axes** (cross-institution AUROC gap, calibrated false-alarm transfer, clinical utility) **vs.\ engineering cost**, not by novelty for its own sake -- the whole thesis of this project is that a small, well-validated model beats a bigger, less-scrutinized one, and that discipline should apply to future work too.

## What the current literature is actually doing (2024-2025)

| Approach | What it does | Representative work |
|---|---|---|
| Attention/Transformer encoders for ICU time series | Replace the recurrent encoder with self-attention; several report strong discrimination and, notably, one framework (MEET-Sepsis) reports competitive accuracy using only ~20% of the ICU monitoring time state-of-the-art methods need | KA-Transformer (kernel attention, reduced parameter count); MEET-Sepsis (multi-scale, cascaded dual-convolution attention) |
| Neural Controlled Differential Equations (Neural CDEs) / Neural ODEs | Model the patient trajectory as a continuous-time process rather than discretizing into hourly bins, which is a more faithful match to genuinely irregular ICU sampling than our current hourly-binned GRU | Kidger et al.\ (Neural CDEs for Irregular Time Series, NeurIPS 2020) and follow-on sepsis-specific applications reporting AUROC around 0.93 with 6h lead time on a large ICU cohort |
| SHAP-enhanced attention + continuous-time nets with uncertainty | Combine an interpretable attention mechanism with explicit uncertainty quantification, addressing calibration and explainability jointly rather than as separate post-hoc steps | "Interpretable and Uncertainty-Aware Deep Learning Framework for Early Sepsis Prediction" (MAKE, 2024) |
| Conformal prediction + deep sequence models | Combine a learned sequence model with conformal risk control for a statistical false-alarm guarantee -- **this is the same idea TinySepsis already implements** (Section 6 of the paper), applied here to non-ICU inpatients | "Time-Series Deep Learning and Conformal Prediction for Improved Sepsis Diagnosis in Non-ICU Hospitalized Patients" (2024) |
| Multi-Gaussian-Process + attention TCN (MGP-AttTCN) | Handle irregular sampling via a Gaussian process layer that interpolates missing measurements probabilistically, then an attention-based TCN on top, explicitly built for interpretability | Original work predates 2024 but remains a relevant interpretability-focused reference point |

## Prioritized roadmap

### Tier 1 -- high expected impact, moderate cost (do these first)

1. **Multi-seed-averaged hyperparameter selection, not single-run.** The paper's own multi-seed analysis (Section 8.3) shows TinySepsis has meaningfully higher seed-to-seed variance (std 0.010-0.024) than the tabular baselines (std 0.000-0.006). Before any architecture change, a proper hyperparameter sweep (learning rate, hidden size, dropout) evaluated by *mean over 5 seeds*, not one run, would likely reduce variance and could move the mean itself -- this is the single highest-value, lowest-risk next step, and it's pure engineering discipline, not a new idea.
2. **Neural CDE / continuous-time encoder in place of the discrete GRU.** This is the most literature-supported single architecture change: it directly targets the paper's own missingness-and-irregularity story (Section 6's mask/time-since-last-measurement channels are, in effect, a hand-engineered approximation of what a Neural CDE does natively) and multiple groups report meaningful AUROC gains from it on comparable ICU cohorts. Cost: moderate -- `torchcde` and `torchdiffeq` are mature libraries; the risk is training-time and numerical-stability overhead on an 8GB-VRAM budget, which should be piloted on a subset before committing.
3. **Ablate the multi-seed protocol onto the calibration/conformal pipeline, not just AUROC.** Section 9.4's finding (TinySepsis's calibrated threshold transfers worse than its ranking does) is itself only a single-run result for the conformal-threshold analysis. Extending the multi-seed replication to the conformal-threshold utility numbers (Table 3) is a direct, mechanical extension of infrastructure already built (`scripts/run_multiseed.py`, `scripts/utility_at_conformal_threshold.py`) and would tell us whether that finding is as robust as the headline AUROC one.

### Tier 2 -- worth doing, higher cost or less certain payoff

4. **Lightweight attention instead of (or alongside) the GRU.** KA-Transformer-style kernel attention specifically targets parameter efficiency, which matches this project's low-resource constraint better than a standard multi-head Transformer would. Given TinySepsis's entire value proposition is small-and-robust, any attention variant considered should be evaluated on parameter count and cross-institution gap, not benchmark AUROC alone -- a bigger, more expressive model that re-introduces the "fits the training hospital too well" failure mode this paper documents would be a regression relative to the paper's actual thesis.
5. **Self-supervised pretraining on unlabeled ICU time series before fine-tuning on the sepsis task.** Foundation-model-style pretraining (masked value/time reconstruction, contrastive next-hour prediction) on all PhysioNet Challenge 2019 data regardless of sepsis label, then fine-tuning the small head, could improve data efficiency and potentially calibration transfer, at the cost of a meaningfully more complex training pipeline.
6. **SHAP-based (not just magnitude-based) per-request explanation in the demo/CDS Hooks card.** The current `top_contributing_factors` in `demo/app.py` and `cds_hooks_app.py` uses raw z-score magnitude as a proxy for importance; a real (even approximate, e.g. DeepSHAP or Integrated Gradients) attribution would be more defensible clinically and is a moderate, self-contained engineering task independent of the core architecture.

### Tier 3 -- interesting, not prioritized

7. **Multi-task learning** (jointly predicting sepsis risk alongside related outcomes like AKI or mortality) is reported to help in some ICU multi-task settings, but adds label-definition complexity this project has deliberately avoided (Section 3's careful, single, well-justified label definition is a strength worth protecting).
8. **Graph neural networks over irregular multivariate series** are an active research direction but the engineering and interpretability cost is high relative to expected gain for a project whose core argument is about simplicity and robustness, not maximal architecture novelty.

## What NOT to do

Chasing benchmark AUROC on the internal split alone would directly undermine this project's own finding: Table 1 already shows XGBoost beats TinySepsis internally by a wide margin, and that isn't the problem this paper is trying to solve. Any architecture change should be evaluated primarily on the **cross-institution gap** and the **conformal-threshold transfer**, exactly the two axes the current paper introduces -- not on internal AUROC, where a bigger model will almost always win and tell us nothing new.
