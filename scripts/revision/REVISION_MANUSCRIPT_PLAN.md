# R2 manuscript and response-letter plan

Status: working plan, confirmed with the lead author on 2026-09-04.

This document records presentation and editing decisions. It does not replace
the scientific provenance recorded by the analysis scripts and output
manifests.

## Main revision themes

The first-page summary of the response letter should be concise and organized
around two major changes:

1. Strengthened the treatment of uncertainty quantification, especially in
   the 20-dimensional experiments, by adding visual evidence and revising the
   discussion of calibration, limitations, and the practical interpretation
   of pointwise credible intervals.
2. Although not explicitly requested in the R2 reports, corrected the
   AdaBoost tuning logic used in the comparison: the number of trees is now
   selected by cross-validation using exponential loss rather than
   classification accuracy. Corresponding numerical results and discussion
   must be updated.

The AdaBoost change should not be described broadly as a change to the
AdaBoost algorithm itself. It is specifically a correction to the criterion
used to select the ensemble size in the density-ratio comparison.

Other revisions, such as figure styling, reference corrections, and detailed
point-by-point edits, belong in the detailed responses rather than the opening
major-change summary unless their scope later becomes substantial.

## 20D figure placement

The following two figures are intended for the main manuscript:

1. the 2-by-4 20D calibration figure, with columns Global Shift, Location
   Shift, Dispersion, and Null, and rows Balanced and Unbalanced;
2. the 2-by-3 20D pointwise-localization figure, with columns Global Shift,
   Location Shift, and Dispersion, and rows Balanced and Unbalanced.

The figures have complementary roles. The calibration figure documents the
frequentist under- or over-coverage of the generalized Bayesian credible
intervals. The localization figure shows that, even when exact coverage is
imperfect, intervals can exclude zero in the correct direction when the true
absolute log-density ratio is sufficiently large, while wrong-direction
exclusions remain rare.

Do not describe zero exclusion as a formally calibrated frequentist null-
hypothesis test. The intended interpretation is posterior evidence that the
log-density ratio is nonzero and evidence about its direction.

The current main-manuscript visualization of estimated 20D density ratios
(currently Figure 4) should move to the supplement. It remains useful as a
qualitative spatial illustration, but it is less directly responsive to the
reviewers' high-dimensional uncertainty-quantification concern than the two
new figures. The main text must explicitly refer readers to the relocated
supplementary figure, and the supplement must provide a short interpretation
rather than presenting the figure without explanation.

No figures in the application section are in scope for this workstream.

## Current status and next sequence

The reviewer-relevant visual materials are provisionally complete. Remaining
work is primarily:

1. revise the response letter, beginning with its first-page summary;
2. draft and revise the corresponding main-text and supplementary discussion;
3. insert the two new 20D figures and move the current Figure 4 to the
   supplement;
4. update captions, cross-references, and figure numbering;
5. compile and check pagination after the substantive text and placement edits
   are stable.

All newly added or changed R2 manuscript/response text must follow the project
blue-markup and Codex-provenance rules in `AGENTS.md`. A local manuscript edit
and an Overleaf push require separate approvals.
