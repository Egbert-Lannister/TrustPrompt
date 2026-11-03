# Supplementary experiments for the analysis report (opened 2026-11-02)

Status: S1 and S2 pre-registered before any run (rules below and in the scripts). Everything is development data;
test is never read. Report: docs/analysis_paper.md.

## S1. Is the prior's signal global or local? (`scripts/local_consistency.py`, CPU)

Purpose: a direct test of the report's central explanation (sec. 6, "local self-consistency"). The frozen
UniDepthV2-L prior, already globally aligned (CAPA pre-check step 0, same 125 images), is re-fitted to the readings
by least squares (P ~ a D + b) inside a w x w window around every pixel; the residual after the local fit is the
score. If wrong readings are locally self-consistent, the local fit explains them about as well as correct readings,
so the AUROC (wrong vs correct, main scope) should fall toward 0.5 as the window shrinks.

Rule (NYU): L1 AUROC(r_31) <= 0.60 and AUROC(global) - AUROC(r_31) >= 0.15 with scene-bootstrap CI low > 0
(2000 draws, seed 20261103); L2 the AUROC does not decrease with the window (w = 15, 31, 63, 127, global; each step may
drop by at most 0.01). supported = L1 and L2. HAMMER, the per-sensor split and CAPA's IR-vs-area relation are
descriptive.

## S2. Does method v1's trust mask stop CAPA from copying? (`scripts/tta_tmask.py`, GPU)

Purpose: the report's implication 2 (test-time adaptation needs a fault model). CAPA with the same code, weights,
config and per-image seed as the CAPA pre-check, adapting only to the readings T v1 trusts (conformal alpha 0.1).
T must be out of fit: NYU val + train-holdout scenes (30 images); HAMMER the 30 CAPA images with the leave-scene-out
cross-fitted T. Oracle mask (C removed) on NYU only, because HAMMER costs about 10 min per image per variant.
Reference: CAPA on all readings and the prior, read from results/tta_precheck_20261101_012932.

Rule: rho = 1 - AbsRel_C(tmask) / AbsRel_C(CAPA raw), pixel-pooled, paired scene bootstrap (2000 draws, seed
20261104). NYU: rho >= 0.50 with CI low > 0; HAMMER: CI low > 0. "rescues" if both, "nyu_only" if only NYU,
otherwise "does_not_rescue". Descriptive: oracle rho, the clean-pixel cost, C AbsRel vs the prior, IR.

## Not started; each needs a decision by the user

- S3 formal test run of method v1: needs the method frozen (T weights, alpha, fill) and recorded before test is
  read; the test is read once.
- S4 non-planar ground truth (answers the Mirror3D plane-fit limitation): TransCG or ClearPose (download and licence
  terms to be accepted by the user), or own capture (docs/capture_plan.md).
- S5 official Mirror3D NYU split for comparison with published numbers (re-train T on that split).

## S1 result (2026-11-02, results/local_consistency_20261102_145700, code aed13a4): **not supported** (L1 True, L2 False)

NYU AUROC by window: w=15 0.426, 31 0.377, 63 0.364, 127 0.389, global 0.839; AUROC(global) - AUROC(31) = 0.462
(CI 0.309-0.626). L1 holds; L2 fails because the local AUROC first falls (15 -> 63) and then rises, i.e. it reaches
its minimum at an intermediate window and lies below 0.5. The monotonicity condition did not anticipate local
AUROCs below chance; not changed afterwards. HAMMER: local and global about equal (0.70-0.73). Descriptive: CAPA IR
at step 100 vs contaminated area, Spearman 0.546 (NYU, 78 images) and 0.419 (HAMMER, 30); at step 0 0.239 / 0.143.

## S2 result (2026-11-03, results/tta_tmask_20261102_110950, code ebe7609): **rescues**

C AbsRel reduction vs CAPA raw: NYU 0.870 (CI 0.744-0.942; 30 images / 25 scenes), HAMMER 0.440 (0.356-0.531;
30 / 10). NYU: prior 0.094, CAPA 1.024, CAPA + T 0.134, CAPA + GT mask 0.066; in-mirror clean pixels 0.316 / 0.048 /
0.367 / 0.016 (the cost of deleting the ~3 % correct mirror readings). HAMMER: prior 0.132, CAPA 0.189, CAPA + T
0.106; clean 0.034 / 0.019 / 0.019. T removes 17.8 % (NYU) and 25.0 % (HAMMER) of the readings per image. One invalid
pixel (NYU oracle). Engineering run results/tta_tmask_20261102_100109 (--limit 1) kept as the check.
