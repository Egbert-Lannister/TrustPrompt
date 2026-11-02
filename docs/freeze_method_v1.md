# Method v1 freeze record (2026-11-02, before any test sample is read)

Decision taken by Claude at the user's request ("你替我决定"). From this record on, method v1 is fixed; nothing below
may change before the test evaluation, and the test split is read once.

## Frozen components

| component | value | file | SHA256 |
|---|---|---|---|
| trust model T v1 | real_sim, no encoder features, error selection (fold 0 of the HAMMER cross-fit = T v1) | `results/trust_t_v1_real_sim_nofeat/best.pt` | 9696f70e50228ae398c748608bcf4d102a8b150be45d33d3fc0cdf4f5033284c |
| T v1 training config | | `results/trust_t_v1_real_sim_nofeat/config.json` | e93b3b31e5e481f2c3eedb1137f8521c240f9eaa8dec054f5ad66a0dbf810587 |
| conformal threshold | alpha = 0.1, danger P > 1.2 GT; lambda NYU 0.72, HAMMER 0.46 | `results/trust_t_v1_crc_d20.json` | a659e072def830c36621b772337bc7cacc87b9d60e72de050d2aede6fd87e590 |
| mask and fill | pixels with T < lambda removed; PromptDA-L: filled with D_g (UniDepthV2-S, global robust alignment); completion baselines: set to 0 | `configs/baselines/cache.yaml`, `configs/promptda/{nyu,hammer}_vitl_trust_prior.yaml` | a45e421f..., 9eb2d3b8..., 479d8352... |
| baselines | pinned sources and weights | `configs/baselines/registry.json` | 517d16ae984175a99b76f4daaf0a1a5d1c63387174a1dba2095fca89123434b7 |
| comparison families (development) | Holm over models, C AbsRel, raw vs + T v1 | `configs/baselines/holm_family_v1.yaml`, `holm_family_cf_v1.yaml` | d99d229d..., 7fd51fb4... |

Repository state at the freeze: commit 71b4e35 (master).

## Test evaluation (S3), to be run after S2

- Populations: NYU Mirror3D test scenes (23 scenes, `configs/splits/mirror3d_nyu_v1.json`) and HAMMER test scenes
  12-14 (3 sensors). Test data are not processed yet; preparation and the test comparison family (same models, metric
  C AbsRel, Holm alpha 0.05) will be written to `configs/baselines/holm_family_test_v1.yaml` and committed before the
  first test sample is read.
- Run with `experiment_role = formal_main_result` (the only role `run_eval` admits for the test split).
- Reported: C AbsRel, IR, DOR, the clean-pixel cost, invalid predictions; no re-tuning after the test is read.

## Not done, by decision

- S4 (non-planar ground truth): not pursued. The limitation concerns plane-based fills; method v1 fills with the
  prior D_g, not with planes, so it is not affected. Transparent-object sets (TransCG, ClearPose) would add another
  HAMMER-like domain, not a check of the mirror ground truth.
- S5 (official Mirror3D split): not pursued. The report does not compare with published numbers; the scene-isolated
  split is kept and stated as a limitation.
