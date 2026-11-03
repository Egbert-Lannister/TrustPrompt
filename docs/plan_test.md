# S3: the one test evaluation of method v1 (pre-registered 2026-11-03, before any test sample is read)

Method v1 is frozen (docs/freeze_method_v1.md, commit ebe7609). This plan fixes the populations, the comparison
family and what is reported. Test samples are read once, after this plan and the family file are committed.

## Populations

- **NYU Mirror3D test**: every annotation of `configs/splits/mirror3d_nyu_v1.json` whose scene is in the test split
  (30 images, 23 scenes), fetched from the official NYU Depth v2 MAT by byte range, processed exactly like the development set
  (`scripts/prepare_mirror3d`, same code path). NYU registered Z is not certified as strict RGB-axis Z
  (docs/intrinsics_audit.md), so its evaluation role is `formal_test_registered_z`; HAMMER uses `formal_main_result`.
- **HAMMER test**: scenes 12-14, all three sensors, frames with index % 5 == 0 (the stride used for the validation
  scene), from the official archive (MIT licence) by `scripts/hammer_remote` and processed like the development set:
  286 frames x 3 sensors = 858 samples, 0.63 GiB compressed download (`configs/hammer_test_v1.yaml`).

## Primary comparison (`configs/baselines/holm_family_test_v1.yaml`)

Six models (PromptDA-L, PriorDA-B, OMNI-DC, CDM, LingBot-Depth, Any2Full-L) x two datasets: raw prompt vs + T v1 mask,
main-scope C AbsRel, paired bootstrap (NYU: scenes; HAMMER: circular blocks of 8 frames, forced by `unit:
{hammer: frame_block}` because a 3-scene bootstrap could never reach the Holm level; inference about the frames of
these scenes), Holm over the 12 tests at alpha 0.05. Missing runs stay in the family with p = 1. The forced unit was
added on 2026-11-03 before any test sample was read: the code would otherwise have switched to a scene bootstrap,
contrary to the plan's wording.

## Secondary, descriptive

- IR, DOR, the clean-pixel cost (main scope and full image), invalid predictions, per sensor on HAMMER.
- Conformal guarantee: the unflagged-danger rate (P > 1.2 GT among pixels T does not flag) at alpha 0.1, per dataset.
- Oracle mask (GT C removed) as the upper bound.
- CAPA raw vs CAPA + T v1 mask (as S2): NYU all test images; HAMMER 1 frame per scene x sensor (9 images; about
  12 min per image per variant).

## What is not done

No threshold, weight or fill is changed after the test is read; no additional models are added to the primary family.

## How it runs

`python -m scripts.run_test_eval` (Windows, `.venv`) runs every step in order and continues where it stopped
(docstring of the script). Each script that may read a test sample takes `--formal-test` and first checks the freeze
(`tp/formal.py`, `configs/freeze_method_v1.json`: SHA256 of the frozen artefacts, the method library code, this
plan and the test family). Without `--formal-test` the test split stays locked as before.
