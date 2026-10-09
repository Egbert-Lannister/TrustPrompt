# Mirror3D / NYU data audit

This is a fresh read-only inspection, not an assertion that the old implementation plan was complete. Machine-readable evidence and file hashes are in `audit_data.json`.

## Existing files and cache qualification

`E:/GLARE_Data/first_surface/mirror3d/` contains the official `nyu.zip` (8,470,463 bytes), `nyu.csv` (5,926 bytes), an incomplete MAT download (40,894,464 bytes), and a selected HDF5 cache (151,758,502 bytes). The selected cache has 125 source RGB slices and 82 source inpainted-depth slices. It has no `rawDepths`, semantic labels or scenes. The incomplete MAT is not used.

The selected cache preserves full source-grid arrays with explicit presence masks. The new bounded fetcher validates its source identity against the live official MAT before reusing exact-index source slices; it does not reuse old model predictions, SampleRecords or cropped samples.

## Numbering and scenes

The old GLARE adapter and its unit tests still use `N - 1`. The historical correction was reported but had not changed that adapter. A fresh re-run of all 21 previously audited train/development frames (22 mirror instances, 613,095 pixels) gives:

| Source index | Pixels within 1 mm of the author plane | Fraction |
| --- | ---: | ---: |
| N | 612,307 | 0.9987147179474632 |
| N - 1 | 490 | 0.0007992236113489753 |

These numbers reproduce the historical evidence exactly. Residual outliers are retained, not excluded. Some individual frames do not achieve 99%; the aggregate regression threshold is a numbering diagnostic, not a license to replace or filter GT pixels.

The author's current [export script](https://raw.githubusercontent.com/3dlg-hcvc/mirror3d/main/mirror3d/utils/export_mat_image.py) names source slice `i` as `i + 1`, in conflict with the released mirror annotations. TrustPrompt follows the independently checked released-annotation pairing `annotation N -> source images[N], rawDepths[N], depths[N], labels[N]`. The CSV scene field also follows the old index: 58 of 125 scene names disagree with the corrected index. Reusing CSV grouping would cause scene leakage.

The new fetcher checks all 125 corrected scene identities against the official MAT `scenes` field before any selected-frame reads. The conservative historical H revision produces 55 train / 18 validation / 23 test scenes and 76 / 19 / 30 frames. Historically exposed actual scenes remain outside test. Five smoke frames (45, 184, 245, 440, 654) belong to five distinct training scenes. Previous-index depth is read only for numbering regression after checking it is outside test scenes.

## Units, grids and target source

The [NYU official field documentation](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) identifies `rawDepths` as metre-valued depth projected to the RGB grid before hole filling; `depths` is the inpainted product. The prompt uses `rawDepths` without filling. The measured official HDF5 layouts are images `[1449,3,640,480]`, rawDepths/depths `[1449,640,480]`, and labels `[1449,640,480]`.

After transpose, all spatial arrays receive the same crop `y=12:468, x=16:624`, producing 456 x 608. The mirror target is reconstructed from `uint16(depths[N] * 1000)` plus the precise delta with uint16 modular addition, then converted to metres. Adding a floating or unbounded signed delta would not reproduce the [author reconstruction](https://raw.githubusercontent.com/3dlg-hcvc/mirror3d/main/mirror3d/dataset/gen_refinedD_from_delta.py).

Mirror planes use the author geometry with `fx=fy=519`, `cx=304`, `cy=228` on the cropped grid. Plane Z in millimetres is `-d / (a*(x-304)/519 + b*(y-228)/519 + c)`. The [Mirror3D data documentation](https://github.com/3dlg-hcvc/mirror3d/blob/main/docs/Mirror3D_dataset.md) names focal length 519, while its coordinate prose conflicts with the positive-Z generation rays; numerical regression follows the generation convention. This approximate plane calibration must not be silently claimed to be exact NYU RGB calibration.

`nyu/mirror_plane/N.json` contains a list of objects with `plane`, `normal`, and `mask_id`. Integer precise-mask IDs are preserved without palette conversion. Outside annotated mirrors, inpainted depth is an explicitly weaker reference; it is not independent first-surface physical GT. Likewise, the complement of a mirror mask is not automatically ordinary material. Semantic classes must be checked before any ordinary-material labeling.

## Fresh raw subset acquisition

`scripts/fetch_nyu_subset.py` ports only the reviewed HTTP-range reader into an independent script. It bounds network payload to 100 MiB, rejects full-body responses, tracks source identity and hashes, reads only approved train-frame payloads, and writes under `data/source_cache/`. The resulting `.json` receipt determines whether acquisition completed; partial outputs are not valid experiment inputs.

Acquisition completed successfully: `nyu_smoke_v1.h5` is 14,064,714 bytes and the official network transfer was 30,408,705 bytes over 33 requests. All 25 frame fields have array hashes. Online scene checks found zero differences from the historical corrected mapping. Five raw-prompt valid fractions range from 56.07% to 80.71%; zero-valued holes are present and no NaN values were observed. These are acquisition/engineering checks, not model benchmark results. No test depth was read.

The 894-class semantic vocabulary includes 23 names containing mirror/glass/window/metal, including glass containers and a metal sheet. These are only candidates, not an audited material taxonomy. Excluding just three semantic IDs and labeling all other objects ordinary would incorrectly assign some transparent/metal objects. The initial conservative mapping should therefore mark the precise mirror instances as material 2 and retain unknown material outside them until a material-specific mapping is qualified.
