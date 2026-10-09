# UniDepth RGB-only smoke

This module is implemented and both its one-frame preflight and five-frame run passed on real training RGB. It is an engineering baseline check, not a formal benchmark or a trained TrustPrompt model.

## Provenance and loading

`baselines/unidepth.py` selectively adapts the local import boundary and native inference call from GLARE `experiments/generate_i3_transphy3d_unidepth_cache.py`, SHA256 `683ed8585ac3e520da5a54ef3e642867e92b77becd8ed61acb74beb16acf7af1`. It does not import GLARE. The configurable external dependency is the existing UniDepth checkout at `D:/Python_Script/GLARE/.external/stage24/repos/unidepth_v2_vits14`, commit `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`; the inspected checkout was clean.

Existing weights are read from `D:/Python_Script/GLARE/models/stage24/unidepth_v2_vits14`, without copying or downloading. `model.safetensors` SHA256 is `93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648`. Configuration SHA256 is `ecc1f898690debe387d10329cca5d9d66a0a447ce83c3a2a7ae3673c7ce43cfe`. Each run records full source-file hashes and both existing weight-format hashes. Upstream source declares CC-BY-NC 4.0.

`sys.dont_write_bytecode=True` prevents writes of bytecode into the external source tree, including later lazy imports. Offline flags and `local_files_only=True` prevent pretrained-weight downloads. Optional W&B visualization names are stubbed during import; no tracking session is created. Runtime caches and generated artifacts stay inside TrustPrompt.

The local project environment provides PyTorch 2.11.0+cu126, torchvision, timm, einops, safetensors and huggingface-hub. xFormers is absent; the inspected DINO implementation has a PyTorch attention fallback. Upstream warnings about KNN and extract-patches pertain to optional upstream evaluation/training operators; the native RGB inference path passed without installing them.

## Input and geometry contract

The driver opens only the `rgb` and `metadata_json` NPZ members. Other NPZ bytes are hashed opaquely for identity. It rejects test/validation and synthetic samples in this smoke entry. The model receives exactly one uint8 `[1,3,H,W]` RGB tensor via `infer(tensor)`. Prompt depth, GT, material, mirror planes and camera intrinsics are never arguments to the predictor.

The inspected official native inference preprocesses RGB itself and returns `depth = points[:, -1:]`, in metres. The wrapper checks exact equality to returned point-cloud Z and requires predictions on the original RGB grid. Nonfinite/nonpositive predictions are retained and counted, rather than silently dropped. Learned intrinsics are cached as model outputs; no GT camera is supplied.

The fixed smoke configuration uses `resolution_level=0`, leaving upstream minimum/maximum bounds intact. A 456 x 608 input becomes 434 x 574 inside the encoder, then returns to 456 x 608. This is a reduced-resolution smoke baseline; a later comparison must freeze and disclose inference resolution.

## Reproduce

From the TrustPrompt project directory:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_unidepth_contract.py -q
.venv/Scripts/python.exe scripts/cache_unidepth.py --limit 1 --output cache/unidepth_preflight_v1
.venv/Scripts/python.exe scripts/cache_unidepth.py
```

Use a fresh `--output cache/<new-name>` to rerun; existing runs are immutable. Configuration is in `configs/unidepth_smoke.yaml`. The seed is 20261007. First run one training frame, then five frames under the same settings. The three contract tests passed and check forbidden GT decoding, test-scene rejection, the RGB-only call, and axial-Z versus radial-range output semantics.

Each output directory contains config, command, source snapshot, working-tree diff, receipt, log, per-frame timing/validation/hash records, and `predictions/<frame>.npz`. Cached fields are `depth[H,W]`, `points[3,H,W]`, `intrinsics[3,3]`, and `confidence[H,W]`, all float32. No scale alignment is performed. Downstream evaluation must use the existing fixed sample/region denominators.

## Verified preflight

Real training frame 45 (`bathroom_0003`) passed on the NVIDIA GeForce RTX 4070 Laptop GPU. Inference took 4.326 seconds after loading; peak PyTorch allocated memory was 348,976,640 bytes and reserved memory 408,944,640 bytes. Output had zero invalid depths and point-cloud Z matched exactly. Import/model loading took 76.919 seconds in this environment; it is recorded separately from inference.

The five-frame run completed with status `real_train_smoke_verified`. All five returned 456 x 608 finite positive depths, with exact point-cloud Z equality. Peak allocated memory was 348,976,640 bytes; reserved memory was 408,944,640 bytes. Per-frame inference times were 1.402, 0.0567, 0.0434, 0.0440 and 0.1464 seconds, respectively, including first-inference warmup on frame 45. These five timings are a bounded smoke observation, not a throughput benchmark. Frame 45's output hash was identical between preflight and the five-frame run. Full source hashes were rechecked after inference and no source file changed.

Machine-readable results are in `cache/unidepth_smoke_v1/receipt.json`, `images.json` and `verification.json`. Evaluation scores belong to the downstream evaluator, using the same original NPZs and method-independent masks; none were used to choose this model, resolution, frames or settings.
