# Fault-aware test-time adaptation: plan (opened 2026-10-30)

Status: **pre-check pre-registered, not run**. Nothing below is a result unless a result file is cited.

## 1. Idea and baseline

A recent family of depth-completion methods adapts a frozen depth foundation model to the sensor readings at test
time, using the readings themselves as the loss:

- Jeong et al., Test-Time Prompt Tuning for Zero-Shot Depth Completion (ICCV 2025);
- CAPA, Depth Completion as Parameter-Efficient Test-Time Adaptation (Ke et al., NVIDIA / ETH, arXiv 2602.14751,
  code github.com/nv-dvl/capa, CC BY-NC 4.0). It adapts LoRA or VPT parameters of VGGT / MoGe-2 / UniDepthV2 with
  an L1 loss to the readings after a robust scale-shift alignment, for 100 steps;
- low-rank decoder adaptation (Seo et al., arXiv 2603.01765).

Every wrong reading is a training target in these methods. CAPA's own failure analysis names glass and reflections
("LiDAR penetrating glass or measuring reflected distance") without a mechanism. Hypothesis for a module: a
low-capacity adapter starting from a strong prior fits readings that agree with the prior first and readings that
contradict it late (the memorisation effect known from learning with noisy labels), so the adaptation's own loss
trajectory could tell which readings to stop fitting, without an external detector.

Baseline used here: CAPA with UniDepthV2-L + LoRA, the repository's `config/unidepth_lora.yaml` (rank 16, alpha 32,
qkv, 100 steps, lr 3e-4, clip 1.0, seed 47) with one deviation fixed before any run: 4000 instead of 15000 random
reading points for CAPA's robust scale-shift alignment. CAPA's exact L1 alignment builds n x n tensors (about
50 n^2 bytes: measured 0.8 GB at 4000, extrapolated ~11 GB at 15000), which cannot fit the 8 GB laptop GPU. VGGT is not used (separate licence
terms); UniDepth is already part of this project. Source pinned at nv-dvl/capa 080739287e18509c3b046ac7e8b3e0593c2d4e7d;
weights lpiccinelli/unidepth-v2-vitl14 as CAPA loads them (the hub's `main`; public, ungated; CC BY-NC 4.0, research
use). On first use `main` is resolved to its commit and the file hashes are recorded; every later run uses that commit
and checks them (trust on first use). Correction 2026-10-30, before any result: the first version of this plan pinned
revision 1d0d3c5, which is the June 2024 checkpoint in the old config format; CAPA's vendored UniDepth (8d8cfe4) cannot
build it (KeyError layer_scale on the user's engineering check). No measurement had been produced.

## 2. Pre-check (pre-registered; script `scripts/tta_precheck.py`)

Data: development only, test never read. NYU / Mirror3D: all 95 development images (73 scenes). HAMMER: 1 frame per
scene x sensor (30 images, 10 scenes). Condition = the raw sensor depth (all valid readings), RGB as given; CAPA
never sees GT. For every image the script runs CAPA once and records the aligned prediction at steps
0, 5, 10, 20, 50 and 100 (step 0 = the frozen prior after CAPA's alignment, before any update; step 100 = CAPA's
output). The recording loop is a copy of CAPA's `_run_optimization` for one frame; on the first image both are run
and must agree (median relative difference <= 1 %, an engineering check, not a result).

Regions as in the project: tau = 0.05, 5-pixel band, main scope = NYU mirror pixels / HAMMER object pixels.

(a) Does CAPA inherit wrong readings? IR (share of main-scope C pixels with |D - P| / P < 0.05 and |D - GT| / GT > 0.05,
tp.eval.metrics) at step 100 vs step 0, pixel-pooled per dataset, paired scene bootstrap (2000 draws, seed 20261030).
**Premise holds** if, on at least one dataset, IR(100) - IR(0) has CI low > 0 and IR(100) >= 0.25.

(b) Does the loss trajectory add information beyond prior disagreement? Score of a reading at step k:
s_k = |D_k - P| / P. AUROC of s_k for wrong (C) vs correct (|P - GT| / GT <= tau) readings in the main scope, pooled
per dataset; scene bootstrap of AUROC(s_10) - AUROC(s_0). **Dynamics add information** if on both datasets the
difference is >= 0.03 with CI low > 0. s_0 is the frozen prior's disagreement with the reading, i.e. the kind of
signal T v1 already uses; s_10 is the residual after 10 % of the adaptation.

Descriptive only: AUROC of every recorded step, of the mean residual over the steps and of s_10 / s_0; the full-image
population; AbsRel on main-scope C and clean pixels at steps 0 and 100; time and GPU memory per image.

Outcomes:

- (a) and (b): go on to design the module (a robust adaptation objective driven by the trajectory); its own rule is
  written before it is run.
- (a) without (b): CAPA inherits, but the trajectory says nothing the prior does not; a module would reduce to
  prior-disagreement weighting inside TTA (close to T v1). Report and decide with the user.
- not (a): the premise fails for this baseline; candidate 1 is dropped.

Fallback fixed in advance: if UniDepthV2-L adaptation runs out of memory on the 8 GB laptop GPU, the whole run uses
UniDepth's lowest resolution level (`resolution_level = 0`, 200k-240k input pixels) and the deviation is recorded.
