# Layered prompt reading: plan (opened 2026-10-29)

Status: **abandoned 2026-10-30 (user decision after the check in sec. 3)**. Nothing below is an experimental result unless a result file is cited.

## 1. Idea and prior-work check

Reframing: on transparent and reflective surfaces a depth sensor does not misread; it measures another surface.
Through glass it sees the background (HAMMER `behind`, the naked frame); in a mirror it sees the virtual image. A
prompt-based completion model could then output two layers, the first surface and the layer the sensor sees, and use
the "wrong" readings as supervision-free evidence for layer 2 instead of discarding them.

Literature check (2026-10-29, web search; details in the session log):

| work | layers | input | treats the off readings as |
|---|---|---|---|
| LayeredDepth (ICCV 2025, Princeton VL) | multi | RGB only | n/a (no sensor input) |
| SeeGroup (CVPR 2026 oral, arXiv 2605.28735) | variable | RGB only | n/a |
| GlassRecon (arXiv 2604.18336), HTMNet (2505.20904), TDCNet, ClueDepth, TCG-Depth, Seeing Glass | single | RGB-D | errors to remove / repair |
| Mirror3D (CVPR 2021) | single (mirror plane) | RGB-D | errors to replace |
| Reflect3r (2509.20607) | uses reflections | RGB | n/a (virtual view for stereo) |
| coded / multipath ToF separation (Kadambi et al.; Bhandari et al. 1404.1116) | multi | raw ToF signal | mixtures, separated from raw correlation data |

No work found that takes a commodity depth sensor's processed output as an observation of a second layer for
prompt-conditioned layered depth completion. Closest: multipath ToF (needs raw signals, not the depth map a completion
model receives) and RGB-only layered depth (no sensor prompt).

## 2. Free check: are HAMMER's wrong object readings the second layer? (pre-registered)

Question: on HAMMER object pixels, what share of the wrong readings (|P - GT| / GT > 0.05) lie within 5 % of the
naked-frame depth `behind`? Script: `scripts/layer2_check.py` (CPU, development scenes 2-11, all 1143 samples; test
never read). Class definitions, placebo layer (B' = 2 behind - GT) and the bootstrap are in the script docstring.

Rule, fixed before the full run:

- primary endpoint: layer-2 share of C readings on tau-separable object pixels, pooled over d435 / l515 / helios,
  95 % CI by scene bootstrap (10 scenes, 10000 draws, seed 20261029);
- **supported**: CI low >= 0.50 and share >= 3 x placebo share;
- **leaning**: point >= 0.50, CI low < 0.50, >= 3 x placebo;
- **sensor-specific**: pooled neither, but a sensor reaches CI low >= 0.50 and >= 3 x placebo;
- **not supported**: otherwise.

Disclosure: while timing the loader one frame per sensor (scene5 traj1_1 000000) was inspected before this rule was
written (C on separable pixels landing on layer 2: d435 6 %, l515 49 %, helios 45 %; between the layers: d435 85 %,
l515 41 %, helios 39 %). The full run includes that frame.

What each outcome means for the direction:

- supported / leaning: build the layered model (layer-2 head; wrong readings supervise layer 2, GT supervises layer 1;
  HAMMER naked GT for evaluation of layer 2). Next pre-check: does a frozen completion model's prediction plus the
  readings already recover layer 2 (a trivial baseline the method must beat)?
- sensor-specific: the claim is narrowed to that sensing principle (e.g. ToF); the paper would study why.
- not supported, with a large "between" share: readings are mixtures of the layers, not a second layer; the layered
  framing would need a mixture model (closer to multipath ToF), which is weaker without raw signals. We would report
  and return to the method choice.

Mirrors: real mirror data with a virtual-image GT is not available here (NYU / Mirror3D give the mirror plane only;
Hypersim mirror readings in this repo are synthesised by us, so they cannot test sensor behaviour). Part N of the
script reports only on which side of the mirror plane NYU's wrong readings lie.

## 3. Result of the layer-2 check (2026-10-30, results/layer2_check_20261030_035620, code 7a33ade)

**Not supported** under the pre-registered rule. Pooled layer-2 share 0.439 (95 % CI 0.372-0.504), placebo 0.024;
l515 0.547 (0.477-0.627), helios 0.505 (0.423-0.588), d435 0.242 (0.170-0.312, 0.672 between the layers). Interior
rows are similar. The ToF sensors' t distribution peaks at t = 1 on top of a continuum over (0, 1); d435 peaks at
t = 0.2-0.4. NYU mirrors (descriptive): 0.990 of wrong readings lie behind the mirror plane. Full tables:
docs/layer2_check.md. Any follow-up (e.g. a two-layer mixture reading model) needs its own rule, written before it is run.
