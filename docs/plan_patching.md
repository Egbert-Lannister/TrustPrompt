# Direction 2: where does a prompt-depth model copy wrong readings? (opened 2026-11-01)

Status: **pre-check pre-registered, not run**.

## 1. Idea

If the copying of wrong readings happens at one identifiable place inside a frozen prompt-depth model, an
intervention at that place alone could stop it without masking the prompt for the whole model. Activation
patching (causal mediation) is standard in language-model interpretability; we found no use of it to locate how
depth-completion models fuse a depth prompt (web search 2026-11-01). Risk stated before the run: the copying may be
spread over all fusion stages, in which case the direction ends.

## 2. Pre-check (script `scripts/patch_precheck.py`; PromptDA-L @ 9161547, inference only)

PromptDA-L uses the 192x256 prompt at five sites: its min / max (normalisation of the prompt and de-normalisation
of the output, N) and the zero-initialised prompt branch of the four DPT fusion blocks (F4 coarsest .. F1 finest).
Per image two prompts: raw and oracle (wrong readings C replaced by the globally aligned UniDepth prior D_g, the
existing oracle_prior variant, same configs and seeds as `configs/promptda/*_vitl_oracle_prior.yaml`). A run gives
the oracle prompt to the sites in a set S and the raw prompt elsewhere; 22 runs per image ({}, all, 5 singles,
10 pairs, 5 all-but-one). {} and all are checked against `model.predict` (max |diff| <= 1e-4 m).

Images: the 125 development images of the CAPA pre-check (NYU 95, HAMMER 30), test never read.
Metric: main-scope C AbsRel, pixel-pooled per dataset; TE = A(raw) - A(oracle); r(S) = (A(raw) - A(S)) / TE.

Rule:
- void unless TE > 0 with scene-bootstrap CI low > 0 on both datasets;
- **localised** if one set S of at most two sites has r(S) >= 0.70 with CI low >= 0.50 on both datasets (the same S);
- **diffuse** otherwise; diffuse ends direction 2. Localised leads to an intervention design at S, with its own rule.
Descriptive: necessity of each site (all-but-one runs), IR, clean-pixel AbsRel.
Bootstrap: 2000 draws over scenes, seed 20261102.

## 3. Result (2026-11-02, results/patch_precheck_20261102_033859, code 66dae65): **void**

The rule needs a total effect on both datasets. HAMMER has none for PromptDA-L: main-scope C AbsRel 0.0477 with the
raw prompt vs 0.0519 with the oracle prompt (TE -0.004, CI -0.030 to 0.013; IR 0.132 vs 0.083), so there is nothing
to localise there and the pre-check is void, neither localised nor diffuse. NYU has a large effect
(0.813 vs 0.215, TE 0.599, CI 0.454-0.753).

Descriptive only (NYU, not a test of the rule): the copying sits in the two finest fusion branches. F2+F1 restores
0.902 (0.871-0.932) of the effect; single sites F2 0.490, F1 0.408, F3 0.269, N 0.226, F4 0.092. Necessity from
all-but-one runs: F1 0.613, F2 0.314, F3 0.069, F4 0.020, N -0.006. Patched forwards matched model.predict exactly
(max |diff| 0). Engineering run results/patch_precheck_20261102_033733 (--limit 2) kept as the check.
