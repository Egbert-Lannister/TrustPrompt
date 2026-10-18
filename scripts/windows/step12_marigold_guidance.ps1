# Marigold-DC guidance track (docs/marigold_guidance_plan.md; rules in configs/baselines/marigold_guidance_v1.yaml).
# Train/val only; test never read. No training; everything runs on this laptop.
#   default  M0 + M1: Marigold-DC weights (about 5.2 GB, once), loop check on 3 NYU images, then the 6 variants on
#            the 30 NYU dev images, evaluation, Holm family and the gate.  GPU time: printed by the M0 check.
#   -Hammer  M2: only after M1 passed the gate (results\marigold_guidance_gate.json): HAMMER, 10 scenes x 3 sensors
#            x 3 frames, raw / T mask / GT mask + the variants that passed, cross-fitted T, Holm family.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step12_marigold_guidance.ps1 [-Hammer] [-HfMirror https://hf-mirror.com]
param([switch]$Hammer, [string]$HfMirror = "", [int]$Parallel = 6)
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
. "$PSScriptRoot\common.ps1"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$bpy = "D:\Python_Script\TrustPrompt\.venvs\baselines\Scripts\python.exe"
if (-not (Test-Path $bpy)) { Write-Host "Run scripts\windows\setup_baselines.ps1 first."; exit 1 }
$EvalOk = @("small_scale_validation_passed", "completed_with_invalid_predictions")
$Label = "Marigold-DC"
$CfCache = "cache/trust_t_v1cf_real_sim_nofeat_hammer_dev_v1"
$startTime = Get-Date

# Cache one variant (1-image preflight, then the population) and queue its evaluations.
function Add-Variant([string]$Ds, [string]$Variant, [string]$Tag, [string[]]$Extra, [string[]]$EvalSet, [string]$EvalCfg, [string]$Prefix) {
  $cacheDir = "cache\bl_marigold_dc_${Variant}${Tag}_${Ds}_dev_v1"
  foreach ($stepDef in @(@{ dir = "${cacheDir}_preflight"; lim = @("--limit", "1") }, @{ dir = $cacheDir; lim = @() })) {
    if (Test-Done $stepDef.dir @("real_dev_verified")) { continue }
    Assert-FreeGB "D" 4
    $cmdArgs = @("-m", "scripts.cache_completion", "--model", "marigold_dc", "--dataset", $Ds, "--variant", $Variant,
                 "--output", $stepDef.dir) + $Extra + $stepDef.lim
    Invoke-Step "$Label [$Variant$Tag] $Ds -> $($stepDef.dir)" { & $bpy @cmdArgs }
  }
  foreach ($sp in @("val", "train")) {
    $evalOut = "results/${Prefix}_bl_marigold_dc_${Variant}${Tag}_${sp}_v1"
    $script:evalJobs += @{ module = "tp.eval.run_eval"; out = $evalOut;
      args = "--config $EvalCfg --method cached_prediction --split $sp --output $evalOut --set prediction_cache=$cacheDir " +
             "--set expected_model=$Label --set method_label=bl_marigold_dc_${Variant}${Tag} " + ($EvalSet -join " ") }
  }
}
$evalJobs = @()

if (-not $Hammer) {
  # ---------------- M0: weights and loop check ----------------
  Assert-FreeGB "D" 8
  $setupArgs = @("-m", "scripts.setup_baselines", "--models", "marigold_dc")
  if ($HfMirror -ne "") { $setupArgs += @("--hf-endpoint", $HfMirror) }
  Invoke-Step "Marigold-DC source and weights (pinned, SHA256)" { & $bpy @setupArgs }
  $checked = @(Get-ChildItem results -Directory -Filter "marigold_guidance_check_*" | Where-Object {
      (Get-Stage $_.FullName) -eq "check_complete" -and (Get-Content (Join-Path $_.FullName "receipt.json") -Raw | ConvertFrom-Json).summary.faithful })
  if ($checked.Count -eq 0) { Invoke-Step "M0: guided loop vs upstream on 3 NYU images" { & $bpy -m scripts.check_marigold_guidance } }

  # ---------------- M1: NYU, 6 variants, 30 images ----------------
  foreach ($vr in @("raw", "tmask", "tonesided", "tsoft", "oracle_mask", "oracle_onesided")) {
    Add-Variant "nyu" $vr "" @() @("--set train_scope=holdout") "configs/mirror3d_dev_eval.yaml" "mirror3d_dev"
  }
  Invoke-Parallel $evalJobs $EvalOk "evaluation" $py $Parallel
  Invoke-Step "pre-registered Holm family (NYU)" { & $py -m scripts.holm_family --spec configs/baselines/holm_marigold_nyu_v1.yaml }
  Invoke-Step "M1 gate" { & $py -m scripts.summarize_guidance --stage nyu }
  Write-Host ("M0 + M1 DONE ({0:hh\:mm\:ss} elapsed): see docs\marigold_guidance_nyu.md. Tell Claude before running -Hammer." -f ((Get-Date) - $startTime))
  exit 0
}

# ---------------- M2: HAMMER, gate-passing variants only ----------------
$gateFile = "results\marigold_guidance_gate.json"
if (-not (Test-Path $gateFile)) { Write-Host "No M1 gate yet: run this script without -Hammer first."; exit 1 }
$passed = @((Get-Content $gateFile -Raw | ConvertFrom-Json).passed)
if ($passed.Count -eq 0) { Write-Host "No guided variant passed the M1 gate: M2 is not run (docs\marigold_guidance_nyu.md)."; exit 0 }
$pg = @("--per-group", "3", "--train-scope", "all")
$evalSet = @("--set train_scope=all", "--set per_group=3")
Add-Variant "hammer" "raw" "_allpg3" ($pg + @("--tag", "_allpg3")) $evalSet "configs/hammer_dev_eval.yaml" "hammer_dev"
Add-Variant "hammer" "oracle_mask" "_allpg3" ($pg + @("--tag", "_allpg3")) $evalSet "configs/hammer_dev_eval.yaml" "hammer_dev"
foreach ($vr in @("tmask") + $passed) {
  Add-Variant "hammer" $vr "_cfpg3" ($pg + @("--tag", "_cfpg3", "--trust-cache", $CfCache)) $evalSet "configs/hammer_dev_eval.yaml" "hammer_dev"
}
Invoke-Parallel $evalJobs $EvalOk "evaluation" $py $Parallel
Invoke-Step "pre-registered Holm family (HAMMER, cross-fitted, 10 scenes)" { & $py -m scripts.holm_family --spec configs/baselines/holm_marigold_cf_v1.yaml }
Invoke-Step "M2 confirmation" { & $py -m scripts.summarize_guidance --stage hammer }
Write-Host ("M2 DONE ({0:hh\:mm\:ss} elapsed): see docs\marigold_guidance_hammer.md. Tell Claude." -f ((Get-Date) - $startTime))
