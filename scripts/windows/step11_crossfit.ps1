# HAMMER leave-scene-out cross-fitting of T v1 (configs/trust/hammer_crossfit_v1.yaml; docs/crossfit_trust.md).
# Train/val only; test never read. Fold 0 is T v1 itself (same fit / selection scenes); folds 1-4 are trained here.
#   GPU: train T for folds 1-4 (about 30 min each), trust maps, PromptDA-L with the cross-fitted T mask (about 50 min);
#        with -WithBaselines also the 5 completion baselines, raw and cross-fitted T mask, on all 10 scenes (about 5-6 h)
#   CPU: assemble the cross-fitted cache + conformal thresholds, evaluations, pre-registered Holm family (10 scenes)
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step11_crossfit.ps1 [-WithBaselines] [-Parallel 6]
param([switch]$WithBaselines, [string[]]$Models = @("priorda", "omnidc", "cdm", "lingbot", "any2full"), [int]$Parallel = 6, [int]$Steps = 4000)
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$bpy = "D:\Python_Script\TrustPrompt\.venvs\baselines\Scripts\python.exe"
$EvalOk = @("small_scale_validation_passed", "completed_with_invalid_predictions")
$startTime = Get-Date
# Must match configs/trust/hammer_crossfit_v1.yaml (scripts.crossfit_trust checks every model's fold).
$Folds = [ordered]@{
  "1" = "test=scene2,scene3;hold=scene4"
  "2" = "test=scene4,scene5;hold=scene6"
  "3" = "test=scene6,scene7,scene8;hold=scene9"
  "4" = "test=scene9,scene10;hold=scene2"
}
$CfCache = "cache/trust_t_v1cf_real_sim_nofeat_hammer_dev_v1"
function Get-Stage([string]$Dir) {
  $r = Join-Path $Dir "receipt.json"
  if (Test-Path $r) { return (Get-Content $r -Raw | ConvertFrom-Json).stage } else { return "missing" }
}
function Invoke-Step([string]$Label, [scriptblock]$Body) {
  Write-Host "=== $Label  ($(Get-Date -Format HH:mm:ss))"
  & $Body
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $Label (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}
function Test-Done([string]$Dir, [string[]]$OkStages) {
  $st = Get-Stage $Dir
  if ($OkStages -contains $st) { return $true }
  if ($st -eq "missing") { return $false }
  Write-Host "$Dir has stage '$st'. Rename that folder (append _failed) and re-run."; exit 1
}
function Assert-FreeGB([string]$Drive, [double]$NeedGB) {
  $freeGB = (Get-PSDrive $Drive).Free / 1GB
  if ($freeGB -lt $NeedGB) { Write-Host ("{0}: has {1:N1} GB free, needs about {2:N1} GB. Free space and re-run." -f $Drive, $freeGB, $NeedGB); exit 1 }
}
function Invoke-Parallel($JobList, [string[]]$OkStages, [string]$What) {
  $todo = @(); foreach ($jb in $JobList) { if (-not (Test-Done $jb.out $OkStages)) { $todo += $jb } }
  $cores = [Environment]::ProcessorCount; $nThreads = [Math]::Max(1, [Math]::Floor($cores / $Parallel))
  foreach ($ev in @("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")) { Set-Item "env:$ev" "$nThreads" }
  Write-Host "=== $What : $($todo.Count) of $($JobList.Count) to do, $Parallel at a time, $nThreads threads each ($(Get-Date -Format HH:mm:ss))"
  New-Item -ItemType Directory -Force results\_logs | Out-Null
  $procs = @()
  foreach ($jb in $todo) {
    while (@($procs | Where-Object { -not $_.HasExited }).Count -ge $Parallel) { Start-Sleep -Seconds 5 }
    while ((Get-PSDrive D).Free / 1GB -lt 3) {
      Write-Host "D: below 3 GB free; waiting"; Start-Sleep -Seconds 60
      if (@($procs | Where-Object { -not $_.HasExited }).Count -eq 0) { Write-Host "D: still below 3 GB free. Free space and re-run."; exit 1 }
    }
    $logFile = "results\_logs\" + ($jb.out -replace '[\\/:]', '_') + ".log"
    Write-Host "start $($jb.out)  ($(Get-Date -Format HH:mm:ss))"
    $procs += Start-Process -FilePath $py -ArgumentList "-m $($jb.module) $($jb.args)" -NoNewWindow -PassThru `
      -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
  }
  $procs | Wait-Process
  foreach ($ev in @("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")) { Remove-Item "env:$ev" -ErrorAction SilentlyContinue }
  $bad = @($JobList | Where-Object { -not ($OkStages -contains (Get-Stage $_.out)) })
  foreach ($jb in $bad) { Write-Host "FAILED ($(Get-Stage $jb.out)): $($jb.out)  -> results\_logs\$($jb.out -replace '[\\/:]', '_').log.err" }
  if ($bad.Count -gt 0) { Write-Host "$($bad.Count) $What runs FAILED; rename failed folders (append _failed) and re-run."; exit 1 }
}

# ---------------- GPU: fold models and their trust maps ----------------
foreach ($fk in $Folds.Keys) {
  $modelDir = "results\trust_t_v1cf_f$fk"
  if (-not (Test-Done $modelDir @("trust_model_trained"))) {
    $cmdArgs = @("-m", "scripts.train_trust", "--data", "cache/trust_data_v1_nyu", "cache/trust_data_v1_hammer",
                 "--regime", "real_sim", "--output", $modelDir, "--steps", "$Steps", "--selection", "error", "--no-features",
                 "--hammer-fold", $Folds[$fk])
    Invoke-Step "train T v1 fold $fk [$($Folds[$fk])]" { & $py @cmdArgs }
  }
  $tcDir = "cache\trust_t_v1cf_f${fk}_hammer_dev_v1"
  if (-not (Test-Done $tcDir @("trust_cache_verified"))) {
    $cmdArgs = @("-m", "scripts.infer_trust", "--model", $modelDir, "--data", "cache/trust_data_v1_hammer", "--output", $tcDir)
    Invoke-Step "trust maps fold $fk" { & $py @cmdArgs }
  }
}
if (-not (Test-Done $CfCache @("trust_cache_verified"))) {
  Invoke-Step "assemble cross-fitted cache + conformal thresholds" { & $py -m scripts.crossfit_trust }
}

# ---------------- GPU: PromptDA-L (method v1) with the cross-fitted mask ----------------
$pdaOut = "cache\promptda_vitl_trust_v1cf_prior_hammer_dev_v1"
foreach ($stepDef in @(@{ dir = "${pdaOut}_preflight"; lim = @("--limit", "3") }, @{ dir = $pdaOut; lim = @() })) {
  if (Test-Done $stepDef.dir @("real_dev_verified")) { continue }
  Assert-FreeGB "D" 6
  $cmdArgs = @("-m", "scripts.cache_promptda", "--config", "configs/promptda/hammer_vitl_trust_prior.yaml", "--output", $stepDef.dir,
               "--set", "trust_cache=$CfCache", "--set", "trust_threshold=per_item") + $stepDef.lim
  Invoke-Step "PromptDA-L + cross-fitted T -> $($stepDef.dir)" { & $py @cmdArgs }
}
$evalJobs = @()
foreach ($sp in @("val", "train")) {
  $evalOut = "results/hammer_dev_promptda_vitl_trust_v1cf_prior_${sp}_v1"
  $evalJobs += @{ module = "tp.eval.run_eval"; out = $evalOut;
    args = "--config configs/hammer_dev_eval.yaml --method cached_prediction --split $sp --output $evalOut " +
           "--set prediction_cache=$pdaOut --set expected_model=PromptDA-L --set method_label=promptda_vitl_trust_v1cf_prior" }
}

# ---------------- GPU (optional): completion baselines, raw and cross-fitted T mask, all 10 scenes ----------------
if ($WithBaselines) {
  $Reg = Get-Content configs\baselines\registry.json -Raw | ConvertFrom-Json
  foreach ($mk in $Models) {
    $label = $Reg.models.$mk.label
    if (-not $label) { Write-Host "Unknown model '$mk'."; exit 1 }
    foreach ($vd in @(@{ v = "raw"; tag = "_all"; extra = @() }, @{ v = "tmask"; tag = "_cfall"; extra = @("--trust-cache", $CfCache) })) {
      $cacheDir = "cache\bl_${mk}_$($vd.v)$($vd.tag)_hammer_dev_v1"
      foreach ($stepDef in @(@{ dir = "${cacheDir}_preflight"; lim = @("--limit", "3") }, @{ dir = $cacheDir; lim = @() })) {
        if (Test-Done $stepDef.dir @("real_dev_verified")) { continue }
        Assert-FreeGB "D" 6
        $cmdArgs = @("-m", "scripts.cache_completion", "--model", $mk, "--dataset", "hammer", "--variant", $vd.v,
                     "--train-scope", "all", "--tag", $vd.tag, "--output", $stepDef.dir) + $vd.extra + $stepDef.lim
        Invoke-Step "$label [$($vd.v)$($vd.tag)] hammer -> $($stepDef.dir)" { & $bpy @cmdArgs }
      }
      foreach ($sp in @("val", "train")) {
        $evalOut = "results/hammer_dev_bl_${mk}_$($vd.v)$($vd.tag)_${sp}_v1"
        $evalJobs += @{ module = "tp.eval.run_eval"; out = $evalOut;
          args = "--config configs/hammer_dev_eval.yaml --method cached_prediction --split $sp --output $evalOut " +
                 "--set prediction_cache=$cacheDir --set expected_model=$label --set method_label=bl_${mk}_$($vd.v)$($vd.tag)" }
      }
    }
  }
}

# ---------------- CPU: evaluation and the pre-registered family ----------------
Assert-FreeGB "D" 4
Invoke-Parallel $evalJobs $EvalOk "evaluation"
Invoke-Step "pre-registered Holm family (cross-fitted, 10 scenes)" { & $py -m scripts.holm_family --spec configs/baselines/holm_family_cf_v1.yaml }
Write-Host ("ALL CROSS-FITTING STEPS SUCCEEDED ({0:hh\:mm\:ss} elapsed). Tell Claude." -f ((Get-Date) - $startTime))
