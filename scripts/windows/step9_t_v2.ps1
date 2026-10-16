# T v2 (R16 follow-up): 3-class trust (clean / measured too far / measured too near) so the upper bound is only
# applied where the prompt measured too far. Train/val only; test never read.
#   GPU: train T v2 in 2 regimes (real_sim with feature dropout, real_sim without features), cache trust + direction maps
#   CPU: direction diagnostics; 20 evaluations of far-type-only capping (GT sign / T v2 sign) on existing PromptDA caches
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step9_t_v2.ps1 [-Parallel 6] [-Steps 4000]
param([int]$Parallel = 6, [int]$Steps = 4000)
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$EvalOk = @("small_scale_validation_passed", "completed_with_invalid_predictions")
$startTime = Get-Date
$Sets = @{
  nyu    = @{ eval = "configs/mirror3d_dev_eval.yaml"; prefix = "mirror3d_dev"; tag = "nyu";    oracle = "cache/promptda_vitl_oracle_prior_nyu_dev_v1" }
  hammer = @{ eval = "configs/hammer_dev_eval.yaml";   prefix = "hammer_dev";   tag = "hammer"; oracle = "cache/promptda_vitl_oracle_prior_hammer_dev_v2" }
}
$Regimes = @("real_sim_nofeat", "real_sim")
function Get-Stage([string]$Dir) {
  $r = Join-Path $Dir "receipt.json"
  if (Test-Path $r) { return (Get-Content $r -Raw | ConvertFrom-Json).stage } else { return "missing" }
}
function Invoke-Step([string]$Label, [scriptblock]$Body) {
  Write-Host "=== $Label  ($(Get-Date -Format HH:mm:ss))"
  & $Body
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $Label (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}
# $true = already done; $false = to do; exits when a failed/partial folder is in the way.
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
function Invoke-PdaCache([string]$CfgPath, [string]$OutDir, [string[]]$Extra) {
  foreach ($stepDef in @(@{ dir = "${OutDir}_preflight"; lim = @("--limit", "3") }, @{ dir = $OutDir; lim = @() })) {
    if (Test-Done $stepDef.dir @("real_dev_verified")) { continue }
    if ($OutDir.StartsWith("E:")) { Assert-FreeGB "E" 4 }
    $cmdArgs = @("-m", "scripts.cache_promptda", "--config", $CfgPath, "--output", $stepDef.dir) + $stepDef.lim + $Extra
    Invoke-Step "PromptDA cache $CfgPath -> $($stepDef.dir)" { & $py @cmdArgs }
  }
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

# ---------------- GPU: train T v2, cache trust + direction ----------------
$dataDirs = @("cache/trust_data_v1_nyu", "cache/trust_data_v1_hammer")
$featDirs = @("cache/encoder_vits_nyu_dev_v1", "cache/encoder_vits_hammer_dev_v1")
foreach ($rg in $Regimes) {
  $modelDir = "results\trust_t_v2_$rg"
  if (-not (Test-Done $modelDir @("trust_model_trained"))) {
    $cmdArgs = @("-m", "scripts.train_trust", "--data") + $dataDirs + @("--features") + $featDirs +
               @("--regime", "real_sim", "--output", $modelDir, "--steps", "$Steps", "--selection", "error", "--classes", "3")
    if ($rg -eq "real_sim_nofeat") { $cmdArgs += "--no-features" } else { $cmdArgs += @("--feat-dropout", "0.3", "--feat-drop-sample", "0.2") }
    Invoke-Step "train T v2 [$rg]" { & $py @cmdArgs }
  }
  foreach ($ds in @("nyu", "hammer")) {
    $tcDir = "cache\trust_t_v2_${rg}_$($Sets[$ds].tag)_dev_v1"
    if (Test-Done $tcDir @("trust_cache_verified")) { continue }
    $cmdArgs = @("-m", "scripts.infer_trust", "--model", $modelDir, "--data", "cache/trust_data_v1_$($Sets[$ds].tag)",
                 "--features", "cache/encoder_vits_$($Sets[$ds].tag)_dev_v1", "--output", $tcDir)
    Invoke-Step "T v2 [$rg] maps $ds" { & $py @cmdArgs }
  }
}
$diagCaches = @(); foreach ($rg in $Regimes) { $diagCaches += "v2_$rg=cache/trust_t_v2_${rg}_{tag}_dev_v1" }
Invoke-Step "direction diagnostics" { & $py -m scripts.trust_direction_diagnostics --data cache/trust_data_v1_nyu cache/trust_data_v1_hammer --caches @diagCaches }

# ---------------- CPU: far-type-only capping on existing PromptDA caches ----------------
$evalJobs = @()
foreach ($ds in @("hammer", "nyu")) {
  $cfgSet = $Sets[$ds]
  $t1 = "cache/trust_t_v1_real_sim_nofeat_$($cfgSet.tag)_dev_v1"
  $pdaT1 = "cache/promptda_vitl_trust_v1_prior_real_sim_nofeat_$($cfgSet.tag)_dev_v1"
  $runs = @(@{ label = "promptda_vitl_oracle_prior_capfar_oracle"; cache = $cfgSet.oracle; cap = "{mask: oracle_c, direction: oracle}" })
  foreach ($rg in $Regimes) {
    $dc = "cache/trust_t_v2_${rg}_$($cfgSet.tag)_dev_v1"
    $runs += @{ label = "promptda_vitl_oracle_prior_capfar_t_v2_$rg"; cache = $cfgSet.oracle; cap = "{mask: oracle_c, direction: trust_far, direction_cache: $dc}" }
    $runs += @{ label = "promptda_vitl_trust_v1_prior_capfar_t_v2_$rg"; cache = $pdaT1; cap = "{mask: trust, trust_cache: $t1, direction: trust_far, direction_cache: $dc}" }
  }
  foreach ($rn in $runs) {
    foreach ($sp in @("train", "val")) {
      $evalOut = "results/$($cfgSet.prefix)_$($rn.label)_${sp}_v1"
      $evalJobs += @{ module = "tp.eval.run_eval"; out = $evalOut;
        args = "--config $($cfgSet.eval) --method cached_prediction --split $sp --output $evalOut " +
               "--set prediction_cache=$($rn.cache) --set expected_model=PromptDA-L --set method_label=$($rn.label) --set `"bound_cap=$($rn.cap)`"" }
    }
  }
}
Assert-FreeGB "D" 4
Invoke-Parallel $evalJobs $EvalOk "evaluation"
Invoke-Step "bound tables" { & $py -m scripts.summarize_bounds }
Write-Host ("ALL T v2 STEPS SUCCEEDED ({0:hh\:mm\:ss} elapsed). Tell Claude." -f ((Get-Date) - $startTime))
