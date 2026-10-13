# T v1 (module B): same network, selection aligned with the goal. Train/val only (test never read).
#   B (CPU)  T arrays v1 = v0 + GT depth, GT validity and main-scope mask (for selection only), in shards
#   C (GPU)  train 4 regimes with replacement-error selection of checkpoint, threshold and post-processing on
#            train-holdout scenes; encoder-feature dropout; cache trust maps; pick the regime with the lowest
#            holdout replacement error; PromptDA-L with that T v1 mask
#   D (CPU)  evaluations in parallel; tables (docs/trust_v0_results.md now lists v0 and v1) and mask diagnostics
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step6_t_v1.ps1 [-Parallel 6] [-Steps 4000]
# Reuses the v0 encoder features. Finished steps are skipped on re-run; success is judged from receipt stages only.
param([int]$Parallel = 6, [int]$Steps = 4000)
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$EvalOk = @("small_scale_validation_passed", "completed_with_invalid_predictions")
$startTime = Get-Date
$Regimes = @("real", "sim", "real_sim", "real_sim_nofeat")
$Sets = @{
  nyu    = @{ eval = "configs/mirror3d_dev_eval.yaml"; prefix = "mirror3d_dev"; pdaCache = "cache";                     tag = "nyu";    shards = 2 }
  hammer = @{ eval = "configs/hammer_dev_eval.yaml";   prefix = "hammer_dev";   pdaCache = "E:\TrustPrompt_data\cache"; tag = "hammer"; shards = 6 }
}
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

# ---------------- prerequisites from step 5 ----------------
foreach ($ds in @("nyu", "hammer")) {
  if (-not (Test-Done "cache\encoder_vits_$($Sets[$ds].tag)_dev_v1" @("features_verified"))) { Write-Host "Run step5_t_v0.ps1 first (encoder features missing)."; exit 1 }
}

# ---------------- B: CPU, T arrays v1 ----------------
Assert-FreeGB "D" 5
$dataJobs = @()
foreach ($ds in @("nyu", "hammer")) {
  $cfgSet = $Sets[$ds]
  for ($k = 0; $k -lt $cfgSet.shards; $k++) {
    $dataJobs += @{ module = "scripts.build_trust_data"; out = "cache\trust_data_v1_$($cfgSet.tag)\shard_${k}of$($cfgSet.shards)";
      args = "--eval-config $($cfgSet.eval) --output cache/trust_data_v1_$($cfgSet.tag) --sim-variants 4 --shard $k --shards $($cfgSet.shards) --version v1" }
  }
}
Invoke-Parallel $dataJobs @("trust_data_verified") "T data v1"

# ---------------- C: GPU, train / infer / PromptDA with T v1 ----------------
$dataDirs = @("cache/trust_data_v1_nyu", "cache/trust_data_v1_hammer")
$featDirs = @("cache/encoder_vits_nyu_dev_v1", "cache/encoder_vits_hammer_dev_v1")
foreach ($rg in $Regimes) {
  $modelDir = "results\trust_t_v1_$rg"
  if (-not (Test-Done $modelDir @("trust_model_trained"))) {
    $base = if ($rg -eq "real_sim_nofeat") { "real_sim" } else { $rg }
    $cmdArgs = @("-m", "scripts.train_trust", "--data") + $dataDirs + @("--features") + $featDirs +
               @("--regime", $base, "--output", $modelDir, "--steps", "$Steps", "--selection", "error")
    if ($rg -eq "real_sim_nofeat") { $cmdArgs += "--no-features" } else { $cmdArgs += @("--feat-dropout", "0.3", "--feat-drop-sample", "0.2") }
    Invoke-Step "train T v1 [$rg]" { & $py @cmdArgs }
  }
  foreach ($ds in @("nyu", "hammer")) {
    $tc = "cache\trust_t_v1_${rg}_$($Sets[$ds].tag)_dev_v1"
    if (Test-Done $tc @("trust_cache_verified")) { continue }
    $cmdArgs = @("-m", "scripts.infer_trust", "--model", $modelDir, "--data", "cache/trust_data_v1_$($Sets[$ds].tag)",
                 "--features", "cache/encoder_vits_$($Sets[$ds].tag)_dev_v1", "--output", $tc)
    Invoke-Step "T v1 [$rg] trust maps $ds" { & $py @cmdArgs }
  }
}
$bestRegime = $null; $bestObj = [double]::MaxValue
foreach ($rg in $Regimes) {
  $obj = (Get-Content "results\trust_t_v1_$rg\receipt.json" -Raw | ConvertFrom-Json).best.objective
  Write-Host ("T v1 [{0}] train-holdout replacement error {1:N4}" -f $rg, $obj)
  if ($obj -lt $bestObj) { $bestObj = $obj; $bestRegime = $rg }
}
Write-Host "selected regime: $bestRegime (train-holdout replacement error only; val not used)"
Set-Content -Path "results\trust_t_v1_selected_regime.txt" -Value $bestRegime
foreach ($ds in @("nyu", "hammer")) {
  $cfgSet = $Sets[$ds]
  $tc = "cache/trust_t_v1_${bestRegime}_$($cfgSet.tag)_dev_v1"
  Invoke-PdaCache "configs/promptda/$($ds)_vitl_trust_prior.yaml" (Join-Path $cfgSet.pdaCache "promptda_vitl_trust_v1_prior_${bestRegime}_$($cfgSet.tag)_dev_v1") @("--set", "trust_cache=$tc")
}

# ---------------- D: CPU evaluations ----------------
$fill = "fill={grow: 3, ring: 10, min_ring: 50, min_area: 20}"
$evalJobs = @()
foreach ($ds in @("hammer", "nyu")) {
  $cfgSet = $Sets[$ds]
  foreach ($sp in @("train", "val")) {
    foreach ($rg in $Regimes) {
      $runLabel = "trustfill_t_v1_${rg}_v1"
      $evalJobs += @{ module = "tp.eval.run_eval"; out = "results/$($cfgSet.prefix)_${runLabel}_${sp}_v1";
        args = "--config $($cfgSet.eval) --method trustfill --split $sp --output results/$($cfgSet.prefix)_${runLabel}_${sp}_v1 " +
               "--set trust_source=cached_trust --set trust_cache=cache/trust_t_v1_${rg}_$($cfgSet.tag)_dev_v1 --set fill_rule=v1 --set method_label=$runLabel --set `"$fill`"" }
    }
    $cacheDir = Join-Path $cfgSet.pdaCache "promptda_vitl_trust_v1_prior_${bestRegime}_$($cfgSet.tag)_dev_v1"
    $evalJobs += @{ module = "tp.eval.run_eval"; out = "results/$($cfgSet.prefix)_promptda_vitl_trust_v1_prior_${sp}_v1";
      args = "--config $($cfgSet.eval) --method cached_prediction --split $sp --output results/$($cfgSet.prefix)_promptda_vitl_trust_v1_prior_${sp}_v1 " +
             "--set prediction_cache=$cacheDir --set expected_model=PromptDA-L --set method_label=promptda_vitl_trust_v1_prior" }
  }
}
Assert-FreeGB "D" 6
Invoke-Parallel $evalJobs $EvalOk "evaluation"
Invoke-Step "T tables" { & $py -m scripts.summarize_trust }
$diagCaches = @(); foreach ($ver in @("v0", "v1")) { foreach ($rg in $Regimes) { $diagCaches += "${ver}_$rg=cache/trust_t_${ver}_${rg}_{tag}_dev_v1" } }
Invoke-Step "mask diagnostics" { & $py -m scripts.trust_mask_diagnostics --data cache/trust_data_v1_nyu cache/trust_data_v1_hammer --caches @diagCaches }
Write-Host ("ALL T v1 STEPS SUCCEEDED ({0:hh\:mm\:ss} elapsed). Tell Claude." -f ((Get-Date) - $startTime))
