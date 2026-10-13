# T v0 (module B) end to end, train/val only (test never read):
#   A (GPU)  PromptDA-L with the Gate mask (missing control from R14) + frozen ViT-S encoder features
#   B (CPU)  T training arrays: real prompts (train/val) + 4 simulated variants per train sample, in shards
#   C (GPU)  train T in 4 regimes (real / sim / real_sim / real_sim without encoder features), cache trust maps,
#            pick the regime with the best train-holdout AUROC, PromptDA-L with that T mask
#   D (CPU)  evaluations in parallel, then tables (docs/trust_v0_results.md and the dev tables)
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step5_t_v0.ps1 [-Parallel 6] [-Steps 4000]
# Finished steps are skipped on re-run; success is judged from receipt stages only.
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

# ---------------- A: GPU ----------------
foreach ($ds in @("nyu", "hammer")) {
  $cfgSet = $Sets[$ds]
  Invoke-PdaCache "configs/promptda/$($ds)_vitl_gate_prior.yaml" (Join-Path $cfgSet.pdaCache "promptda_vitl_gate_prior_$($cfgSet.tag)_dev_v1") @()
  $featDir = "cache\encoder_vits_$($cfgSet.tag)_dev_v1"
  foreach ($fd in @(@{ dir = "${featDir}_preflight"; lim = @("--limit", "3") }, @{ dir = $featDir; lim = @() })) {
    if (Test-Done $fd.dir @("features_verified")) { continue }
    $cmdArgs = @("-m", "scripts.cache_encoder_features", "--config", "configs/trust/encoder_$ds.yaml", "--output", $fd.dir) + $fd.lim
    Invoke-Step "encoder features $ds -> $($fd.dir)" { & $py @cmdArgs }
  }
}

# ---------------- B: CPU, T training arrays ----------------
Assert-FreeGB "D" 4
$dataJobs = @()
foreach ($ds in @("nyu", "hammer")) {
  $cfgSet = $Sets[$ds]
  for ($k = 0; $k -lt $cfgSet.shards; $k++) {
    $dataJobs += @{ module = "scripts.build_trust_data"; out = "cache\trust_data_v0_$($cfgSet.tag)\shard_${k}of$($cfgSet.shards)";
      args = "--eval-config $($cfgSet.eval) --output cache/trust_data_v0_$($cfgSet.tag) --sim-variants 4 --shard $k --shards $($cfgSet.shards)" }
  }
}
Invoke-Parallel $dataJobs @("trust_data_verified") "T data"

# ---------------- C: GPU, train / infer / PromptDA with T ----------------
$dataDirs = @("cache/trust_data_v0_nyu", "cache/trust_data_v0_hammer")
$featDirs = @("cache/encoder_vits_nyu_dev_v1", "cache/encoder_vits_hammer_dev_v1")
foreach ($rg in $Regimes) {
  $modelDir = "results\trust_t_v0_$rg"
  if (-not (Test-Done $modelDir @("trust_model_trained"))) {
    $base = if ($rg -eq "real_sim_nofeat") { "real_sim" } else { $rg }
    $cmdArgs = @("-m", "scripts.train_trust", "--data") + $dataDirs + @("--features") + $featDirs + @("--regime", $base, "--output", $modelDir, "--steps", "$Steps")
    if ($rg -eq "real_sim_nofeat") { $cmdArgs += "--no-features" }
    Invoke-Step "train T v0 [$rg]" { & $py @cmdArgs }
  }
  foreach ($ds in @("nyu", "hammer")) {
    $tc = "cache\trust_t_v0_${rg}_$($Sets[$ds].tag)_dev_v1"
    if (Test-Done $tc @("trust_cache_verified")) { continue }
    $cmdArgs = @("-m", "scripts.infer_trust", "--model", $modelDir, "--data", "cache/trust_data_v0_$($Sets[$ds].tag)",
                 "--features", "cache/encoder_vits_$($Sets[$ds].tag)_dev_v1", "--output", $tc)
    Invoke-Step "T v0 [$rg] trust maps $ds" { & $py @cmdArgs }
  }
}
$bestRegime = $null; $bestAuc = -1
foreach ($rg in $Regimes) {
  $auc = (Get-Content "results\trust_t_v0_$rg\receipt.json" -Raw | ConvertFrom-Json).best.selection_auroc
  Write-Host ("T v0 [{0}] train-holdout AUROC {1:N4}" -f $rg, $auc)
  if ($auc -gt $bestAuc) { $bestAuc = $auc; $bestRegime = $rg }
}
Write-Host "selected regime: $bestRegime (by train-holdout AUROC only; val not used)"
Set-Content -Path "results\trust_t_v0_selected_regime.txt" -Value $bestRegime
foreach ($ds in @("nyu", "hammer")) {
  $cfgSet = $Sets[$ds]
  $tc = "cache/trust_t_v0_${bestRegime}_$($cfgSet.tag)_dev_v1"
  Invoke-PdaCache "configs/promptda/$($ds)_vitl_trust_prior.yaml" (Join-Path $cfgSet.pdaCache "promptda_vitl_trust_prior_${bestRegime}_$($cfgSet.tag)_dev_v1") @("--set", "trust_cache=$tc")
}

# ---------------- D: CPU evaluations ----------------
$fill = "fill={grow: 3, ring: 10, min_ring: 50, min_area: 20}"
$evalJobs = @()
foreach ($ds in @("hammer", "nyu")) {
  $cfgSet = $Sets[$ds]
  foreach ($sp in @("train", "val")) {
    foreach ($rg in $Regimes) {
      $runLabel = "trustfill_t_v0_${rg}_v1"
      $evalJobs += @{ module = "tp.eval.run_eval"; out = "results/$($cfgSet.prefix)_${runLabel}_${sp}_v1";
        args = "--config $($cfgSet.eval) --method trustfill --split $sp --output results/$($cfgSet.prefix)_${runLabel}_${sp}_v1 " +
               "--set trust_source=cached_trust --set trust_cache=cache/trust_t_v0_${rg}_$($cfgSet.tag)_dev_v1 --set fill_rule=v1 --set method_label=$runLabel --set `"$fill`"" }
    }
    foreach ($pv in @(@{ label = "promptda_vitl_gate_prior"; cache = "promptda_vitl_gate_prior_$($cfgSet.tag)_dev_v1" },
                      @{ label = "promptda_vitl_trust_prior"; cache = "promptda_vitl_trust_prior_${bestRegime}_$($cfgSet.tag)_dev_v1" })) {
      $cacheDir = Join-Path $cfgSet.pdaCache $pv.cache
      $evalJobs += @{ module = "tp.eval.run_eval"; out = "results/$($cfgSet.prefix)_$($pv.label)_${sp}_v1";
        args = "--config $($cfgSet.eval) --method cached_prediction --split $sp --output results/$($cfgSet.prefix)_$($pv.label)_${sp}_v1 " +
               "--set prediction_cache=$cacheDir --set expected_model=PromptDA-L --set method_label=$($pv.label)" }
    }
  }
}
Assert-FreeGB "D" 6
Invoke-Parallel $evalJobs $EvalOk "evaluation"
Invoke-Step "NYU tables" { & $py -m scripts.summarize_dev }
Invoke-Step "HAMMER tables" { & $py -m scripts.summarize_dev --dataset hammer }
Invoke-Step "T v0 tables" { & $py -m scripts.summarize_trust }
Write-Host ("ALL T v0 STEPS SUCCEEDED ({0:hh\:mm\:ss} elapsed). Tell Claude." -f ((Get-Date) - $startTime))
