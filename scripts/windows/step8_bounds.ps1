# R16 items 1-2 (train/val only; test never read):
#   1. upper-bound cap: regenerate the HAMMER PromptDA-L + T v1 cache (deleted with E:) and a train+val HAMMER
#      GT-mask cache, then evaluate each with and without capping flagged pixels at the measured depth;
#   2. conformal threshold (alpha = 0.1, danger = P > 1.2 GT, calibrated on train-holdout scenes;
#      results/trust_t_v1_crc_d20.json): PromptDA-L with that T v1 mask, with and without the cap.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step8_bounds.ps1 [-Parallel 6]
# Finished steps are skipped on re-run; success is judged from receipt stages only.
param([int]$Parallel = 6)
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$EvalOk = @("small_scale_validation_passed", "completed_with_invalid_predictions")
$startTime = Get-Date
$Sets = @{
  nyu    = @{ eval = "configs/mirror3d_dev_eval.yaml"; prefix = "mirror3d_dev"; tag = "nyu" }
  hammer = @{ eval = "configs/hammer_dev_eval.yaml";   prefix = "hammer_dev";   tag = "hammer" }
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

Invoke-Step "OpenMP check (torch + numpy.linalg, sequential MKL)" { & $py -c "import os; os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL'); import numpy as np, torch; torch.zeros(2).cuda(); a=np.random.rand(500,3); print('lstsq ok', np.linalg.lstsq(a, a@np.ones(3), rcond=None)[0].round(6))" }
Invoke-Step "restore/verify PromptDA" { & powershell -ExecutionPolicy Bypass -File scripts\windows\restore_promptda.ps1 }

$crcFile = "results\trust_t_v1_crc_d20.json"
if (-not (Test-Path $crcFile)) { Write-Host "$crcFile missing (scripts.conformal_threshold --danger-tau 0.2)."; exit 1 }
$crc = Get-Content $crcFile -Raw | ConvertFrom-Json
$lamNyu = $crc.datasets.mirror3d_nyu.alphas.'0.1'.lambda_hat; $lamHam = $crc.datasets.hammer.alphas.'0.1'.lambda_hat
Write-Host "conformal thresholds (alpha 0.1): NYU $lamNyu, HAMMER $lamHam"
$tc = @{ nyu = "cache/trust_t_v1_real_sim_nofeat_nyu_dev_v1"; hammer = "cache/trust_t_v1_real_sim_nofeat_hammer_dev_v1" }
$lam = @{ nyu = $lamNyu; hammer = $lamHam }

# ---------------- GPU: PromptDA-L caches ----------------
Assert-FreeGB "D" 8
Invoke-PdaCache "configs/promptda/hammer_vitl_trust_prior.yaml" "cache\promptda_vitl_trust_v1_prior_real_sim_nofeat_hammer_dev_v1" @("--set", "trust_cache=$($tc.hammer)")
Invoke-PdaCache "configs/promptda/hammer_vitl_oracle_prior.yaml" "cache\promptda_vitl_oracle_prior_hammer_dev_v2" @("--set", "splits=[train, val]")
foreach ($ds in @("nyu", "hammer")) {
  Invoke-PdaCache "configs/promptda/$($ds)_vitl_trust_prior.yaml" "cache\promptda_vitl_trust_v1_crc10_prior_real_sim_nofeat_$($Sets[$ds].tag)_dev_v1" @("--set", "trust_cache=$($tc[$ds])", "--set", "trust_threshold=$($lam[$ds])")
}

# ---------------- CPU: evaluations ----------------
$runs = @(
  @{ ds = "hammer"; label = "promptda_vitl_trust_v1_prior_regen";      cache = "cache/promptda_vitl_trust_v1_prior_real_sim_nofeat_hammer_dev_v1"; cap = "" },
  @{ ds = "hammer"; label = "promptda_vitl_trust_v1_prior_cap";        cache = "cache/promptda_vitl_trust_v1_prior_real_sim_nofeat_hammer_dev_v1"; cap = "trust" },
  @{ ds = "nyu";    label = "promptda_vitl_trust_v1_prior_cap";        cache = "cache/promptda_vitl_trust_v1_prior_real_sim_nofeat_nyu_dev_v1";    cap = "trust" },
  @{ ds = "hammer"; label = "promptda_vitl_oracle_prior_v2";           cache = "cache/promptda_vitl_oracle_prior_hammer_dev_v2";                    cap = "" },
  @{ ds = "hammer"; label = "promptda_vitl_oracle_prior_v2_cap";       cache = "cache/promptda_vitl_oracle_prior_hammer_dev_v2";                    cap = "oracle" },
  @{ ds = "nyu";    label = "promptda_vitl_oracle_prior_cap";          cache = "cache/promptda_vitl_oracle_prior_nyu_dev_v1";                       cap = "oracle" },
  @{ ds = "nyu";    label = "promptda_vitl_trust_v1_crc10_prior";      cache = "cache/promptda_vitl_trust_v1_crc10_prior_real_sim_nofeat_nyu_dev_v1";    cap = "" },
  @{ ds = "nyu";    label = "promptda_vitl_trust_v1_crc10_prior_cap";  cache = "cache/promptda_vitl_trust_v1_crc10_prior_real_sim_nofeat_nyu_dev_v1";    cap = "crc" },
  @{ ds = "hammer"; label = "promptda_vitl_trust_v1_crc10_prior";      cache = "cache/promptda_vitl_trust_v1_crc10_prior_real_sim_nofeat_hammer_dev_v1"; cap = "" },
  @{ ds = "hammer"; label = "promptda_vitl_trust_v1_crc10_prior_cap";  cache = "cache/promptda_vitl_trust_v1_crc10_prior_real_sim_nofeat_hammer_dev_v1"; cap = "crc" }
)
$evalJobs = @()
foreach ($rn in $runs) {
  $cfgSet = $Sets[$rn.ds]
  $capArg = ""
  if ($rn.cap -eq "trust")  { $capArg = " --set `"bound_cap={mask: trust, trust_cache: $($tc[$rn.ds])}`"" }
  if ($rn.cap -eq "crc")    { $capArg = " --set `"bound_cap={mask: trust, trust_cache: $($tc[$rn.ds]), trust_threshold: $($lam[$rn.ds])}`"" }
  if ($rn.cap -eq "oracle") { $capArg = " --set `"bound_cap={mask: oracle_c}`"" }
  foreach ($sp in @("train", "val")) {
    $evalOut = "results/$($cfgSet.prefix)_$($rn.label)_${sp}_v1"
    $evalJobs += @{ module = "tp.eval.run_eval"; out = $evalOut;
      args = "--config $($cfgSet.eval) --method cached_prediction --split $sp --output $evalOut " +
             "--set prediction_cache=$($rn.cache) --set expected_model=PromptDA-L --set method_label=$($rn.label)$capArg" }
  }
}
Assert-FreeGB "D" 4
Invoke-Parallel $evalJobs $EvalOk "evaluation"
Invoke-Step "bound tables" { & $py -m scripts.summarize_bounds }
Write-Host ("ALL R16 STEP-8 RUNS SUCCEEDED ({0:hh\:mm\:ss} elapsed). Tell Claude." -f ((Get-Date) - $startTime))
