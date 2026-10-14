# R15 (after T v1): (1) repair the masked prompt with module C (ring fits, rule v1) before PromptDA-L instead of
# filling it with D_g; (2) high-recall threshold for T v1 on NYU (train-holdout rule: recall >= 0.95 with <= 10%
# false flags; HAMMER cannot meet it and keeps the model threshold). Train/val only; test never read.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step7_repair.ps1 [-Parallel 6]
# Finished steps are skipped on re-run; success is judged from receipt stages only.
param([int]$Parallel = 6)
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$EvalOk = @("small_scale_validation_passed", "completed_with_invalid_predictions")
$startTime = Get-Date
$Sets = @{
  nyu    = @{ eval = "configs/mirror3d_dev_eval.yaml"; prefix = "mirror3d_dev"; pdaCache = "cache";                     tag = "nyu" }
  hammer = @{ eval = "configs/hammer_dev_eval.yaml";   prefix = "hammer_dev";   pdaCache = "E:\TrustPrompt_data\cache"; tag = "hammer" }
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

# ---------------- GPU: PromptDA-L caches ----------------
$hrFile = "results\trust_t_v1_highrecall_threshold.json"
if (-not (Test-Path $hrFile)) { Write-Host "$hrFile missing (made by scripts.select_recall_threshold)."; exit 1 }
$hrNyu = (Get-Content $hrFile -Raw | ConvertFrom-Json).datasets.mirror3d_nyu.chosen.threshold
Write-Host "NYU high-recall threshold (train holdout): $hrNyu"
$tcNyu = "cache/trust_t_v1_real_sim_nofeat_nyu_dev_v1"; $tcHam = "cache/trust_t_v1_real_sim_nofeat_hammer_dev_v1"
$runs = @(
  @{ ds = "nyu";    label = "promptda_vitl_repair_oracle_prior";  cfg = "repair_prior"; extra = @("--set", "mask_source=oracle_c") },
  @{ ds = "nyu";    label = "promptda_vitl_repair_t_v1_prior";    cfg = "repair_prior"; extra = @("--set", "mask_source=trust", "--set", "trust_cache=$tcNyu") },
  @{ ds = "nyu";    label = "promptda_vitl_repair_t_v1_hr_prior"; cfg = "repair_prior"; extra = @("--set", "mask_source=trust", "--set", "trust_cache=$tcNyu", "--set", "trust_threshold=$hrNyu") },
  @{ ds = "nyu";    label = "promptda_vitl_trust_v1_hr_prior";    cfg = "trust_prior";  extra = @("--set", "trust_cache=$tcNyu", "--set", "trust_threshold=$hrNyu") },
  @{ ds = "hammer"; label = "promptda_vitl_repair_t_v1_prior";    cfg = "repair_prior"; extra = @("--set", "mask_source=trust", "--set", "trust_cache=$tcHam") },
  @{ ds = "hammer"; label = "promptda_vitl_repair_oracle_prior";  cfg = "repair_prior"; extra = @("--set", "mask_source=oracle_c") }
)
foreach ($rn in $runs) {
  $cfgSet = $Sets[$rn.ds]
  Invoke-PdaCache "configs/promptda/$($rn.ds)_vitl_$($rn.cfg).yaml" (Join-Path $cfgSet.pdaCache "$($rn.label)_$($cfgSet.tag)_dev_v1") $rn.extra
}

# ---------------- CPU: evaluations ----------------
$evalJobs = @()
foreach ($rn in $runs) {
  $cfgSet = $Sets[$rn.ds]
  $cacheDir = Join-Path $cfgSet.pdaCache "$($rn.label)_$($cfgSet.tag)_dev_v1"
  foreach ($sp in @("train", "val")) {
    $evalOut = "results/$($cfgSet.prefix)_$($rn.label)_${sp}_v1"
    $evalJobs += @{ module = "tp.eval.run_eval"; out = $evalOut;
      args = "--config $($cfgSet.eval) --method cached_prediction --split $sp --output $evalOut " +
             "--set prediction_cache=$cacheDir --set expected_model=PromptDA-L --set method_label=$($rn.label)" }
  }
}
Assert-FreeGB "D" 3
Invoke-Parallel $evalJobs $EvalOk "evaluation"
Invoke-Step "T tables" { & $py -m scripts.summarize_trust }
Write-Host ("ALL R15 REPAIR STEPS SUCCEEDED ({0:hh\:mm\:ss} elapsed). Tell Claude." -f ((Get-Date) - $startTime))
