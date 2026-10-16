# Depth-completion baselines with and without T v1 (plan revision v2 sec. 3.7.5; docs/baselines.md).
# Run scripts\windows\setup_baselines.ps1 once first. Train/val only; test never read.
#   GPU: for each model x dataset (nyu, hammer) x prompt (raw, tmask, oracle_mask): 3-image preflight, then val +
#        train-holdout scenes  -> cache\bl_<model>_<variant>_<dataset>_dev_v1
#   CPU: evaluation of every cache (val; train restricted to holdout scenes), tables, pre-registered Holm family
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step10_baselines.ps1 [-Models priorda,omnidc] [-WithMarigold] [-Parallel 6]
param([string[]]$Models = @("priorda", "omnidc", "cdm", "lingbot", "any2full"), [switch]$WithMarigold, [int]$Parallel = 6)
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$bpy = "D:\Python_Script\TrustPrompt\.venvs\baselines\Scripts\python.exe"
if (-not (Test-Path $bpy)) { Write-Host "Run scripts\windows\setup_baselines.ps1 first."; exit 1 }
if ($WithMarigold -and -not ($Models -contains "marigold_dc")) { $Models += "marigold_dc" }
$EvalOk = @("small_scale_validation_passed", "completed_with_invalid_predictions")
$startTime = Get-Date
$Reg = Get-Content configs\baselines\registry.json -Raw | ConvertFrom-Json
$Sets = @{
  nyu    = @{ eval = "configs/mirror3d_dev_eval.yaml"; prefix = "mirror3d_dev" }
  hammer = @{ eval = "configs/hammer_dev_eval.yaml";   prefix = "hammer_dev" }
}
$Variants = @("raw", "tmask", "oracle_mask")
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

# ---------------- GPU: baseline caches ----------------
$evalJobs = @()
foreach ($mk in $Models) {
  $label = $Reg.models.$mk.label
  if (-not $label) { Write-Host "Unknown model '$mk' (see configs\baselines\registry.json)."; exit 1 }
  Invoke-Step "weights of $mk present and verified" { & $bpy -c "from baselines.completion import verify_weights; verify_weights('$mk'); print('ok')" }
  foreach ($ds in @("nyu", "hammer")) {
    foreach ($vr in $Variants) {
      $cacheDir = "cache\bl_${mk}_${vr}_${ds}_dev_v1"
      foreach ($stepDef in @(@{ dir = "${cacheDir}_preflight"; lim = @("--limit", "3") }, @{ dir = $cacheDir; lim = @() })) {
        if (Test-Done $stepDef.dir @("real_dev_verified")) { continue }
        Assert-FreeGB "D" 6
        $cmdArgs = @("-m", "scripts.cache_completion", "--model", $mk, "--dataset", $ds, "--variant", $vr, "--output", $stepDef.dir) + $stepDef.lim
        Invoke-Step "$label [$vr] $ds -> $($stepDef.dir)" { & $bpy @cmdArgs }
      }
      foreach ($sp in @("val", "train")) {
        $evalOut = "results/$($Sets[$ds].prefix)_bl_${mk}_${vr}_${sp}_v1"
        $evalJobs += @{ module = "tp.eval.run_eval"; out = $evalOut;
          args = "--config $($Sets[$ds].eval) --method cached_prediction --split $sp --output $evalOut " +
                 "--set prediction_cache=$cacheDir --set expected_model=$label --set method_label=bl_${mk}_${vr} --set train_scope=holdout" }
      }
    }
  }
}

# ---------------- CPU: evaluation, tables, Holm family ----------------
Assert-FreeGB "D" 4
Invoke-Parallel $evalJobs $EvalOk "evaluation"
Invoke-Step "baseline tables" { & $py -m scripts.summarize_baselines }
Invoke-Step "pre-registered Holm family" { & $py -m scripts.holm_family }
Write-Host ("ALL BASELINE STEPS SUCCEEDED ({0:hh\:mm\:ss} elapsed). Tell Claude." -f ((Get-Date) - $startTime))
