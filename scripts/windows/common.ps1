# Helpers shared by the step scripts (dot-source: . "$PSScriptRoot\common.ps1"). Same behaviour as the copies inside
# step10_baselines.ps1 / step11_crossfit.ps1, which predate this file and are left as they ran.
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
# Run CPU jobs @{ module; out; args } with python $Py, $Parallel at a time; skip jobs whose receipt is in $OkStages.
function Invoke-Parallel($JobList, [string[]]$OkStages, [string]$What, [string]$Py, [int]$Parallel) {
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
    $procs += Start-Process -FilePath $Py -ArgumentList "-m $($jb.module) $($jb.args)" -NoNewWindow -PassThru `
      -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
  }
  $procs | Wait-Process
  foreach ($ev in @("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")) { Remove-Item "env:$ev" -ErrorAction SilentlyContinue }
  $bad = @($JobList | Where-Object { -not ($OkStages -contains (Get-Stage $_.out)) })
  foreach ($jb in $bad) { Write-Host "FAILED ($(Get-Stage $jb.out)): $($jb.out)  -> results\_logs\$($jb.out -replace '[\\/:]', '_').log.err" }
  if ($bad.Count -gt 0) { Write-Host "$($bad.Count) $What runs FAILED; rename failed folders (append _failed) and re-run."; exit 1 }
}
