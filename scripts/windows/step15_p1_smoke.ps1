# Mainline P1, laptop check before renting a GPU (docs/plan_omnidc.md sec. 17-18): runs scripts\train_omnidc.py in its
# --smoke mode for f0 / f1 / f2, a resume and a short pilot (about 10 minutes on the 8 GB card) -> docs\omnidc_p1_smoke.md
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step15_p1_smoke.ps1
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
. "$PSScriptRoot\common.ps1"
$bpy = "D:\Python_Script\TrustPrompt\.venvs\baselines\Scripts\python.exe"
$startTime = Get-Date
$env:MKL_THREADING_LAYER = "SEQUENTIAL"   # torch + Anaconda MKL in one process (OMP Error #15, progress 2026-10-14)
Assert-FreeGB "D" 4
& $bpy -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pytest') else 1)"
if ($LASTEXITCODE -eq 0) { Invoke-Step "P1 unit tests" { & $bpy -m pytest -q tests\test_omnidc_train.py } } else { Write-Host "pytest not in .venvs\baselines: unit tests skipped (they passed on Claude's side)" }
Invoke-Step "P1 training-script smoke test" { & $bpy -m scripts.omnidc_p1_smoke }
Write-Host ("P1 SMOKE DONE ({0:hh\:mm\:ss} elapsed): docs\omnidc_p1_smoke.md. Tell Claude." -f ((Get-Date) - $startTime))
