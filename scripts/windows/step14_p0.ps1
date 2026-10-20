# Mainline P0, last two checks (docs/plan_omnidc.md sec. 15). Needs the Hypersim download (step13) to be complete.
#   D3  synthesised Hypersim errors vs real NYU / HAMMER errors (CPU, about 15 min) -> docs\hypersim_synthesis.md
#   D2  OMNI-DC + gate: open-gate check and 3 training steps on this GPU (a few minutes) -> docs\omnidc_train_smoke.md
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step14_p0.ps1
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
. "$PSScriptRoot\common.ps1"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$bpy = "D:\Python_Script\TrustPrompt\.venvs\baselines\Scripts\python.exe"
$startTime = Get-Date
$stages = Get-Content data\source_cache\hypersim_v1\stages.jsonl | ForEach-Object { ($_ | ConvertFrom-Json).stage }
if (-not ($stages -contains "frames")) { Write-Host "Hypersim stage B not finished: run scripts\windows\step13_hypersim.ps1 first."; exit 1 }
Invoke-Step "D3: synthesis statistics" { & $py -m scripts.hypersim_synthesis_stats }
Invoke-Step "D2: OMNI-DC + gate training smoke" { & $bpy -m scripts.omnidc_train_smoke }
Write-Host ("P0 CHECKS DONE ({0:hh\:mm\:ss} elapsed): docs\hypersim_synthesis.md, docs\omnidc_train_smoke.md. Tell Claude." -f ((Get-Date) - $startTime))
