# Hypersim subset for the mainline structured-error synthesis (docs/plan_omnidc.md sec. 4; P0-D3).
# Public data, CC BY-SA 3.0, no agreement to accept. Scenes: configs\hypersim\selection_v1.json (85 scenes with
# mirror objects). Files go to data\source_cache\hypersim_v1\ (not tracked by git).
#   1. plan: read each scene ZIP's directory, print exact sizes; stop if stage A + the stage B upper bound exceed
#      -MaxGB (40: the range agreed on 2026-10-19)
#   2. stage A (semantic maps of all 14,116 public frames), frame selection, stage B (image + depth of the selected
#      frames). Resumable: re-run after an interruption. -PlanOnly stops after 1.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step13_hypersim.ps1 [-PlanOnly] [-Workers 8] [-MaxGB 40]
param([switch]$PlanOnly, [int]$Workers = 8, [double]$MaxGB = 40)
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
. "$PSScriptRoot\common.ps1"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$startTime = Get-Date
Invoke-Step "plan: scene ZIP directories and exact sizes" { & $py -m scripts.hypersim_fetch plan --max-gb $MaxGB }
if ($PlanOnly) { Write-Host "PLAN DONE."; exit 0 }
Invoke-Step "stage A: semantic maps" { & $py -m scripts.hypersim_fetch semantic --workers $Workers }
Invoke-Step "frame selection (mirror >= 1 % or window >= 5 %, + 25 % clean)" { & $py -m scripts.hypersim_fetch select }
Invoke-Step "stage B: tone-mapped images + depth of the selected frames" { & $py -m scripts.hypersim_fetch frames --workers $Workers }
Write-Host ("HYPERSIM DOWNLOAD DONE ({0:hh\:mm\:ss} elapsed). Tell Claude." -f ((Get-Date) - $startTime))
