# P1 coupling analysis (docs/plan_omnidc.md sec. 23): inference only on this GPU, free, about 30-60 min.
# Replaces F1's / F0's input confidence at inference to test whether F1's loss on correct readings and its correction of
# wrong ones are coupled through the trust channel -> docs\omnidc_p1_coupling.md
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step20_p1_coupling.ps1
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
. "$PSScriptRoot\common.ps1"
$bpy = "D:\Python_Script\TrustPrompt\.venvs\baselines\Scripts\python.exe"
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
$startTime = Get-Date
Invoke-Step "coupling analysis (F1 then F0, confidence replaced at inference)" { & $bpy -m scripts.p1_coupling }
Write-Host ("P1 COUPLING DONE ({0:hh\:mm\:ss} elapsed): docs\omnidc_p1_coupling.md. Tell Claude." -f ((Get-Date) - $startTime))
