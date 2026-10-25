# P1 v2 laptop pre-check (docs/plan_omnidc.md sec. 21): free, on this 8 GB GPU. Trains P1's F1 recipe twice for 1500
# steps (256 x 320, batch 1) - control with the two-sided cell label, candidate with the soft cell label - then
# compares their input confidence on NYU / HAMMER dev with OMNI-DC v1.1 -> docs\omnidc_p1v2_precheck.md
# Roughly 1.5-2.5 h per training run plus about 15 min of evaluation; a finished run is not trained again.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step18_p1v2_precheck.ps1
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
. "$PSScriptRoot\common.ps1"
$bpy = "D:\Python_Script\TrustPrompt\.venvs\baselines\Scripts\python.exe"
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
$startTime = Get-Date
& $bpy -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pytest') else 1)"
if ($LASTEXITCODE -eq 0) { Invoke-Step "label unit tests" { & $bpy -m pytest -q tests\test_omnidc_labels.py } }
foreach ($v in @("f1", "f1s")) {
  $done = Get-ChildItem results -Directory -Filter "p1v2pre_${v}_*" | Where-Object { (Get-Stage $_.FullName) -eq "training_complete" }
  if ($done) { Write-Host "$v already trained: $($done[-1].Name)"; continue }
  Assert-FreeGB "D" 6
  Invoke-Step "pre-check training $v (1500 steps)" { & $bpy -m scripts.train_omnidc --config configs/omnidc/p1v2_precheck.yaml --variant $v }
}
Invoke-Step "pre-check evaluation" { & $bpy -m scripts.p1_label_precheck }
Write-Host ("P1 V2 PRECHECK DONE ({0:hh\:mm\:ss} elapsed): docs\omnidc_p1v2_precheck.md. Tell Claude." -f ((Get-Date) - $startTime))
