# P1 v2 laptop pre-check, long version (docs/plan_omnidc.md sec. 22): free, on this 8 GB GPU. As step18 but 6000 steps
# per run (4x) with the weights kept every 1500 steps, so the trend of the confidence is visible. About 12 h in all
# (two training runs of about 5.5-6 h, then about 30 min of evaluation) -> docs\omnidc_p1v2_precheck_long.md
# A finished run is not trained again; an interrupted run is resumed from its last checkpoint (every 500 steps).
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step19_p1v2_precheck_long.ps1
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
. "$PSScriptRoot\common.ps1"
$bpy = "D:\Python_Script\TrustPrompt\.venvs\baselines\Scripts\python.exe"
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
$Config = "configs/omnidc/p1v2_precheck_long.yaml"
$startTime = Get-Date
foreach ($v in @("f1", "f1s")) {
  $runs = @(Get-ChildItem results -Directory -Filter "p1v2long_${v}_*" | Sort-Object Name)
  if ($runs | Where-Object { (Get-Stage $_.FullName) -eq "training_complete" }) { Write-Host "$v already trained"; continue }
  Assert-FreeGB "D" 8
  $last = $runs | Select-Object -Last 1
  if ($last -and (Test-Path (Join-Path $last.FullName "checkpoints\last.pt"))) {
    Invoke-Step "resume pre-check training $v ($($last.Name))" { & $bpy -m scripts.train_omnidc --config $Config --variant $v --resume "results/$($last.Name)" }
  } else {
    Invoke-Step "pre-check training $v (6000 steps)" { & $bpy -m scripts.train_omnidc --config $Config --variant $v }
  }
}
Invoke-Step "pre-check evaluation (final weights + trend)" { & $bpy -m scripts.p1_label_precheck --prefix p1v2long --snapshots --doc omnidc_p1v2_precheck_long.md }
Write-Host ("P1 V2 LONG PRECHECK DONE ({0:hh\:mm\:ss} elapsed): docs\omnidc_p1v2_precheck_long.md. Tell Claude." -f ((Get-Date) - $startTime))
