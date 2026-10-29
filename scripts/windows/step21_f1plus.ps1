# F1+ (docs/plan_omnidc.md sec. 24): P1's F1 with a conformally calibrated trust floor, inference only, free.
#   1. engineering check on 3 images per set (cache\p1_f1plus_*_dev_limit, docs\omnidc_p1_trust_floor_limit.md)
#   2. calibration + F1+ caches + evaluation on the gate's populations (about 2-3 h; HAMMER evaluation is the slow part)
#   3. the unchanged P1 gate with F1+ in place of F1 -> docs\omnidc_p1plus_gate.md
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step21_f1plus.ps1
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
. "$PSScriptRoot\common.ps1"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$bpy = "D:\Python_Script\TrustPrompt\.venvs\baselines\Scripts\python.exe"
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
$startTime = Get-Date
$EvalOk = @("small_scale_validation_passed", "completed_with_invalid_predictions")
foreach ($d in @("cache\p1_f1plus_nyu_dev", "cache\p1_f1plus_hammer_dev")) {
  if ((Test-Path $d) -and -not (($EvalOk -contains (Get-Stage "$d\evaluation_val")) -and ($EvalOk -contains (Get-Stage "$d\evaluation_train")))) {
    Write-Host "$d exists but is incomplete. Rename it (append _failed) and re-run."; exit 1
  }
}
if (-not (Test-Path "cache\p1_f1plus_hammer_dev_limit")) {
  Invoke-Step "F1+ engineering check (3 images per set)" { & $bpy -m scripts.p1_trust_floor --limit 3 }
}
if (-not (Test-Path "cache\p1_f1plus_hammer_dev")) {
  if (Test-Path "cache\p1_f1plus_nyu_dev") { Write-Host "cache\p1_f1plus_nyu_dev exists without the HAMMER run: rename it (append _failed) and re-run."; exit 1 }
  Assert-FreeGB "D" 8
  Invoke-Step "F1+ calibration, caches and evaluation" { & $bpy -m scripts.p1_trust_floor }
}
Invoke-Step "P1 gate with F1+ (pre-registered)" { & $py -m scripts.p1_gate --spec configs/omnidc/p1plus_gate_v1.yaml }
Write-Host ("F1+ DONE ({0:hh\:mm\:ss} elapsed): docs\omnidc_p1_trust_floor.md, docs\omnidc_p1plus_gate.md. Tell Claude." -f ((Get-Date) - $startTime))
