# Mainline P1 evaluation on this machine (docs/plan_omnidc.md sec. 7, 19), after the AutoDL F0 / F1 runs finished.
# Needs both run folders copied back into results\ WITHOUT last.pt (receipt.json, config.json, training_frames.json,
# log.jsonl, ..., checkpoints\final.pt). The expected final.pt SHA-256 is read from the training job's record
# (results\autodl_full_20261024\mirror\job\operator.json), never from the copied file itself.
#   GPU  scripts.cache_omnidc_finetuned (final.pt, float32, --evaluate): NYU val + train-holdout, HAMMER all 10 dev scenes
#   CPU  the pre-registered P1 gate (configs\omnidc\p1_gate_v1.yaml) -> docs\omnidc_p1_gate.md
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step16_p1_eval.ps1
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
. "$PSScriptRoot\common.ps1"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$bpy = "D:\Python_Script\TrustPrompt\.venvs\baselines\Scripts\python.exe"
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
$startTime = Get-Date
$Job = Get-Content results\autodl_full_20261024\mirror\job\operator.json -Raw | ConvertFrom-Json
$EvalOk = @("small_scale_validation_passed", "completed_with_invalid_predictions")
foreach ($v in @("f0", "f1")) {
  $run = $Job.run_dirs.$v
  $sha = $Job.checkpoints.$v.final.sha256
  if (-not $sha) { Write-Host "No final.pt SHA-256 for $v in the job record: training not finished."; exit 1 }
  $ck = Join-Path $run "checkpoints\final.pt"
  if (-not (Test-Path $ck)) { Write-Host "$ck missing: copy the $v run folder back from the server first."; exit 1 }
  $local = (Get-FileHash $ck -Algorithm SHA256).Hash.ToLower()
  if ($local -ne $sha) { Write-Host "$ck SHA-256 $local differs from the job record ${sha}; copy it again."; exit 1 }
  Write-Host "$v : $run, final.pt SHA-256 verified"
  foreach ($ds in @("nyu", "hammer")) {
    $out = "cache\p1_${v}_final_${ds}_dev"
    $done = (Test-Path "$out\receipt.json") -and ((Get-Stage $out) -eq "real_dev_verified") -and
            ($EvalOk -contains (Get-Stage "$out\evaluation_val")) -and ($EvalOk -contains (Get-Stage "$out\evaluation_train"))
    if ($done) { Write-Host "$out done"; continue }
    if (Test-Path $out) { Write-Host "$out exists but is incomplete (stage $(Get-Stage $out)). Rename it (append _failed) and re-run."; exit 1 }
    Assert-FreeGB "D" 6
    $cmdArgs = @("-m", "scripts.cache_omnidc_finetuned", "--run-dir", $run, "--variant", $v, "--checkpoint", "final.pt",
                 "--checkpoint-sha256", $sha, "--dataset", $ds, "--split", "val", "--split", "train", "--output", $out, "--evaluate")
    Invoke-Step "OMNI-DC + P1 $v on $ds -> $out" { & $bpy @cmdArgs }
  }
}
Invoke-Step "P1 gate (pre-registered)" { & $py -m scripts.p1_gate }
Write-Host ("P1 EVALUATION DONE ({0:hh\:mm\:ss} elapsed): docs\omnidc_p1_gate.md. Tell Claude." -f ((Get-Date) - $startTime))
