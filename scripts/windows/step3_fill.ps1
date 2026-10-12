# Step 3 (revision R13): fill module upper bounds. NYU train (55 scenes) and HAMMER val/train, CPU only.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step3_fill.ps1
# trust source x fill rule; oracle_* read GT-derived masks (upper bounds, not methods); gate_global is deployable.
# Runs in parallel processes (one per configuration); finished runs are skipped on re-run.
param([int]$Parallel = 4)
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$fill = "fill={grow: 3, ring: 10, min_ring: 50, min_area: 20}"
$jobs = @()
foreach ($c in @("oracle_c:k0","oracle_c:k1","oracle_c:k2","oracle_c:k3","oracle_c:v1","oracle_c:oracle_best","oracle_focus:v1","gate_global:v1")) {
  $src, $rule = $c.Split(":"); $label = "trustfill_${src}_${rule}"
  $jobs += @{cfg="configs/mirror3d_dev_eval.yaml"; split="train"; out="results/mirror3d_dev_${label}_train_v1"; src=$src; rule=$rule; label=$label}
  foreach ($s in @("val","train")) {
    if ($s -eq "train" -and @("k0","k1","oracle_best") -contains $rule) { continue }   # HAMMER train: main rules only
    $jobs += @{cfg="configs/hammer_dev_eval.yaml"; split=$s; out="results/hammer_dev_${label}_${s}_v1"; src=$src; rule=$rule; label=$label}
  }
}
$todo = @()
foreach ($j in $jobs) {
  $r = "$($j.out)\receipt.json"
  if (Test-Path $r) {
    $stage = (Get-Content $r -Raw | ConvertFrom-Json).stage
    if (@("small_scale_validation_passed","completed_with_invalid_predictions") -contains $stage) { continue }
  }
  $todo += $j
}
Write-Host "$($todo.Count) runs to do, $Parallel at a time  ($(Get-Date -Format HH:mm:ss))"
New-Item -ItemType Directory -Force results\_logs | Out-Null
$running = @()
foreach ($j in $todo) {
  while (@($running | Where-Object { -not $_.HasExited }).Count -ge $Parallel) { Start-Sleep -Seconds 5 }
  $resume = if (Test-Path "$($j.out)\receipt.json") { "--resume" } else { "" }
  $argline = "-m tp.eval.run_eval --config $($j.cfg) --method trustfill --split $($j.split) --output $($j.out) $resume " +
          "--set trust_source=$($j.src) --set fill_rule=$($j.rule) --set method_label=$($j.label) --set `"$fill`""
  $log = "results\_logs\" + (Split-Path $j.out -Leaf) + ".log"
  Write-Host "start $($j.out)"
  $running += Start-Process -FilePath $py -ArgumentList $argline -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
}
$running | Wait-Process
# Start-Process -PassThru leaves ExitCode null in Windows PowerShell 5 (no cached handle), so success is
# judged from each run's receipt stage, the same test used to skip finished runs above.
$bad = @()
foreach ($j in $jobs) {
  $r = "$($j.out)\receipt.json"
  $stage = if (Test-Path $r) { (Get-Content $r -Raw | ConvertFrom-Json).stage } else { "missing" }
  if (-not (@("small_scale_validation_passed","completed_with_invalid_predictions") -contains $stage)) {
    Write-Host "FAILED ($stage): $($j.out)"; $bad += $j
  }
}
if ($bad.Count -gt 0) { Write-Host "$($bad.Count) runs FAILED; see results\_logs\*.err, then re-run this script."; exit 1 }
& $py -m scripts.summarize_dev; if ($LASTEXITCODE -ne 0) { exit 1 }
& $py -m scripts.summarize_dev --dataset hammer; if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "ALL STEP-3 FILL RUNS SUCCEEDED ($(Get-Date -Format HH:mm:ss)). Tell Claude."
