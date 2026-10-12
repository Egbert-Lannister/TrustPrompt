# Revision R14 experiments, train/val only (test never read):
#   (a) DEPTHOR++-style detector as a deployable trust source: fill (k0 / v1 / sector hold-out) and PromptDA-L with
#       the flagged prompt pixels removed; paper thresholds and a "Spearman gate off" sensitivity (hi = 1.01).
#   (b) fill selection rules AIC / GRIC / sector hold-out / radial hold-out with GT contamination masks.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step4_r14.ps1            # everything
#   ... -Phase gpu   (PromptDA caches only)    ... -Phase cpu   (evaluations only)    ... -Parallel 8
# Finished runs are skipped on re-run. Success is judged from each run's receipt stage, never from ExitCode.
param([int]$Parallel = 6, [ValidateSet("all", "gpu", "cpu")][string]$Phase = "all")
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$Good = @("small_scale_validation_passed", "completed_with_invalid_predictions")
$t0 = Get-Date
function Get-Stage([string]$Dir) {
  $r = Join-Path $Dir "receipt.json"
  if (Test-Path $r) { return (Get-Content $r -Raw | ConvertFrom-Json).stage } else { return "missing" }
}
function Invoke-Step([string]$Name, [scriptblock]$Body) {
  Write-Host "=== $Name  ($(Get-Date -Format HH:mm:ss))"
  & $Body
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $Name (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}
$Sets = @{
  nyu    = @{ eval = "configs/mirror3d_dev_eval.yaml"; prefix = "mirror3d_dev"; cache = "cache";                     tag = "nyu" }
  hammer = @{ eval = "configs/hammer_dev_eval.yaml";   prefix = "hammer_dev";   cache = "E:\TrustPrompt_data\cache"; tag = "hammer" }
}
$Det = @{ detector = "detector={seed: 0}"; detector_nogate = "detector={seed: 0, hi: 1.01}" }   # must equal the PromptDA configs
$fill = "fill={grow: 3, ring: 10, min_ring: 50, min_area: 20}"

# ---------------- GPU: PromptDA-L with detector-masked prompts (3-sample preflight, then full) ----------------
if ($Phase -ne "cpu") {
  foreach ($ds in @("nyu", "hammer")) {
    foreach ($v in @("detector", "detector_nogate")) {
      $cacheName = "promptda_vitl_${v}_prior_$($Sets[$ds].tag)_dev_v1"
      $out = Join-Path $Sets[$ds].cache $cacheName
      $cfg = "configs/promptda/$($ds)_vitl_${v}_prior.yaml"
      $steps = @(
        @{ dir = "${out}_preflight"; extra = @("--limit", "3", "--output", "${out}_preflight") },
        @{ dir = $out; extra = @("--output", $out) })
      foreach ($p in $steps) {
        $st = Get-Stage $p.dir
        if ($st -eq "real_dev_verified") { continue }
        if ($st -ne "missing") { Write-Host "$($p.dir) has stage '$st'. Rename that folder (append _failed) and re-run."; exit 1 }
        if ($ds -eq "hammer") {
          $free = (Get-PSDrive E).Free / 1GB
          if ($free -lt 4) { Write-Host ("E: has only {0:N1} GB free; a HAMMER cache needs about 2 GB. Free space and re-run." -f $free); exit 1 }
        }
        $extra = $p.extra
        Invoke-Step "cache $cfg -> $($p.dir)" { & $py -m scripts.cache_promptda --config $cfg @extra }
      }
    }
  }
  Write-Host "GPU phase done ($(Get-Date -Format HH:mm:ss))"
}
if ($Phase -eq "gpu") { exit 0 }

# ---------------- CPU: evaluations in parallel processes ----------------
$jobs = @()
foreach ($ds in @("hammer", "nyu")) {
  foreach ($s in @("train", "val")) {
    $d = $Sets[$ds]
    $rules = @("aic", "gric", "holdout_sector", "holdout_radial"); if ($ds -eq "hammer" -and $s -eq "train") { $rules += "oracle_best" }
    foreach ($rule in $rules) {
      $label = "trustfill_oracle_c_$rule"
      $jobs += @{ ds = $ds; out = "results/$($d.prefix)_${label}_${s}_v1"; args = "--config $($d.eval) --method trustfill --split $s " +
        "--set trust_source=oracle_c --set fill_rule=$rule --set method_label=$label --set `"$fill`"" }
    }
    foreach ($v in @("detector", "detector_nogate")) {
      $tag = if ($v -eq "detector") { "depthor" } else { "depthor_nogate" }
      $rl = if ($v -eq "detector") { @("k0", "v1", "holdout_sector") } else { @("k0", "v1") }
      foreach ($rule in $rl) {
        $label = "trustfill_${tag}_$rule"
        $jobs += @{ ds = $ds; out = "results/$($d.prefix)_${label}_${s}_v1"; args = "--config $($d.eval) --method trustfill --split $s " +
          "--set trust_source=depthor --set fill_rule=$rule --set method_label=$label --set `"$fill`" --set `"$($Det[$v])`"" }
      }
      $cache = Join-Path $d.cache "promptda_vitl_${v}_prior_$($d.tag)_dev_v1"
      if ((Get-Stage $cache) -eq "real_dev_verified") {
        $label = "promptda_vitl_${v}_prior"
        $jobs += @{ ds = $ds; out = "results/$($d.prefix)_${label}_${s}_v1"; args = "--config $($d.eval) --method cached_prediction --split $s " +
          "--set prediction_cache=$cache --set expected_model=PromptDA-L --set method_label=$label" }
      } else { Write-Host "WARNING: $cache not verified yet; its evaluation is skipped (run -Phase gpu first)." }
    }
  }
}
$todo = @(); $bad = @()
foreach ($j in $jobs) {
  $st = Get-Stage $j.out
  if ($Good -contains $st) { continue }
  if ($st -eq "missing") { $j.resume = "" }
  elseif (@("partial_time_budget", "started") -contains $st) { $j.resume = "--resume" }
  else { Write-Host "$($j.out) has stage '$st'. Rename that folder (append _failed) and re-run."; $bad += $j; continue }
  $todo += $j
}
if ($bad.Count -gt 0) { exit 1 }
$cores = [Environment]::ProcessorCount; $threads = [Math]::Max(1, [Math]::Floor($cores / $Parallel))
foreach ($v in @("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")) { Set-Item "env:$v" "$threads" }
Write-Host "$($todo.Count) of $($jobs.Count) evaluation runs to do, $Parallel at a time, $threads threads each ($(Get-Date -Format HH:mm:ss))"
New-Item -ItemType Directory -Force results\_logs | Out-Null
$running = @()
foreach ($j in $todo) {
  while (@($running | Where-Object { -not $_.HasExited }).Count -ge $Parallel) { Start-Sleep -Seconds 5 }
  $log = "results\_logs\" + (Split-Path $j.out -Leaf) + ".log"
  Write-Host "start $($j.out)  ($(Get-Date -Format HH:mm:ss))"
  $running += Start-Process -FilePath $py -ArgumentList "-m tp.eval.run_eval $($j.args) --output $($j.out) $($j.resume)" `
    -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
}
$running | Wait-Process
$failed = @($jobs | Where-Object { -not ($Good -contains (Get-Stage $_.out)) })
foreach ($j in $failed) { Write-Host "FAILED ($(Get-Stage $j.out)): $($j.out)  -> see results\_logs\$(Split-Path $j.out -Leaf).log.err" }
if ($failed.Count -gt 0) { Write-Host "$($failed.Count) runs FAILED; fix, rename failed folders if needed, then re-run this script."; exit 1 }
Invoke-Step "NYU tables" { & $py -m scripts.summarize_dev }
Invoke-Step "HAMMER tables" { & $py -m scripts.summarize_dev --dataset hammer }
Invoke-Step "R14 selection/detector tables" { & $py -m scripts.summarize_selection }
Write-Host ("ALL R14 RUNS SUCCEEDED ({0:hh\:mm\:ss} elapsed). Tell Claude." -f ((Get-Date) - $t0))
