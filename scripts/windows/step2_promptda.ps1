# Step 2 (plan revision v2 R6, C1 first): PromptDA-S / -L / -S-Transparent on the NYU and HAMMER dev sets,
# plus two ORACLE analyses (GT focus mask, GT contamination mask; prompt refilled from the aligned prior).
#   powershell -ExecutionPolicy Bypass -File scripts\windows\step2_promptda.ps1
#   add  -HfMirror https://hf-mirror.com  if huggingface.co is slow or blocked
# Downloads: official code pinned to the audited commit (D:\Python_Script\TrustPrompt\external\PromptDA) and three
# checkpoints with their model cards (D:\Python_Script\TrustPrompt\checkpoints\promptda, ~1.6 GB). HAMMER prediction caches
# went to E:\TrustPrompt_data\cache until 2026-10-14; all caches now live in TrustPrompt\cache.
# Every step stops on failure; finished steps are skipped on re-run. Test splits are never read.
param([string]$HfMirror = "")
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = ".\.venv\Scripts\python.exe"
$Src = "D:\Python_Script\TrustPrompt\external\PromptDA"; $W = "D:\Python_Script\TrustPrompt\checkpoints\promptda"
$Commit = "9161547ca7151e14fcc8f490ad3cc02fa9437483"
function Invoke-Step([string]$Name, [scriptblock]$Body) {
  Write-Host "=== $Name  ($(Get-Date -Format HH:mm:ss))"
  & $Body
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $Name (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}
function Test-Stage([string]$Receipt, [string[]]$Good) {
  if (-not (Test-Path $Receipt)) { return $false }
  $stage = (Get-Content $Receipt -Raw | ConvertFrom-Json).stage
  if ($Good -contains $stage) { return $true }
  Write-Host "$Receipt has stage '$stage'. Rename that folder (append _failed) and re-run."; exit 1
}

# 0. HAMMER instance contact sheets + CSV template for manual material labelling (train/val scenes only).
if (-not (Test-Path results\hammer_instance_sheets_v1\receipt.json)) {
  Invoke-Step "HAMMER instance sheets" { & $py -m scripts.hammer_instance_sheets --root D:/Python_Script/TrustPrompt/data/source_cache/hammer_subset_v1 --output results/hammer_instance_sheets_v1 }
}

# 1. Official source at the audited commit; file hashes must equal docs/decisions.md section 8.
if (-not (Test-Path "$Src\.git")) {
  New-Item -ItemType Directory -Force (Split-Path $Src) | Out-Null
  Invoke-Step "clone PromptDA" { git clone https://github.com/DepthAnything/PromptDA.git $Src }
}
# E: does not record file ownership, so git refuses the clone as "dubious". Trust only this directory, per
# command (-c safe.directory), without touching the user's global git configuration.
$SafeSrc = $Src -replace '\\', '/'
Invoke-Step "checkout audited commit" { git -c "safe.directory=$SafeSrc" -C $Src checkout --quiet $Commit }
Invoke-Step "confirm commit" { $h = git -c "safe.directory=$SafeSrc" -C $Src rev-parse HEAD; Write-Host "HEAD $h"; if ($h -ne $Commit) { exit 1 } }
Invoke-Step "verify audited source hashes" { & $py -c @"
import hashlib,sys
want={'promptda/promptda.py':'bc0ca4bb4ec61106f01f5da514ea25c45cd652c0b526409c3eef161aa14ea7b6',
      'promptda/model/blocks.py':'ca0b2b2463d4c8186ff6a7e4595ac293bb63e8316f31e7b5dba9b14c4c0f832b',
      'promptda/model/dpt.py':'446a7a43cb1d2a64a15e9244b6f6dfc1c680364daed0e17ba8763b43279ad962'}
bad=[f for f,h in want.items() if hashlib.sha256(open(r'$Src/'+f,'rb').read().replace(b'\r\n',b'\n')).hexdigest()!=h]
print('source hashes OK' if not bad else 'MISMATCH: '+str(bad)); sys.exit(1 if bad else 0)
"@ }

# 2. Checkpoints + model cards (license recorded with each run).
if ($HfMirror -ne "") { $env:HF_ENDPOINT = $HfMirror }
foreach ($repo in @("prompt-depth-anything-vits", "prompt-depth-anything-vitl", "prompt-depth-anything-vits-transparent")) {
  if (-not (Test-Path "$W\$repo\model.ckpt")) {
    Invoke-Step "download $repo" { & $py -c "from huggingface_hub import hf_hub_download as d; [d('depth-anything/$repo', f, local_dir=r'$W\$repo') for f in ('README.md','model.ckpt')]" }
  }
}
Invoke-Step "opencv available" { & $py -c "import cv2; print('cv2', cv2.__version__)" }

# 3. Prediction caches: 3-sample preflight, then full run.
$runs = @(
  @{cfg="nyu_vits_raw"; label="promptda_vits_raw"; model="PromptDA-S"; ds="nyu"; splits=@("val","train")},
  @{cfg="nyu_vitl_raw"; label="promptda_vitl_raw"; model="PromptDA-L"; ds="nyu"; splits=@("val","train")},
  @{cfg="nyu_vits_transparent_raw"; label="promptda_vits_transparent_raw"; model="PromptDA-S-Transparent"; ds="nyu"; splits=@("val","train")},
  @{cfg="nyu_vitl_focus_prior"; label="promptda_vitl_focus_prior"; model="PromptDA-L"; ds="nyu"; splits=@("val","train")},
  @{cfg="nyu_vitl_oracle_prior"; label="promptda_vitl_oracle_prior"; model="PromptDA-L"; ds="nyu"; splits=@("val","train")},
  @{cfg="hammer_vits_raw"; label="promptda_vits_raw"; model="PromptDA-S"; ds="hammer"; splits=@("val")},
  @{cfg="hammer_vitl_raw"; label="promptda_vitl_raw"; model="PromptDA-L"; ds="hammer"; splits=@("val","train")},
  @{cfg="hammer_vits_transparent_raw"; label="promptda_vits_transparent_raw"; model="PromptDA-S-Transparent"; ds="hammer"; splits=@("val")},
  @{cfg="hammer_vitl_focus_prior"; label="promptda_vitl_focus_prior"; model="PromptDA-L"; ds="hammer"; splits=@("val")},
  @{cfg="hammer_vitl_oracle_prior"; label="promptda_vitl_oracle_prior"; model="PromptDA-L"; ds="hammer"; splits=@("val")}
)
foreach ($r in $runs) {
  $cfgPath = "configs/promptda/$($r.cfg).yaml"
  $out = if ($r.ds -eq "nyu") { "cache\$($r.label)_nyu_dev_v1" } else { "cache\$($r.label)_hammer_dev_v1" }
  if (-not (Test-Stage "${out}_preflight\receipt.json" @("real_dev_verified"))) {
    Invoke-Step "preflight $($r.cfg)" { & $py -m scripts.cache_promptda --config $cfgPath --limit 3 --output "${out}_preflight" }
  }
  if (-not (Test-Stage "$out\receipt.json" @("real_dev_verified"))) {
    Invoke-Step "cache $($r.cfg)" { & $py -m scripts.cache_promptda --config $cfgPath --output $out }
  }
  $evalCfg = if ($r.ds -eq "nyu") { "configs/mirror3d_dev_eval.yaml" } else { "configs/hammer_dev_eval.yaml" }
  $prefix = if ($r.ds -eq "nyu") { "mirror3d_dev" } else { "hammer_dev" }
  foreach ($s in $r.splits) {
    $res = "results/${prefix}_$($r.label)_${s}_v1"
    if (Test-Stage "$res\receipt.json" @("small_scale_validation_passed", "completed_with_invalid_predictions")) { continue }
    $resume = @(); if (Test-Path "$res\receipt.json") { $resume = @("--resume") }
    Invoke-Step "eval $($r.cfg) $s" { & $py -m tp.eval.run_eval --config $evalCfg --method cached_prediction --split $s --output $res `
        --set "prediction_cache=$out" --set "expected_model=$($r.model)" --set "method_label=$($r.label)" @resume }
  }
}
Invoke-Step "NYU tables" { & $py -m scripts.summarize_dev }
Invoke-Step "HAMMER tables" { & $py -m scripts.summarize_dev --dataset hammer }
Write-Host "ALL STEP-2 PROMPTDA RUNS SUCCEEDED ($(Get-Date -Format HH:mm:ss)). Tell Claude."
