# Restore the PromptDA source and checkpoints on E: (deleted during disk clean-up, found 2026-10-14) and verify
# them against what every earlier PromptDA run recorded: source commit 9161547 with the audited file hashes
# (docs/decisions.md sec. 8) and each checkpoint's SHA256 from the run receipts. Anything that differs stops.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\restore_promptda.ps1 [-HfMirror https://hf-mirror.com]
param([string]$HfMirror = "")
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$Src = "E:\TrustPrompt_data\external\PromptDA"; $W = "E:\TrustPrompt_data\weights\promptda"
$Commit = "9161547ca7151e14fcc8f490ad3cc02fa9437483"
$Ckpts = [ordered]@{
  "prompt-depth-anything-vitl"             = "4b3a3e0a60ea68e719cad3b2e22da8c95ce78f9559ee04824c064a2a06b0f10f"
  "prompt-depth-anything-vits"             = "51b99fef5e3c467f3ccd12f2f4687220e7f1e88f6a07fe21f7a66adffca2af99"
  "prompt-depth-anything-vits-transparent" = "4202b8167c4877f5415a98969f9077ad60447e6424164aa1336fab9e741c15e2"
}
function Invoke-Step([string]$Label, [scriptblock]$Body) {
  Write-Host "=== $Label  ($(Get-Date -Format HH:mm:ss))"
  & $Body
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $Label (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}
if ((Get-PSDrive E).Free / 1GB -lt 3) { Write-Host "E: needs about 3 GB free for the checkpoints."; exit 1 }

# 1. Source at the audited commit
if ((Test-Path $Src) -and -not (Test-Path "$Src\.git")) { Write-Host "$Src exists but is not a git checkout; rename it and re-run."; exit 1 }
if (-not (Test-Path "$Src\.git")) {
  New-Item -ItemType Directory -Force (Split-Path $Src) | Out-Null
  Invoke-Step "clone PromptDA" { git clone https://github.com/DepthAnything/PromptDA.git $Src }
}
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

# 2. Checkpoints + model cards, then the recorded SHA256
if ($HfMirror -ne "") { $env:HF_ENDPOINT = $HfMirror }
foreach ($repo in $Ckpts.Keys) {
  if (-not (Test-Path "$W\$repo\model.ckpt")) {
    Invoke-Step "download $repo" { & $py -c "from huggingface_hub import hf_hub_download as d; [d('depth-anything/$repo', f, local_dir=r'$W\$repo') for f in ('README.md','model.ckpt')]" }
  }
  $got = (Get-FileHash "$W\$repo\model.ckpt" -Algorithm SHA256).Hash.ToLower()
  if ($got -ne $Ckpts[$repo]) { Write-Host "$repo checkpoint hash $got differs from the recorded $($Ckpts[$repo]). Stop."; exit 1 }
  Write-Host "$repo checkpoint hash OK"
}
Write-Host "PromptDA source and checkpoints restored and verified."
