# One-time setup for the depth-completion baselines (docs/baselines.md): a separate venv that sees the Anaconda
# packages (torch etc.) plus a few pinned extras, the official sources at pinned commits, and the pinned weights
# with size + SHA256 checks. Downloads about 30 GB (36 GB with -WithMarigold). No licence is accepted on your
# behalf: every repository used is public and ungated; licences are recorded in checkpoints\baselines\setup_receipt.json.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\setup_baselines.ps1 [-WithMarigold] [-HfMirror https://hf-mirror.com]
param([switch]$WithMarigold, [string]$HfMirror = "")
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
function Invoke-Step([string]$Label, [scriptblock]$Body) {
  Write-Host "=== $Label  ($(Get-Date -Format HH:mm:ss))"
  & $Body
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $Label (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}
$needGB = 32; if ($WithMarigold) { $needGB = 38 }
$freeGB = (Get-PSDrive D).Free / 1GB
if ($freeGB -lt $needGB) { Write-Host ("D: has {0:N1} GB free; the baselines need about {1} GB. Free space and re-run." -f $freeGB, $needGB); exit 1 }

# 1. venv on the same Anaconda interpreter as .venv, with its site packages (torch, numpy, scipy, cv2, yaml)
$cfg = Get-Content ".\.venv\pyvenv.cfg" | Where-Object { $_ -match '^executable\s*=' }
$basePy = ($cfg -split '=', 2)[1].Trim()
if (-not (Test-Path $basePy)) { Write-Host "Base interpreter $basePy not found (from .venv\pyvenv.cfg)."; exit 1 }
$bpy = "D:\Python_Script\TrustPrompt\.venvs\baselines\Scripts\python.exe"
if (-not (Test-Path $bpy)) {
  Invoke-Step "create .venvs\baselines" { & $basePy -m venv --system-site-packages D:\Python_Script\TrustPrompt\.venvs\baselines }
}
$pins = (Get-Content configs\baselines\registry.json -Raw | ConvertFrom-Json).pip
Invoke-Step "pip install pinned extras" { & $bpy -m pip install --disable-pip-version-check @pins }
Invoke-Step "torch / CUDA check" { & $bpy -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else ''); sys.exit(0 if torch.cuda.is_available() else 1)" }

# 2. sources + weights (Python does the work; it stops on any hash, size or commit mismatch)
$setupArgs = @("-m", "scripts.setup_baselines")
if ($WithMarigold) { $setupArgs += "--with-optional" }
if ($HfMirror -ne "") { $setupArgs += @("--hf-endpoint", $HfMirror) }
Invoke-Step "sources and weights" { & $bpy @setupArgs }
Write-Host "Baseline setup verified. Next: powershell -ExecutionPolicy Bypass -File scripts\windows\step10_baselines.ps1"
Write-Host "If a SHA256 differs while the size matches, the pinned value in configs\baselines\registry.json may be mistyped: tell Claude, do not edit it yourself."
