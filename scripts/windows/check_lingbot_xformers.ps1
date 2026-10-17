# Check that the cached LingBot results (made without xFormers) match the official xFormers path.
# Installs xformers 0.0.35 into .venvs\baselines with --no-deps (torch is left untouched and checked), then compares
# 30 NYU + 30 HAMMER images; the decision rule is in scripts\check_lingbot_xformers.py.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\check_lingbot_xformers.ps1 [-N 30]
param([int]$N = 30)
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$bpy = "D:\Python_Script\TrustPrompt\.venvs\baselines\Scripts\python.exe"
if (-not (Test-Path $bpy)) { Write-Host "Run scripts\windows\setup_baselines.ps1 first."; exit 1 }
function Invoke-Step([string]$Label, [scriptblock]$Body) {
  Write-Host "=== $Label  ($(Get-Date -Format HH:mm:ss))"
  & $Body
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $Label (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}
$torchBefore = (& $bpy -c "import torch; print(torch.__version__)").Trim()
Invoke-Step "pip install xformers==0.0.35 --no-deps" { & $bpy -m pip install --disable-pip-version-check --no-deps xformers==0.0.35 }
$torchAfter = (& $bpy -c "import torch; print(torch.__version__)").Trim()
Write-Host "torch before $torchBefore, after $torchAfter"
if ($torchBefore -ne $torchAfter) { Write-Host "torch changed; stop and tell Claude."; exit 1 }
Invoke-Step "xformers info" { & $bpy -m xformers.info }
Invoke-Step "LingBot xFormers vs cached fallback ($N + $N images)" { & $bpy -m scripts.check_lingbot_xformers --n $N }
Write-Host "LINGBOT XFORMERS CHECK FINISHED. Tell Claude the verdict line above."
