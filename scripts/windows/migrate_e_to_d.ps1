# Copy the project's data from E: into the project on D: (2026-10-14), verify it, then E: is no longer needed.
# Nothing on E: is moved or deleted: E: stays as a backup until you remove it yourself.
#   E:\TrustPrompt_data\cache\*             -> cache\*                              (HAMMER PromptDA caches)
#   E:\TrustPrompt_data\weights\promptda    -> checkpoints\promptda                 (PromptDA checkpoints)
#   E:\TrustPrompt_data\external\PromptDA   -> external\PromptDA                    (official source, git checkout)
#   E:\TrustPrompt_data\hammer_subset_v1    -> data\source_cache\hammer_subset_v1   (HAMMER raw subset)
#   E:\GLARE_Data\first_surface\mirror3d\nyu.csv, nyu.zip -> data\source_cache\mirror3d\  (read-only copy; GLARE untouched)
#   powershell -ExecutionPolicy Bypass -File scripts\windows\migrate_e_to_d.ps1
# Safe to re-run: robocopy only copies what is missing or different; verification runs every time.
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$Proj = "D:\Python_Script\TrustPrompt"
if (-not (Test-Path "E:\TrustPrompt_data")) { Write-Host "E:\TrustPrompt_data not found; connect E: and re-run."; exit 1 }
$dirPairs = @(
  @{ src = "E:\TrustPrompt_data\cache";             dst = "$Proj\cache" },
  @{ src = "E:\TrustPrompt_data\weights\promptda";  dst = "$Proj\checkpoints\promptda" },
  @{ src = "E:\TrustPrompt_data\external\PromptDA"; dst = "$Proj\external\PromptDA" },
  @{ src = "E:\TrustPrompt_data\hammer_subset_v1";  dst = "$Proj\data\source_cache\hammer_subset_v1" }
)
$m3dSrc = "E:\GLARE_Data\first_surface\mirror3d"; $m3dDst = "$Proj\data\source_cache\mirror3d"; $m3dFiles = @("nyu.csv", "nyu.zip")

# space check
$needBytes = 0
foreach ($pr in $dirPairs) {
  if (Test-Path $pr.src) { $needBytes += (Get-ChildItem $pr.src -Recurse -File -Force | Measure-Object Length -Sum).Sum }
  else { Write-Host "note: $($pr.src) not found, skipped" }
}
foreach ($fn in $m3dFiles) { if (Test-Path "$m3dSrc\$fn") { $needBytes += (Get-Item "$m3dSrc\$fn").Length } }
$freeGB = (Get-PSDrive D).Free / 1GB; $needGB = $needBytes / 1GB
Write-Host ("data on E: to copy {0:N1} GB; D: free {1:N1} GB" -f $needGB, $freeGB)
if ($freeGB -lt $needGB + 10) { Write-Host "D: needs at least the data size + 10 GB free. Free space and re-run."; exit 1 }

# copy (robocopy exit codes 0-7 = success, >= 8 = failure)
New-Item -ItemType Directory -Force results\_logs | Out-Null
$logFile = "results\_logs\migrate_e_to_d.log"
foreach ($pr in $dirPairs) {
  if (-not (Test-Path $pr.src)) { continue }
  Write-Host "=== copy $($pr.src) -> $($pr.dst)  ($(Get-Date -Format HH:mm:ss))"
  robocopy $pr.src $pr.dst /E /COPY:DAT /DCOPY:T /R:2 /W:5 /MT:8 /NP /NFL /NDL "/LOG+:$logFile" | Out-Null
  if ($LASTEXITCODE -ge 8) { Write-Host "robocopy FAILED ($LASTEXITCODE) for $($pr.src); see $logFile"; exit 1 }
}
$m3dCopy = @($m3dFiles | Where-Object { Test-Path "$m3dSrc\$_" })
if ($m3dCopy.Count -gt 0) {
  Write-Host "=== copy Mirror3D metadata ($($m3dCopy -join ', '))"
  robocopy $m3dSrc $m3dDst @m3dCopy /COPY:DAT /R:2 /W:5 /NP /NFL /NDL "/LOG+:$logFile" | Out-Null
  if ($LASTEXITCODE -ge 8) { Write-Host "robocopy FAILED ($LASTEXITCODE) for Mirror3D metadata; see $logFile"; exit 1 }
}

# verify: sizes of every file, every cached prediction against its recorded SHA256, checkpoint hashes
$verifyArgs = @("-m", "scripts.verify_migration")
foreach ($pr in $dirPairs) { if (Test-Path $pr.src) { $verifyArgs += @("--pair", "$($pr.src)=$($pr.dst)") } }
foreach ($fn in $m3dCopy) { $verifyArgs += @("--pair", "$m3dSrc\$fn=$m3dDst\$fn") }
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$verifyArgs += @("--output", "results/migration_e_to_d_$stamp")
Write-Host "=== verify (hashes every cached prediction; takes a while)  ($(Get-Date -Format HH:mm:ss))"
& $py @verifyArgs
if ($LASTEXITCODE -ne 0) { Write-Host "VERIFY FAILED: see results\migration_e_to_d_$stamp\report.json. E: is untouched."; exit 1 }

# the restored source must be at the audited commit and the checkpoints must match the recorded hashes
& powershell -ExecutionPolicy Bypass -File scripts\windows\restore_promptda.ps1
if ($LASTEXITCODE -ne 0) { Write-Host "PromptDA check FAILED"; exit 1 }
Write-Host "MIGRATION DONE AND VERIFIED. The project no longer reads E:. E: was not modified. Tell Claude."
