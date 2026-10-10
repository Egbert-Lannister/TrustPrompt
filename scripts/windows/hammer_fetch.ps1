# HAMMER step B: download only the selected subset (see configs/hammer_subset_v1*.yaml).
#   powershell -ExecutionPolicy Bypass -File scripts\windows\hammer_fetch.ps1 -Dest E:\TrustPrompt_data\hammer_subset_v1
#   add  -Stride2  to take every second frame (7.9 GiB download / 9.6 GiB on disk instead of 15.8 / 19.1 GiB)
# Parallel: -Workers N Range connections (default 8), like aria2's multi-connection mode, but only the
# selected ZIP members are transferred (aria2 cannot fetch sub-ranges of one file).
# Resumable: re-run the same command after any interruption; finished files are skipped.
param([string]$Dest = "", [switch]$Stride2, [int]$Workers = 8)
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = ".\.venv\Scripts\python.exe"
if ($Dest -eq "") { Write-Host "Give -Dest <folder on a drive with enough space>, e.g. -Dest E:\TrustPrompt_data\hammer_subset_v1"; exit 1 }
$cfg = if ($Stride2) { "configs/hammer_subset_v1_stride2.yaml" } else { "configs/hammer_subset_v1.yaml" }
$needGB = if ($Stride2) { 11 } else { 21 }
$drive = (Split-Path -Qualifier $Dest).TrimEnd(":")
$freeGB = [math]::Floor((Get-PSDrive $drive).Free / 1GB)
Write-Host "Selection $cfg needs ~$needGB GB on disk; $drive`: has $freeGB GB free."
if ($freeGB -lt $needGB) { Write-Host "Not enough space. Use another drive or -Stride2."; exit 1 }
for ($attempt = 1; $attempt -le 20; $attempt++) {
  & $py -m scripts.hammer_remote pfetch --index data/source_cache/hammer_index_v1.json --select $cfg --dest $Dest --workers $Workers --max-download-gib 20
  if ($LASTEXITCODE -eq 0) { Write-Host "HAMMER SUBSET COMPLETE in $Dest. Tell Claude the folder."; exit 0 }
  Write-Host "Attempt $attempt failed (network?). Resuming in 60 s..."; Start-Sleep -Seconds 60
}
Write-Host "Still failing after 20 attempts; tell Claude."; exit 1
