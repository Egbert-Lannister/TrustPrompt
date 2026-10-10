# HAMMER step A: list the remote ZIP's contents and sizes WITHOUT downloading the archive.
# Reads only the ZIP central directory via HTTP Range (a few MB; capped at 256 MiB).
# Output: data\source_cache\hammer_index_v1.json  -> tell Claude, who designs the subset selection.
$ErrorActionPreference = "Stop"
Set-Location D:\Python_Script\TrustPrompt
$py = ".\.venv\Scripts\python.exe"
if (Test-Path data\source_cache\hammer_index_v1.json) { Write-Host "Index already exists."; exit 0 }
& $py -m scripts.hammer_remote index --output data/source_cache/hammer_index_v1.json
if ($LASTEXITCODE -ne 0) {
  Write-Host "https failed; trying the http:// link given on the dataset page"
  & $py -m scripts.hammer_remote index --output data/source_cache/hammer_index_v1.json --url http://www.campar.in.tum.de/public_datasets/2022_arxiv_jung/_dataset_processed.zip
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: the server may not support HTTP Range; tell Claude."; exit 1 }
}
Write-Host "INDEX DONE."
