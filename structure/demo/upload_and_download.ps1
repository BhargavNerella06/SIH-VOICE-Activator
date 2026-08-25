<#
Simple PowerShell helper to upload an audio file to the demo server and download outputs.
Usage:
  .\upload_and_download.ps1 -FilePath .\mixture.wav -Method ica -NComponents 2
  .\upload_and_download.ps1 -FilePath .\mixture.wav -Method conv_tasnet -ModelPath C:\abs\path\model.pth
#>
param(
  [Parameter(Mandatory=$true)][string]$FilePath,
  [string]$Method = "ica",
  [int]$NComponents = 2,
  [string]$ModelPath = "",
  [string]$Server = "http://localhost:5000"
)

if (-not (Test-Path $FilePath)) {
  Write-Error "File not found: $FilePath"; exit 1
}

# Build curl command (uses system curl)
$cmd = "curl -s -F `"file=@$FilePath`" -F `"method=$Method`""
if ($Method -eq 'ica') { $cmd += " -F `"n_components=$NComponents`"" }
if ($Method -eq 'conv_tasnet' -and $ModelPath) { $cmd += " -F `"model_path=$ModelPath`"" }
$cmd += " `"$Server/separate`""

Write-Output "Uploading with: $cmd"
$json = Invoke-Expression $cmd

if (-not $json) {
  Write-Error "No response from server"; exit 1
}

$obj = $json | ConvertFrom-Json
if ($obj.error) {
  Write-Error "Server error: $($obj.error)"; exit 1
}

if (-not $obj.outputs) {
  Write-Output "No outputs returned"; exit 0
}

# Download each output via /download?path=<urlencoded>
foreach ($o in $obj.outputs) {
  $remote = $o.path
  $name = $o.name
  $escaped = [uri]::EscapeDataString($remote)
  $outLocal = Join-Path -Path (Get-Location) -ChildPath $name
  $dlCmd = "curl -s -o `"$outLocal`" `"$Server/download?path=$escaped`""
  Write-Output "Downloading $name -> $outLocal"
  Invoke-Expression $dlCmd
}

Write-Output "Done. Files saved to $(Get-Location)"
