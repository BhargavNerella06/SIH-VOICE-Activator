<#
Start the demo Flask server with a persistent local virtual environment.

This script prevents "No module named 'torchaudio'" on fresh starts by:
1) Creating .venv (once)
2) Verifying demo dependencies
3) Installing from requirements.txt when needed
#>
[CmdletBinding()]
param(
  [int]$Port = 5000,
  [switch]$EnableDebug,
  [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Invoke-BootstrapPython {
  param(
    [Parameter(Mandatory = $true)][string[]]$Args
  )

  if (Get-Command python -ErrorAction SilentlyContinue) {
    & python @Args
    return
  }

  if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 @Args
    return
  }

  throw "Python was not found in PATH. Install Python 3.10+ and retry."
}

function Test-DemoDependencies {
  param(
    [Parameter(Mandatory = $true)][string]$PythonExe
  )

  $checkScript = @'
import importlib.util
import sys

mods = [
    'flask',
    'numpy',
    'soundfile',
    'librosa',
    'sklearn',
    'torch',
    'torchaudio',
    'resampy',
    'dotenv',
]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print('Missing modules: ' + ', '.join(missing))
    sys.exit(1)
print('All demo dependencies are available.')
'@

  $checkOutput = & $PythonExe -c $checkScript 2>&1
  $isReady = ($LASTEXITCODE -eq 0)

  if (-not $isReady -and $checkOutput) {
    Write-Host ($checkOutput -join [Environment]::NewLine)
  }

  return $isReady
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $scriptDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$requirements = Join-Path $scriptDir "requirements.txt"

if (-not (Test-Path $venvPython)) {
  Write-Host "Creating demo virtual environment at: $venvDir"
  Invoke-BootstrapPython -Args @("-m", "venv", $venvDir)
}

$installNeeded = -not $SkipInstall
if (-not $SkipInstall) {
  if (Test-DemoDependencies -PythonExe $venvPython) {
    $installNeeded = $false
  }
}

if ($installNeeded) {
  Write-Host "Installing demo dependencies from: $requirements"
  & $venvPython -m pip install --upgrade pip
  & $venvPython -m pip install -r $requirements
}

$env:PORT = "$Port"
$env:FLASK_DEBUG = if ($EnableDebug.IsPresent) { "1" } else { "0" }

Write-Host "Starting Flask demo server at http://localhost:$Port"
Write-Host "Interpreter: $venvPython"

Push-Location $scriptDir
try {
  & $venvPython app.py
}
finally {
  Pop-Location
}
