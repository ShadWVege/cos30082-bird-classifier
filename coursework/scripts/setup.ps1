param(
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [ValidateSet('cpu','cu128')][string]$Compute = 'cu128'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& $PythonExe -c "import sys; assert (3, 10) <= sys.version_info[:2] < (3, 15), 'Use Python 3.10-3.14; 3.12 recommended'"
if ($LASTEXITCODE -ne 0) { throw 'Unsupported Python interpreter' }
& $PythonExe -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Could not create the project environment' }
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
& $projectPython -m pip install --force-reinstall torch torchvision --index-url "https://download.pytorch.org/whl/$Compute"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $projectPython -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $projectPython -m src.doctor
