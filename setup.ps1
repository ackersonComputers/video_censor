<#
.SYNOPSIS
    One-time environment setup for the censor tool.

.DESCRIPTION
    Creates a virtual environment in .\.venv, installs dependencies, and
    verifies the installation. Safe to re-run.

.PARAMETER Cuda
    Install the CUDA 12.8 build of PyTorch for NVIDIA GPU acceleration.
    Omit this on machines without an NVIDIA GPU.

.EXAMPLE
    .\setup.ps1
.EXAMPLE
    .\setup.ps1 -Cuda
#>
[CmdletBinding()]
param(
    [switch]$Cuda
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = $PSScriptRoot
$venv = Join-Path $root '.venv'
$py   = Join-Path $venv 'Scripts\python.exe'

function Write-Step { param([string]$Message) Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Message) Write-Host "    OK: $Message" -ForegroundColor Green }

# --------------------------------------------------------------------------- #
# Guard: refuse to run if another pip is already working.
# --------------------------------------------------------------------------- #
Write-Step 'Checking for existing Python installer processes'
$busy = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -and $_.CommandLine -match 'pip install' })
if ($busy.Count -gt 0) {
    Write-Host 'A pip install is already running:' -ForegroundColor Yellow
    $busy | Select-Object ProcessId, CommandLine | Format-List
    throw 'Refusing to start a second concurrent install. Wait for it to finish, or stop it, then re-run.'
}
Write-Ok 'no conflicting installer running'

# --------------------------------------------------------------------------- #
# Prerequisites
# --------------------------------------------------------------------------- #
Write-Step 'Checking prerequisites'

$systemPython = Get-Command python -ErrorAction SilentlyContinue
if (-not $systemPython) { throw 'Python was not found on PATH. Install Python 3.10 or newer.' }

$versionText = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
$version = [version]$versionText
if ($version -lt [version]'3.10') { throw "Python $versionText found, but 3.10 or newer is required." }
Write-Ok "Python $versionText"

foreach ($tool in 'ffmpeg', 'ffprobe') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool was not found on PATH. Install ffmpeg (winget install Gyan.FFmpeg), then open a NEW terminal."
    }
    Write-Ok "$tool found"
}

# WhisperX 3.8.6 installs TorchCodec 0.7. On Windows that TorchCodec release
# supports FFmpeg 4-7 and needs a shared build (the FFmpeg DLLs as well as the
# executables). A static/newer build still works for censor's direct ffmpeg
# calls, but TorchCodec will emit a warning and disable its decoder.
$ffmpegSharedDirs = @(
    foreach ($entry in ($env:PATH -split [IO.Path]::PathSeparator)) {
        $entry = $entry.Trim().Trim('"')
        if (-not $entry -or -not (Test-Path -LiteralPath $entry -PathType Container)) { continue }
        $sharedDll = @(Get-ChildItem -LiteralPath $entry -Filter 'avcodec-*.dll' -File -ErrorAction SilentlyContinue)
        if ($sharedDll.Count -gt 0) { $entry }
    }
)
if ($ffmpegSharedDirs.Count -eq 0) {
    Write-Host '    Warning: WhisperX/TorchCodec on Windows needs an FFmpeg 7.x full-shared build on PATH.' -ForegroundColor Yellow
    Write-Host '    Install: winget install --id Gyan.FFmpeg.Shared --version 7.1.1 --exact' -ForegroundColor Yellow
    Write-Host '    Then open a new terminal so its shared-library folder is added to PATH.' -ForegroundColor Yellow
}

# --------------------------------------------------------------------------- #
# Virtual environment
# --------------------------------------------------------------------------- #
if (Test-Path -LiteralPath $py) {
    Write-Step 'Reusing existing virtual environment'
} else {
    Write-Step 'Creating virtual environment (.venv)'
    & python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the virtual environment.' }
}
if (-not (Test-Path -LiteralPath $py)) { throw "Virtual environment is broken: $py not found." }
Write-Ok $py

Write-Step 'Upgrading pip'
& $py -m pip install --upgrade pip --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'Failed to upgrade pip.' }

# --------------------------------------------------------------------------- #
# PyTorch (versions must match WhisperX 3.8.6 and TorchCodec 0.7)
# --------------------------------------------------------------------------- #
Write-Step 'Installing PyTorch (large download, please be patient)'
if ($Cuda) {
    & $py -m pip install 'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0' --index-url https://download.pytorch.org/whl/cu128
} else {
    & $py -m pip install 'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0'
}
if ($LASTEXITCODE -ne 0) { throw 'Failed to install PyTorch.' }
Write-Ok 'PyTorch installed'

# --------------------------------------------------------------------------- #
# Project dependencies
# --------------------------------------------------------------------------- #
Write-Step 'Installing project dependencies (this takes several minutes)'
& $py -m pip install -r (Join-Path $root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Failed to install requirements.txt.' }

Write-Step 'Installing the censor package (editable, with asr and dev extras)'
$pkg = Join-Path $root 'solution'
& $py -m pip install -e "$pkg[asr,dev]"
if ($LASTEXITCODE -ne 0) { throw 'Failed to install the censor package.' }
Write-Ok 'censor installed'

# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
Write-Step 'Verifying'

& $py -m censor.cli --version
if ($LASTEXITCODE -ne 0) { throw 'The censor CLI failed to start.' }

$cudaAvailable = (& $py -c "import torch; print(torch.cuda.is_available())").Trim()
Write-Ok "CUDA available to PyTorch: $cudaAvailable"
if ($Cuda -and $cudaAvailable -ne 'True') {
    throw 'CUDA setup was requested, but PyTorch cannot use CUDA. Do not process video until this is resolved.'
}
if (-not $Cuda -and $cudaAvailable -ne 'True') {
    Write-Host '    Note: running on CPU. Speech recognition will be slow.' -ForegroundColor Yellow
    Write-Host '    For an NVIDIA GPU, use: .\setup.ps1 -Cuda' -ForegroundColor Yellow
}
if ($cudaAvailable -eq 'True') {
    $gpuName = (& $py -c "import torch; print(torch.cuda.get_device_name(0))").Trim()
    Write-Ok "GPU: $gpuName"
}

Write-Step 'Running unit tests'
& $py -m pytest (Join-Path $pkg 'tests') -q
if ($LASTEXITCODE -ne 0) { throw 'Unit tests failed. Do not process video until this is resolved.' }

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host 'Next step - inspect a file without writing any video:' -ForegroundColor Green
Write-Host '    .\run_default.ps1 -Path "C:\path\to\video.mkv" -ReportOnly' -ForegroundColor Green
