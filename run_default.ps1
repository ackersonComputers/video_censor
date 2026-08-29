<#
.SYNOPSIS
    Runs the DEFAULT CASE: mute profanity only. The video is never re-encoded.

.DESCRIPTION
    Invokes `censor run` for one file or `censor batch` for a directory, with
    no scene-editing options. The video stream is copied bit-for-bit; only
    audio is altered. Source files are never modified or overwritten.

.PARAMETER Path
    A video file to process, or a directory of video files to process. Quote
    paths containing spaces or brackets. A directory scan is non-recursive
    unless -Recursive is supplied.

.PARAMETER ReportOnly
    Analyse and report what would be muted, without writing any video.
    Always do this first on an unfamiliar file.

.PARAMETER Output
    Optional output path. Defaults to <name>_censored.mkv beside the input.
    This is only valid when -Path names one video file.

.PARAMETER OutputDirectory
    Optional directory for the censored files when -Path names a directory.
    The input tree is mirrored there. Defaults to the input directory, so each
    output sits beside its source with the _censored suffix.

.PARAMETER Recursive
    When -Path names a directory, include video files in subdirectories.

.PARAMETER Overwrite
    Replace an existing output file.

.PARAMETER ExtraArgs
    Additional censor flags. Scene-editing flags are rejected.

.EXAMPLE
    .\run_default.ps1 -Path "C:\video\show.s01e01.mkv" -ReportOnly
.EXAMPLE
    .\run_default.ps1 -Path "C:\video\show.s01e01.mkv"
.EXAMPLE
    .\run_default.ps1 -Path "C:\video\Season 1"
.EXAMPLE
    .\run_default.ps1 -Path "C:\video\Show" -OutputDirectory "D:\Clean\Show" -Recursive
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Path,

    [switch]$ReportOnly,
    [string]$Output,
    [string]$OutputDirectory,
    [switch]$Recursive,
    [switch]$Overwrite,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = $PSScriptRoot
$py   = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $py)) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}

# -LiteralPath matters: release filenames often contain [ ] which PowerShell
# would otherwise treat as wildcard characters.
if (-not (Test-Path -LiteralPath $Path)) {
    throw "Input path not found: $Path"
}
$inputItem = Get-Item -LiteralPath $Path

# --------------------------------------------------------------------------- #
# Guard the mute-only contract.
# --------------------------------------------------------------------------- #
$forbidden = @('--edl', '-e', '--edl-dir', '--edl-out', '--edl-preview', '--video-encoder', 'manual')
foreach ($arg in $ExtraArgs) {
    if ($forbidden -contains $arg -or $arg -match '^(--edl|--edl-dir|--edl-out|--edl-preview|--video-encoder)=') {
        throw "'$arg' enables video editing or re-encoding and is not allowed in the default case. Use the censor CLI directly if that is genuinely intended."
    }
}

$cliArgs = @('-m', 'censor.cli')
if ($inputItem.PSIsContainer) {
    if ($Output) {
        throw '-Output is for a single video file. Use -OutputDirectory when -Path is a directory.'
    }

    $inputDirectory = $inputItem.FullName
    $destinationDirectory = if ($OutputDirectory) {
        [System.IO.Path]::GetFullPath($OutputDirectory)
    } else {
        $inputDirectory
    }

    # Continue through the folder even if one file is malformed or unsupported;
    # the batch command returns a failure status and lists those files at the end.
    $cliArgs += @('batch', $inputDirectory, '--output-dir', $destinationDirectory, '--continue-on-error')
    if ($Recursive) { $cliArgs += '--recursive' }

    Write-Host 'Mode  : MUTE ONLY - video is stream-copied, never re-encoded' -ForegroundColor Cyan
    Write-Host "Input : $inputDirectory (directory)" -ForegroundColor Cyan
    Write-Host "Output: $destinationDirectory" -ForegroundColor Cyan
} else {
    if ($OutputDirectory) {
        throw '-OutputDirectory is for a directory. Use -Output when -Path is one video file.'
    }

    $inputFile = $inputItem.FullName
    $cliArgs += @('run', $inputFile)
    if ($Output) { $cliArgs += @('-o', $Output) }

    Write-Host 'Mode  : MUTE ONLY - video is stream-copied, never re-encoded' -ForegroundColor Cyan
    Write-Host "Input : $inputFile" -ForegroundColor Cyan
}

if ($ReportOnly) { $cliArgs += '--report-only' }
if ($Overwrite)  { $cliArgs += '--overwrite' }
$cliArgs += $ExtraArgs

if ($ReportOnly) { Write-Host 'Report-only: no video will be written.' -ForegroundColor Yellow }
Write-Host ''

& $py @cliArgs
exit $LASTEXITCODE
