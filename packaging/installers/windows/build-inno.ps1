param(
    [Parameter(Mandatory = $true)][string]$BundleDir,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [string]$Icon = "",
    [string]$Version = "0.0.0"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$issPath = Join-Path $scriptDir "audio-transcriber.iss"

$resolvedBundle = (Resolve-Path -LiteralPath $BundleDir).Path
$resolvedOut = [System.IO.Path]::GetFullPath($OutDir)
if (-not (Test-Path -LiteralPath $resolvedOut)) {
    New-Item -ItemType Directory -Path $resolvedOut -Force | Out-Null
}

function Find-Iscc {
    if ($env:INNO_SETUP_ISCC -and (Test-Path -LiteralPath $env:INNO_SETUP_ISCC)) {
        return $env:INNO_SETUP_ISCC
    }
    $candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 5\ISCC.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }
    $command = Get-Command iscc.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    throw "Не найден ISCC.exe (Inno Setup). Установите Inno Setup 6 или задайте INNO_SETUP_ISCC."
}

$iscc = Find-Iscc
$arguments = @(
    "/DBundleDir=$resolvedBundle",
    "/DOutputDir=$resolvedOut",
    "/DAppVersion=$Version",
    "/DIconFile=$Icon",
    $issPath
)

Write-Host "+ `"$iscc`" $($arguments -join ' ')"
& $iscc @arguments
if ($LASTEXITCODE -ne 0) {
    throw "ISCC завершился с кодом $LASTEXITCODE"
}

$output = Join-Path $resolvedOut "audio-transcriber-$Version-setup.exe"
if (-not (Test-Path -LiteralPath $output)) {
    throw "ISCC отработал, но установщик не найден: $output"
}
Write-Host "Готово: $output"
