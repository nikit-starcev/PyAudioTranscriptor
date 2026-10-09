# Запуск портативного бандла AudioTranscriptor (веб-интерфейс) для Windows.
#
# Рассчитан на one-dir-бандл из `python scripts/build_portable.py`
# (`audio-transcriber\` с `audio-transcriber.exe` внутри). Исполняемый файл
# ищется РЯДОМ С САМИМ ЛАУНЧЕРОМ, поэтому лаунчер можно положить как внутрь
# каталога бандла, так и рядом с ним.
#
# Запуск двойным щелчком по run-web.bat либо из PowerShell:
#   .\run-web.ps1
#   .\run-web.ps1 -Port 9000
#   .\run-web.ps1 -NoBrowser
#
# Остановить сервер — Ctrl+C.

param(
    [string]$WebHost = "127.0.0.1",
    [int]$Port = 8790,
    [switch]$NoBrowser
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$exeName = "audio-transcriber.exe"

# Порядок поиска: лаунчер внутри бандла → рядом с бандлом → на уровень выше.
$candidates = @(
    (Join-Path $ScriptDir $exeName),
    (Join-Path (Join-Path $ScriptDir "audio-transcriber") $exeName),
    (Join-Path (Join-Path (Split-Path $ScriptDir -Parent) "audio-transcriber") $exeName)
)

$exe = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1

if (-not $exe) {
    Write-Host "Ошибка: рядом с лаунчером не найден исполняемый файл бандла." -ForegroundColor Red
    Write-Host "Положите этот скрипт внутрь каталога бандла или рядом с ним."
    Write-Host "Ожидался один из путей:"
    $candidates | ForEach-Object { Write-Host "  $_" }
    Write-Host "Соберите бандл: python scripts\build_portable.py --target windows --out dist"
    Read-Host "Нажмите Enter, чтобы закрыть..."
    exit 1
}

$cliArgs = @("web", "--host", $WebHost, "--port", "$Port")
if ($NoBrowser) { $cliArgs += "--no-browser" }

Write-Host "Запуск AudioTranscriptor (веб-интерфейс). Ctrl+C — остановить."
& $exe @cliArgs
exit $LASTEXITCODE
