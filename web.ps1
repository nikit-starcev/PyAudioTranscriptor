# Запуск веб-интерфейса AudioTranscriptor для Windows.
#
# Поднимает локальный сервер (`audio-transcriber web`) и открывает браузер.
# Запускается двойным щелчком по web.bat либо из PowerShell:
#
#   .\web.ps1                # 8790 либо ближайший свободный порт + браузер
#   .\web.ps1 -Port 9000     # строго заданный порт (занят — понятная ошибка)
#   .\web.ps1 -NoBrowser     # не открывать браузер автоматически
#   .\web.ps1 -WebHost 0.0.0.0   # слушать на всех интерфейсах (осторожно!)
#
# Фактический адрес и порт печатает сама команда (`audio-transcriber web`),
# поэтому оболочка передаёт `--port` только когда он задан явно и не дублирует
# URL: при автоподборе порт может отличаться от 8790.
#
# Остановить сервер — Ctrl+C.

param(
    [string]$WebHost = "127.0.0.1",
    [int]$Port = 8790,
    [switch]$NoBrowser,
    [switch]$Reload
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location -Path $PSScriptRoot

$cliArgs = @("web", "--host", $WebHost)
# `--port` пробрасываем только при явном указании: иначе CLI сам подберёт
# свободный порт, а не получит «зашитый» 8790 как обязательный.
if ($PSBoundParameters.ContainsKey("Port")) { $cliArgs += @("--port", "$Port") }
if ($NoBrowser) { $cliArgs += "--no-browser" }
if ($Reload) { $cliArgs += "--reload" }

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython) -and -not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv не найден — устанавливаю..."
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Ctrl+C — остановить."

if (Test-Path $venvPython) {
    & $venvPython -c "import fastapi, uvicorn" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Веб-интерфейс недоступен: не установлены веб-зависимости."
        Write-Host 'Установите их: uv pip install --python .venv\Scripts\python.exe ".[web]"'
        Read-Host "Нажмите Enter, чтобы закрыть..."
        exit 1
    }
    $exe = Join-Path $PSScriptRoot ".venv\Scripts\audio-transcriber.exe"
    if (Test-Path $exe) {
        & $exe @cliArgs
    } else {
        & $venvPython -m audio_transcriber @cliArgs
    }
} else {
    # .venv ещё нет — `uv run` создаст окружение (extra `web` нужно доустановить).
    uv run audio-transcriber @cliArgs
}

exit $LASTEXITCODE
