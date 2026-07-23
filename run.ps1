# Универсальный запуск AudioTranscriptor для Windows.
#
# Что делает:
#   1. Устанавливает uv, если он ещё не установлен.
#   2. Подтягивает Python нужной версии и все зависимости (это делает uv run
#      автоматически при первом запуске — отдельно ничего ставить не нужно).
#   3. Запускает транскрибацию с параметрами из config.env.
#
# Использование (PowerShell):
#   .\run.ps1 путь\к\записи.mp3
#
# Также можно перетащить аудиофайл мышью на run.bat — это то же самое,
# но без необходимости открывать PowerShell.
#
# Перед первым запуском:
#   Copy-Item config.example.env config.env
#   # затем откройте config.env и укажите свой токен Hugging Face и параметры

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$AudioFile
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location -Path $PSScriptRoot

$configPath = Join-Path $PSScriptRoot "config.env"
if (-not (Test-Path $configPath)) {
    Write-Host "Не найден config.env."
    Write-Host "Скопируйте config.example.env в config.env и укажите там параметры (см. README.md)."
    exit 1
}

$config = @{}
Get-Content $configPath -Encoding UTF8 | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
        $key, $value = $line.Split("=", 2)
        $config[$key.Trim()] = $value.Trim()
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv не найден — устанавливаю..."
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

if (-not $config["HF_TOKEN"]) {
    Write-Host "Внимание: HF_TOKEN не задан в config.env - определение говорящих завершится ошибкой (см. README)."
}

$cliArgs = @("transcribe", $AudioFile)
if ($config["OUTPUT_DIR"]) { $cliArgs += @("--output-dir", $config["OUTPUT_DIR"]) }
if ($config["MODEL"]) { $cliArgs += @("--model", $config["MODEL"]) }
if ($config["LANGUAGE"]) { $cliArgs += @("--language", $config["LANGUAGE"]) }
if ($config["DEVICE"]) { $cliArgs += @("--device", $config["DEVICE"]) }

if ($config["FORMATS"]) {
    foreach ($fmt in $config["FORMATS"].Split(",")) {
        $cliArgs += @("--format", $fmt.Trim())
    }
}

if ($config["NUM_SPEAKERS"]) { $cliArgs += @("--num-speakers", $config["NUM_SPEAKERS"]) }

if ($config["SPEAKER_NAMES"]) {
    foreach ($name in $config["SPEAKER_NAMES"].Split(",")) {
        $cliArgs += @("--speaker-name", $name.Trim())
    }
}

if ($config["HF_TOKEN"]) { $cliArgs += @("--hf-token", $config["HF_TOKEN"]) }
if ($config["VOCABULARY_FILE"]) { $cliArgs += @("--vocabulary-file", $config["VOCABULARY_FILE"]) }
if ($config["HOTWORDS"]) { $cliArgs += @("--hotwords", $config["HOTWORDS"]) }
if ($config["VERBOSE"] -eq "true") { $cliArgs += "--verbose" }

uv run audio-transcriber @cliArgs
exit $LASTEXITCODE
