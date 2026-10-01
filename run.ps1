# Универсальный запуск AudioTranscriptor для Windows.
#
# Что делает:
#   1. Устанавливает uv, если он ещё не установлен.
#   2. Подтягивает Python нужной версии и все зависимости (это делает uv run
#      автоматически при первом запуске — отдельно ничего ставить не нужно).
#   3. Запускает транскрибацию с параметрами из config.env.
#
# Разбор config.env выполняется без bash: строки вида КЛЮЧ=значение, пустые
# строки и комментарии (#) пропускаются, окружающие кавычки у значений
# снимаются. Поддерживаются те же переменные, что и в run.sh, в том числе
# ASR_BACKEND, WHISPER_CPP_MODEL/BINARY/LIB_PATH, LLM_* (LLM_ENABLED, LLM_MODEL,
# LLM_BINARY, LLM_LIB_PATH, LLM_GPU, LLM_CONTEXT, LLM_EXTRACT_NAMES,
# LLM_SUMMARY, LLM_SUGGEST_TERMS, LLM_PROMPT_EXTRA, LLM_PROMPT_FILE) и
# GLOSSARY_PATH — поэтому гибрид whisper.cpp/Vulkan и LLM-постобработка
# работают через config.env так же, как на Linux/macOS.
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

# Разбор config.env: КЛЮЧ=значение, без комментариев и с снятием кавычек.
$config = @{}
Get-Content $configPath -Encoding UTF8 | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
        $key, $value = $line.Split("=", 2)
        $config[$key.Trim()] = $value.Trim().Trim('"').Trim("'")
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

# Пара «флаг + значение», если значение задано (не пусто).
function Get-ValueArg([string]$name, [string]$key) {
    $value = $config[$key]
    if ($value) {
        return @($name, $value)
    }
    return @()
}

$cliArgs = @("transcribe", $AudioFile)

$cliArgs += Get-ValueArg "--output-dir" "OUTPUT_DIR"
$cliArgs += Get-ValueArg "--model" "MODEL"
$cliArgs += Get-ValueArg "--language" "LANGUAGE"
$cliArgs += Get-ValueArg "--device" "DEVICE"
$cliArgs += Get-ValueArg "--asr-backend" "ASR_BACKEND"
$cliArgs += Get-ValueArg "--whisper-cpp-model" "WHISPER_CPP_MODEL"
$cliArgs += Get-ValueArg "--whisper-cpp-binary" "WHISPER_CPP_BINARY"
$cliArgs += Get-ValueArg "--whisper-cpp-lib-path" "WHISPER_CPP_LIB_PATH"

if ($config["FORMATS"]) {
    foreach ($fmt in $config["FORMATS"].Split(",")) {
        $cliArgs += @("--format", $fmt.Trim())
    }
}

$cliArgs += Get-ValueArg "--num-speakers" "NUM_SPEAKERS"
$cliArgs += Get-ValueArg "--min-speakers" "MIN_SPEAKERS"
$cliArgs += Get-ValueArg "--max-speakers" "MAX_SPEAKERS"
$cliArgs += Get-ValueArg "--min-duration-off" "DIARIZATION_MIN_DURATION_OFF"
$cliArgs += Get-ValueArg "--clustering-threshold" "DIARIZATION_CLUSTERING_THRESHOLD"
$cliArgs += Get-ValueArg "--clustering-fb" "DIARIZATION_CLUSTERING_FB"

if ($config["SPEAKER_NAMES"]) {
    foreach ($name in $config["SPEAKER_NAMES"].Split(",")) {
        $cliArgs += @("--speaker-name", $name.Trim())
    }
}

if ($config["SPEAKER_REFERENCES"]) {
    foreach ($ref in $config["SPEAKER_REFERENCES"].Split(",")) {
        $cliArgs += @("--speaker-reference", $ref.Trim())
    }
}

$cliArgs += Get-ValueArg "--enrollment-min-similarity" "ENROLLMENT_MIN_SIMILARITY"
$cliArgs += Get-ValueArg "--voices-dir" "VOICES_DIR"
if ($config["EXPORT_SPEAKER_SAMPLES"] -eq "false") { $cliArgs += "--no-speaker-samples" }

$cliArgs += Get-ValueArg "--hf-token" "HF_TOKEN"
$cliArgs += Get-ValueArg "--pyannote-local-model" "PYANNOTE_LOCAL_MODEL"
if ($config["ENABLE_CORRECTION"] -eq "true") { $cliArgs += "--enable-correction" }
if ($config["CLEAN_ARTIFACTS"] -eq "false") { $cliArgs += "--no-clean" }
if ($config["COLLAPSE_REPEATS"] -eq "false") { $cliArgs += "--no-collapse-repeats" }
$cliArgs += Get-ValueArg "--repeat-min-words" "REPEAT_MIN_WORDS"
$cliArgs += Get-ValueArg "--repeat-similarity" "REPEAT_SIMILARITY"
if ($config["NORMALIZE_TEXT"] -eq "false") { $cliArgs += "--no-normalize" }
if ($config["MARK_OVERLAP"] -eq "false") { $cliArgs += "--no-overlap" }
$cliArgs += Get-ValueArg "--low-confidence-threshold" "LOW_CONFIDENCE_THRESHOLD"
if ($config["DENOISE"] -eq "false") { $cliArgs += "--no-denoise" }
if ($config["USE_CACHE"] -eq "false") { $cliArgs += "--no-cache" }
if ($config["CLEAR_CACHE"] -eq "true") { $cliArgs += "--clear-cache" }
$cliArgs += Get-ValueArg "--cache-dir" "CACHE_DIR"
if ($config["NOTIFICATIONS"] -eq "false") { $cliArgs += "--no-notify" }
if ($config["TIMELINE"] -eq "false") { $cliArgs += "--no-timeline" }
$cliArgs += Get-ValueArg "--correction-min-word-length" "CORRECTION_MIN_WORD_LENGTH"
$cliArgs += Get-ValueArg "--correction-min-similarity" "CORRECTION_MIN_SIMILARITY"
$cliArgs += Get-ValueArg "--correction-max-candidates" "CORRECTION_MAX_CANDIDATES"
$cliArgs += Get-ValueArg "--hotwords" "HOTWORDS"
if ($config["VERBOSE"] -eq "true") { $cliArgs += "--verbose" }

# --- LLM-постобработка (llama.cpp) ---
if ($config["LLM_ENABLED"] -eq "true") { $cliArgs += "--llm" }
$cliArgs += Get-ValueArg "--llm-model" "LLM_MODEL"
$cliArgs += Get-ValueArg "--llm-binary" "LLM_BINARY"
$cliArgs += Get-ValueArg "--llm-lib-path" "LLM_LIB_PATH"
if ($config["LLM_GPU"] -eq "false") { $cliArgs += "--llm-cpu" }
$cliArgs += Get-ValueArg "--llm-context" "LLM_CONTEXT"
if ($config["LLM_EXTRACT_NAMES"] -eq "false") { $cliArgs += "--llm-no-names" }
if ($config["LLM_SUMMARY"] -eq "false") { $cliArgs += "--no-llm-summary" }
if ($config["LLM_SUGGEST_TERMS"] -eq "true") { $cliArgs += "--llm-suggest-terms" }
$cliArgs += Get-ValueArg "--llm-prompt-extra" "LLM_PROMPT_EXTRA"
$cliArgs += Get-ValueArg "--llm-prompt-file" "LLM_PROMPT_FILE"
$cliArgs += Get-ValueArg "--glossary" "GLOSSARY_PATH"

# Для бэкенда whisper-cpp (гибрид на AMD) используется CPU-сборка torch,
# установленная вручную в .venv. `uv run` сверяется с uv.lock и может
# переустановить CUDA-сборку torch, поэтому, если бинарник уже есть,
# вызываем его напрямую — как в run.sh на Linux/macOS.
$venvExe = Join-Path $PSScriptRoot ".venv\Scripts\audio-transcriber.exe"
if (Test-Path $venvExe -PathType Leaf) {
    & $venvExe @cliArgs
} else {
    uv run audio-transcriber @cliArgs
}
exit $LASTEXITCODE
