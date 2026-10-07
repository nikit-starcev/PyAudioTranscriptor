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
# ASR_BACKEND, WHISPER_CPP_MODEL/BINARY/LIB_PATH, LLM_* (LLM_ENABLED,
# LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL, LLM_MODEL_NAME, LLM_API_KEY, LLM_BINARY,
# LLM_LIB_PATH, LLM_GPU, LLM_CONTEXT, LLM_EXTRACT_NAMES, LLM_SUMMARY,
# LLM_SUGGEST_TERMS, LLM_PROMPT_EXTRA, LLM_PROMPT_FILE) и
# GLOSSARY_PATH — поэтому гибрид whisper.cpp/Vulkan и LLM-постобработка
# (локальная llama.cpp или внешний OpenAI-совместимый API) работают через
# config.env так же, как на Linux/macOS.
#
# Использование (PowerShell):
#   .\run.ps1 путь\к\записи.mp3
#
# Также можно перетащить аудиофайл мышью на run.bat — это то же самое,
# но без необходимости открывать PowerShell.
#
# Перед первым запуском:
#   Copy-Item config.example.env config.env
#   # в файле токены — ограничьте доступ (на Linux/macOS: chmod 600 config.env)
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

# Экспортируем все значения из config.env в окружение процесса — так же, как
# это делает run.sh (`set -a; source config.env`). Благодаря этому переменные,
# которые код читает напрямую из окружения (WHISPER_CPP_VAD_MODEL,
# WHISPER_CPP_CHUNK_SECONDS/OVERLAP, GLOSSARY_DB и т.п.), работают на Windows
# так же, как на Linux/macOS. Пустое значение снимает переменную.
foreach ($entry in $config.GetEnumerator()) {
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value)
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv не найден — устанавливаю..."
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

if (-not $config["HF_TOKEN"]) {
    Write-Host "Внимание: HF_TOKEN не задан в config.env - определение говорящих завершится ошибкой (см. README)."
}

# Маппинг config.env → флаги CLI живёт в одном месте — в коде
# (audio_transcriber.cli.env_config): CLI сам читает config.env (как веб и TUI),
# а флаги лишь переопределяют настройки. Раньше обёртки дублировали этот
# маппинг, и он расходился (CLEAR_CACHE и env-only WHISPER_CPP_CHUNK_*,
# issue #90). Поэтому здесь не перечисляем переменные, а передаём только
# обязательный путь к файлу: остальное подхватит CLI.
#
# Значения config.env уже экспортированы в окружение процесса выше, поэтому
# переменные, которые код читает напрямую из окружения
# (WHISPER_CPP_VAD_MODEL, WHISPER_CPP_CHUNK_* и т.п.), тоже доступны.
#
# Секреты (HF_TOKEN, LLM_API_KEY) намеренно НЕ пробрасываются флагами argv:
# они видны в списке процессов. CLI берёт их из окружения/файла сам.

# Для бэкенда whisper-cpp (гибрид на AMD) используется CPU-сборка torch,
# установленная вручную в .venv. `uv run` сверяется с uv.lock и может
# переустановить CUDA-сборку torch, поэтому, если бинарник уже есть,
# вызываем его напрямую — как в run.sh на Linux/macOS.
$venvExe = Join-Path $PSScriptRoot ".venv\Scripts\audio-transcriber.exe"
if (Test-Path $venvExe -PathType Leaf) {
    & $venvExe transcribe $AudioFile
} else {
    uv run audio-transcriber transcribe $AudioFile
}
exit $LASTEXITCODE
