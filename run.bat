@echo off
REM Позволяет запускать AudioTranscriptor двойным щелчком/перетаскиванием
REM аудиофайла на этот .bat, без открытия PowerShell вручную.
REM Логика самого запуска — в run.ps1 (общий с "ручным" запуском из PowerShell).

if "%~1"=="" (
    echo Перетащите аудиофайл на run.bat, либо запустите: run.bat путь\к\записи.mp3
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
pause
