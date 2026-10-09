@echo off
REM Запуск портативного бандла AudioTranscriptor (веб-интерфейс) двойным щелчком.
REM Вся логика — в run-web.ps1 (общий файл с запуском из PowerShell).
REM Остановить сервер можно по Ctrl+C в этом окне.

chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-web.ps1" %*
pause
