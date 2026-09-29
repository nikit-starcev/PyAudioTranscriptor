@echo off
REM Запуск веб-интерфейса AudioTranscriptor двойным щелчком.
REM Логика самого запуска — в web.ps1 (общий файл с запуском из PowerShell).
REM Остановить сервер можно по Ctrl+C в этом окне.

chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0web.ps1" %*
pause
