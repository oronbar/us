@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0SZMC2Sync.ps1" -Mode Dashboard
endlocal
