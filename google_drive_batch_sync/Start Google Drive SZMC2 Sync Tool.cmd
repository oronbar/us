@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0GoogleDriveSZMC2Sync.ps1" -Mode Dashboard
endlocal
