@echo off
setlocal
cd /d "%~dp0.."
"C:\Program Files\PowerShell\7\pwsh.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_llm.ps1"

