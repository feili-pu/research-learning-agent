@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0check-env.ps1" %*
