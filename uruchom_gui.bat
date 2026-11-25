@echo off
rem Uruchamia aplikacje (.venv jesli istnieje, inaczej systemowy Python)
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" src\gui.py
) else (
    python src\gui.py
)
if errorlevel 1 pause
