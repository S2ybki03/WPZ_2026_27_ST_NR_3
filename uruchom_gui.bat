@echo off
rem Uruchamia aplikacje (.venv jesli istnieje, inaczej systemowy Python)
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" src\gui.py
) else (
    start "" pythonw src\gui.py
)
