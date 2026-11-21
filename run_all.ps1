# Pełna seria eksperymentów: trening + ocena czterech modeli, potem porównanie.
# Uruchomienie (z folderu projektu):  powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"

foreach ($m in "mobilenet_v2", "efficientnet_b0", "resnet50", "simple_cnn") {
    Write-Host "===== $m =====" -ForegroundColor Cyan
    & $py src\train.py --model $m
    if ($LASTEXITCODE -ne 0) { throw "train $m" }
    & $py src\evaluate.py --run $m
    if ($LASTEXITCODE -ne 0) { throw "evaluate $m" }
}
& $py src\compare.py
