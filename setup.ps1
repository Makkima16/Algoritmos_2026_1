# setup.ps1 — Crea el entorno virtual e instala todas las dependencias (Windows / PowerShell)
# Uso:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $raiz

if (-not (Test-Path ".venv")) {
    Write-Host "Creando entorno virtual .venv ..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "Actualizando pip e instalando dependencias ..." -ForegroundColor Cyan
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "`nListo. Activa el entorno con:  .venv\Scripts\Activate.ps1" -ForegroundColor Green
