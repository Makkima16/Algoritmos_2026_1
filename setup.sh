#!/usr/bin/env bash
# setup.sh — Crea el entorno virtual e instala todas las dependencias (Git Bash / Linux / macOS)
# Uso:  bash setup.sh
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creando entorno virtual .venv ..."
    python -m venv .venv
fi

# Ruta del python del venv (Windows usa Scripts/, Unix usa bin/)
if [ -f ".venv/Scripts/python.exe" ]; then
    PY=".venv/Scripts/python.exe"
else
    PY=".venv/bin/python"
fi

echo "Actualizando pip e instalando dependencias ..."
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt

echo ""
echo "Listo. Activa el entorno con:  source .venv/Scripts/activate  (o .venv/bin/activate en Linux/macOS)"
