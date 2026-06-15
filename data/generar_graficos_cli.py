# -*- coding: utf-8 -*-
"""
Generador de gráficas STANDALONE para los XLSX de la suite 2026-1.

Sirve para cuando el suite (`run_suite_2026.py`) se interrumpe o se cuelga antes
de llegar a la fase final de gráficas (p. ej. por el tiempo largo de las pruebas):
los resultados numéricos ya quedaron guardados en el XLSX (se guardan lote a lote),
pero las hojas 'Datos_Graficos' y 'Graficos' no se alcanzaron a crear.

Este script pregunta en la terminal cuál archivo de `results_test/` graficar y le
añade (o regenera) las hojas de gráficas SIN volver a correr ninguna prueba.

Uso:
    python data/generar_graficos_cli.py
    python data/generar_graficos_cli.py <archivo.xlsx>   # opcional: saltarse el menú

Reutiliza exactamente la misma lógica que el suite:
  - `generar_graficos(wb, PLAN_N, COLS)` de `graficos_suite.py`.
  - Las constantes `PLAN_N` y `COLS` de `run_suite_2026.py` (fuente única de verdad).
La operación es idempotente: si el archivo ya tenía gráficas, se reemplazan.
"""

import sys
from pathlib import Path

# La consola de Windows suele venir en cp1252 y revienta con caracteres como '✓' o
# acentos. Forzamos UTF-8 en la salida para imprimir sin sobresaltos.
for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import openpyxl

# El directorio del script (data/) ya está en sys.path al ejecutarse directamente,
# por lo que estos imports hermanos funcionan sin tocar PYTHONPATH.
from graficos_suite import generar_graficos
from run_suite_2026 import PLAN_N, COLS, RESULTS_TEST_DIR


def listar_xlsx(directorio: Path) -> list:
    """XLSX reales del directorio, ordenados por fecha de modificación (más reciente primero).

    Ignora los temporales de Excel (`~$...`).
    """
    if not directorio.is_dir():
        return []
    archivos = [
        p for p in directorio.glob("*.xlsx")
        if not p.name.startswith("~$")
    ]
    return sorted(archivos, key=lambda p: p.stat().st_mtime, reverse=True)


def formatear_tam(num_bytes: int) -> str:
    for unidad in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unidad == "GB":
            return f"{num_bytes:.0f} {unidad}" if unidad == "B" else f"{num_bytes:.1f} {unidad}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def elegir_archivo(archivos: list) -> Path:
    """Muestra un menú numerado y devuelve el archivo elegido (o None si se cancela)."""
    from datetime import datetime

    print(f"\nArchivos disponibles en {RESULTS_TEST_DIR}:\n")
    for i, p in enumerate(archivos, start=1):
        st = p.stat()
        fecha = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  [{i}] {p.name}  ({formatear_tam(st.st_size)}, {fecha})")
    print("  [q] Cancelar\n")

    while True:
        eleccion = input("¿A qué archivo le genero las gráficas? > ").strip().lower()
        if eleccion in ("q", "quit", "salir", ""):
            return None
        if eleccion.isdigit():
            idx = int(eleccion)
            if 1 <= idx <= len(archivos):
                return archivos[idx - 1]
        print(f"  Opción inválida. Ingresa un número entre 1 y {len(archivos)}, o 'q'.")


def resolver_argumento(arg: str, archivos: list) -> Path:
    """Resuelve el archivo pasado por línea de comandos (ruta o solo nombre)."""
    candidato = Path(arg)
    if candidato.is_file():
        return candidato
    # Buscar por nombre dentro de results_test/.
    por_nombre = RESULTS_TEST_DIR / arg
    if por_nombre.is_file():
        return por_nombre
    # Coincidencia parcial sobre los nombres listados.
    coincidencias = [p for p in archivos if arg.lower() in p.name.lower()]
    if len(coincidencias) == 1:
        return coincidencias[0]
    if len(coincidencias) > 1:
        print(f"'{arg}' coincide con varios archivos; sé más específico:")
        for p in coincidencias:
            print(f"  - {p.name}")
        return None
    print(f"No se encontró un XLSX que coincida con '{arg}' en {RESULTS_TEST_DIR}.")
    return None


def main() -> int:
    archivos = listar_xlsx(RESULTS_TEST_DIR)

    if len(sys.argv) > 1:
        if not archivos and not Path(sys.argv[1]).is_file():
            print(f"No hay archivos .xlsx en {RESULTS_TEST_DIR}.")
            return 1
        destino = resolver_argumento(sys.argv[1], archivos)
    else:
        if not archivos:
            print(f"No hay archivos .xlsx en {RESULTS_TEST_DIR}.")
            print("Corre primero el suite (data/run_suite_2026.py) para generar resultados.")
            return 1
        destino = elegir_archivo(archivos)

    if destino is None:
        print("Cancelado. No se generaron gráficas.")
        return 0

    print(f"\nAbriendo: {destino}")
    try:
        wb = openpyxl.load_workbook(destino)
    except Exception as exc:  # noqa: BLE001
        print(f"  No se pudo abrir el archivo: {type(exc).__name__}: {exc}")
        return 1

    print("Generando gráficas comparativas (QNodes vs Geométrica)...")
    try:
        genero = generar_graficos(wb, PLAN_N, COLS)
    except Exception as exc:  # noqa: BLE001
        print(f"  [gráficos] error al generar: {type(exc).__name__}: {exc}")
        return 1

    if not genero:
        print("  No había datos de K-particiones en el archivo; nada que graficar.")
        return 1

    try:
        wb.save(destino)
    except PermissionError:
        print(f"  No se pudo guardar: {destino.name} parece estar abierto en Excel. "
              "Ciérralo y reintenta.")
        return 1

    print(f"\n✓ Gráficas generadas y guardadas en: {destino}")
    print("  Hojas creadas/actualizadas: 'Datos_Graficos' y 'Graficos'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
