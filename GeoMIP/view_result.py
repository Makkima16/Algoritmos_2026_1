# -*- coding: utf-8 -*-
"""
view_result.py — Visualizador interactivo de resultados KGeoMIP.

Permite elegir un archivo JSON de la carpeta results/ y muestra
las particiones en formato visual legible.

Uso:
    python view_result.py
"""

import json
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

# Ruta fija del proyecto
RESULTS_DIR = Path(__file__).resolve().parent / 'results'

# Texto para conjuntos vacíos
VOID_STR = "∅"

# Si necesitas importar módulos locales
PROJECT_ROOT = RESULTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────


def particion_desde_json(datos: dict) -> str:
    """Reconstruye el string visual de una k-partición desde JSON."""

    partes_fmt = []

    for parte in datos.get("partes", []):

        fut = ",".join(parte.get("futuro", [])) \
            if parte.get("futuro") else VOID_STR

        pres = ",".join(parte.get("presente", [])) \
            if parte.get("presente") else VOID_STR

        ancho = max(len(fut), len(pres)) + 2

        partes_fmt.append(
            (
                f"|{fut:^{ancho}}|",
                f"|{pres:^{ancho}}|"
            )
        )

    linea_top = " ⊗ ".join(t for t, _ in partes_fmt)
    linea_bot = "   ".join(b for _, b in partes_fmt)

    return f"{linea_top}\n{linea_bot}"


def sep(titulo: str):
    print("\n" + "=" * 60)
    print(f"  {titulo}")
    print("=" * 60)


def elegir_archivo() -> Path | None:
    """Abre el explorador directamente en la carpeta results/."""

    archivos = sorted(
        RESULTS_DIR.glob("*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not archivos:
        print(f"\nNo hay archivos JSON en:\n{RESULTS_DIR}")
        return None

    # ── Explorador gráfico ─────────────────────────────
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        ruta_str = filedialog.askopenfilename(
            initialdir=str(RESULTS_DIR),
            title="Seleccionar resultado KGeoMIP",
            filetypes=[
                ("Archivos JSON", "*.json"),
                ("Todos los archivos", "*.*")
            ]
        )

        root.destroy()

        if ruta_str:
            return Path(ruta_str)

    except Exception as e:
        print(f"\nError abriendo explorador: {e}")

    # ── Fallback consola ───────────────────────────────
    print("\nArchivos disponibles:\n")

    for i, archivo in enumerate(archivos):
        print(f"[{i}] {archivo.name}")

    entrada = input(
        "\nElige número del archivo "
        "(Enter = más reciente): "
    ).strip()

    idx = int(entrada) if entrada.isdigit() else 0

    return archivos[min(idx, len(archivos) - 1)]


def mostrar_resultado(datos: dict):
    """Imprime el resultado completo."""

    sep(
        f"Dataset: {datos.get('dataset', '?')} | "
        f"Estado: {datos.get('estado_inicial', '?')}"
    )

    print(f"\nEstrategia : {datos.get('estrategia', '?')}")
    print(f"Pérdida φ  : {datos.get('perdida_phi', 0):.6f}")
    print(f"Tiempo     : {datos.get('tiempo_total', 0):.3f}s")

    print("\n── Partición óptima ─────────────────────────────")

    particion = datos.get("particion")

    if isinstance(particion, dict) and "partes" in particion:
        print(particion_desde_json(particion))
    else:
        print(particion)

    historico = datos.get("historico_comparaciones", [])

    if historico:

        print("\n── Histórico de k-particiones ─────────────────")

        for entrada in historico:

            k = entrada.get("k", "?")
            perdida = entrada.get("perdida", 0)

            print(f"\nk={k} | pérdida={perdida:.6f}")

            part = entrada.get("particion")

            if isinstance(part, dict) and "partes" in part:
                print(particion_desde_json(part))
            else:
                print(
                    entrada.get(
                        "particion_grafica",
                        str(part)
                    )
                )


def main():

    sep("Visualizador de Resultados KGeoMIP")

    if not RESULTS_DIR.exists():

        print(f"\nLa carpeta no existe:\n{RESULTS_DIR}")
        sys.exit(1)

    archivo = elegir_archivo()

    if not archivo:
        print("\nNo se seleccionó ningún archivo.")
        sys.exit(0)

    print(f"\nCargando: {archivo.name}")

    try:

        with open(archivo, encoding="utf-8") as f:
            datos = json.load(f)

    except json.JSONDecodeError:
        print("\nError: el archivo JSON está corrupto.")
        sys.exit(1)

    except Exception as e:
        print(f"\nError leyendo archivo: {e}")
        sys.exit(1)

    mostrar_resultado(datos)

    print("\n")


if __name__ == "__main__":
    main()