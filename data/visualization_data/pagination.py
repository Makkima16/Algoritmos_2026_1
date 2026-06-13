# -*- coding: utf-8 -*-
"""
pagination.py — Visor de TPM por Chunks/Bloques

Permite seleccionar un archivo CSV con una TPM (Transition Probability Matrix)
y visualizarlo en bloques paginados desde la terminal, sin cargar todo en memoria.

Uso:
    python pagination.py
    python pagination.py --chunk 128
    python pagination.py --archivo ruta/al/archivo.csv --chunk 256
"""

import sys
import argparse
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Utilidades de presentación
# ─────────────────────────────────────────────

def sep(titulo: str, ancho: int = 62):
    print(f"\n{'═' * ancho}")
    print(f"  {titulo}")
    print('═' * ancho)


def encabezado_columnas(n_nodos: int) -> str:
    """Genera la línea de encabezado con los nombres de nodo."""
    nombres = [f"  N{i:<5}" for i in range(n_nodos)]
    return "Estado  │" + "".join(nombres)


def formatear_fila(estado_bin: str, valores: list) -> str:
    """Formatea una fila: etiqueta binaria + valores de probabilidad."""
    celdas = [f"  {v:.4f}" for v in valores]
    return f"{estado_bin:<8}│{''.join(celdas)}"


# ─────────────────────────────────────────────
#  Selección de archivo
# ─────────────────────────────────────────────

# parents[1] = carpeta data/ de la raíz.
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples_binary"


def seleccionar_archivo_gui() -> Path | None:
    """Abre el explorador de archivos nativo para seleccionar un CSV."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        ruta = filedialog.askopenfilename(
            initialdir=str(SAMPLES_DIR) if SAMPLES_DIR.exists() else str(Path.home()),
            title="Seleccionar TPM (archivo CSV)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()
        return Path(ruta) if ruta else None
    except Exception as e:
        print(f"  ⚠ No se pudo abrir el explorador gráfico: {e}")
        return None


def seleccionar_archivo_terminal() -> Path | None:
    """Fallback: selección manual desde la terminal."""
    print("\n  Ingrese la ruta al archivo CSV (o presione Enter para cancelar):")
    ruta = input("  > ").strip()
    return Path(ruta) if ruta else None


# ─────────────────────────────────────────────
#  Lógica principal de paginación
# ─────────────────────────────────────────────

def contar_filas(ruta: Path) -> int:
    """Cuenta filas del CSV sin cargarlo completo en memoria."""
    with open(ruta, 'r') as f:
        return sum(1 for _ in f)


def calcular_chunk_optimo(total_filas: int) -> int:
    """
    Sugiere un tamaño de chunk razonable según el total de estados.
    Siempre una potencia de 2 (coherente con la estructura binaria de la TPM).
    """
    if total_filas <= 64:
        return total_filas          # Mostrar todo de una vez
    elif total_filas <= 1_024:
        return 64
    elif total_filas <= 32_768:     # N ≤ 15
        return 256
    elif total_filas <= 1_048_576:  # N ≤ 20
        return 512
    else:
        return 1024


def visualizar_chunk(chunk_idx: int, n_chunks: int, chunk: pd.DataFrame,
                     estado_offset: int, n_nodos: int):
    """Imprime en terminal un bloque de la TPM con etiquetas binarias."""
    estado_inicio = estado_offset
    estado_fin = estado_offset + len(chunk) - 1

    sep(f"Chunk {chunk_idx + 1}/{n_chunks}  │  Estados {estado_inicio} – {estado_fin}"
        f"  │  ({len(chunk)} filas × {n_nodos} nodos)")

    # Encabezado de columnas
    print(encabezado_columnas(n_nodos))
    print("─" * (9 + n_nodos * 8))

    # Filas con etiqueta binaria
    for local_idx, (_, fila) in enumerate(chunk.iterrows()):
        estado_global = estado_inicio + local_idx
        etiqueta = format(estado_global, f'0{n_nodos}b')
        print(formatear_fila(etiqueta, fila.tolist()))


def menu_navegacion(chunk_actual: int, n_chunks: int) -> str:
    """
    Muestra las opciones de navegación y retorna la acción elegida.
    Retorna: 'siguiente', 'anterior', 'saltar', 'salir'
    """
    print("\n" + "─" * 62)
    opciones = []
    if chunk_actual < n_chunks - 1:
        opciones.append("[S] Siguiente")
    if chunk_actual > 0:
        opciones.append("[A] Anterior")
    opciones += ["[I] Ir a chunk", "[Q] Salir"]

    print("  " + "   ".join(opciones))
    print("─" * 62)
    eleccion = input("  Acción: ").strip().upper()

    if eleccion == 'S' and chunk_actual < n_chunks - 1:
        return 'siguiente'
    elif eleccion == 'A' and chunk_actual > 0:
        return 'anterior'
    elif eleccion == 'I':
        return 'saltar'
    elif eleccion == 'Q':
        return 'salir'
    else:
        print("  ⚠ Opción no válida.")
        return menu_navegacion(chunk_actual, n_chunks)


def visor_paginado(ruta: Path, chunk_size: int):
    """
    Núcleo del visor: carga chunks bajo demanda y permite navegar entre ellos.
    """
    sep(f"TPM Viewer  │  {ruta.name}")

    # ── Metadatos del archivo ──────────────────────────────────────
    total_filas = contar_filas(ruta)
    # Inferir N desde el número de columnas (primera fila)
    primera = pd.read_csv(ruta, header=None, nrows=1)
    n_nodos = primera.shape[1]
    n_chunks = (total_filas + chunk_size - 1) // chunk_size
    n_esperado = int(np.log2(total_filas)) if total_filas & (total_filas - 1) == 0 else "~"

    print(f"\n  Archivo   : {ruta}")
    print(f"  Filas     : {total_filas:,}  (2^{n_nodos} = {2**n_nodos})")
    print(f"  Nodos (N) : {n_nodos}")
    print(f"  Chunk size: {chunk_size} estados por bloque")
    print(f"  Total     : {n_chunks} chunks")

    if total_filas != 2 ** n_nodos:
        print(f"\n  ⚠ Advertencia: se esperaban {2**n_nodos} filas para N={n_nodos}, "
              f"pero el archivo tiene {total_filas}. Verifique el CSV.")

    # ── Pre-cargar índices de chunks (offsets de línea) ───────────
    # Guardamos cada chunk como slice de índices para acceso O(1)
    offsets = list(range(0, total_filas, chunk_size))  # inicio de cada chunk

    chunk_actual = 0

    while True:
        offset = offsets[chunk_actual]
        chunk = pd.read_csv(
            ruta,
            header=None,
            skiprows=offset,
            nrows=chunk_size
        )

        visualizar_chunk(chunk_actual, n_chunks, chunk, offset, n_nodos)

        if n_chunks == 1:
            print("\n  (Archivo completo en un solo chunk. Presione Enter para salir.)")
            input()
            break

        accion = menu_navegacion(chunk_actual, n_chunks)

        if accion == 'siguiente':
            chunk_actual += 1
        elif accion == 'anterior':
            chunk_actual -= 1
        elif accion == 'saltar':
            try:
                destino = int(input(f"  Ir al chunk (1 – {n_chunks}): ").strip()) - 1
                if 0 <= destino < n_chunks:
                    chunk_actual = destino
                else:
                    print(f"  ⚠ Número fuera de rango (1–{n_chunks}).")
            except ValueError:
                print("  ⚠ Ingrese un número entero válido.")
        elif accion == 'salir':
            print("\n  ✓ Visor cerrado.\n")
            break


# ─────────────────────────────────────────────
#  Punto de entrada
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visor paginado de TPM en formato CSV (proyecto KGeoMIP)."
    )
    parser.add_argument(
        "--archivo", "-f",
        type=str, default=None,
        help="Ruta directa al CSV. Si se omite, se abre el explorador de archivos."
    )
    parser.add_argument(
        "--chunk", "-c",
        type=int, default=None,
        help="Número de filas por chunk. Por defecto se calcula automáticamente."
    )
    args = parser.parse_args()

    sep("TPM Chunk Viewer — KGeoMIP")

    # ── Resolución del archivo ─────────────────────────────────────
    if args.archivo:
        ruta = Path(args.archivo)
    else:
        print("\n  Abriendo explorador de archivos...")
        ruta = seleccionar_archivo_gui()
        if ruta is None:
            print("  Explorador cancelado. Selección manual:")
            ruta = seleccionar_archivo_terminal()

    if ruta is None or not ruta.exists():
        print(f"\n  ✗ Error: Archivo no encontrado → {ruta}")
        sys.exit(1)

    if ruta.suffix.lower() != ".csv":
        print(f"\n  ⚠ El archivo seleccionado no es un .csv: {ruta.name}")
        continuar = input("  ¿Continuar de todas formas? (s/n): ").strip().lower()
        if continuar != 's':
            sys.exit(0)

    # ── Resolución del chunk_size ──────────────────────────────────
    if args.chunk:
        chunk_size = args.chunk
    else:
        total = sum(1 for _ in open(ruta))
        chunk_size = calcular_chunk_optimo(total)
        print(f"\n  Chunk size automático: {chunk_size} estados por bloque")
        override = input(f"  ¿Usar otro valor? (Enter para confirmar {chunk_size}): ").strip()
        if override.isdigit() and int(override) > 0:
            chunk_size = int(override)

    # ── Iniciar visor ──────────────────────────────────────────────
    visor_paginado(ruta, chunk_size)


if __name__ == "__main__":
    main()