# -*- coding: utf-8 -*-
"""
Lectura de datasets, CSV de pruebas y resultados guardados (JSON manual + XLSX bloque),
y utilidades de conversión etiquetas→binario para la ejecución por bloque desde la GUI.
"""

import re
import csv
import json
from pathlib import Path

import openpyxl

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parents[1]

ALGO_ROOTS = {
    "qnodes": REPO_ROOT / "QNodes",
    "geomip": REPO_ROOT / "GeoMIP",
}

# Fuente ÚNICA y compartida de datasets/pruebas para ambos algoritmos
# (la misma carpeta canónica que usa el suite runner).
DATA_SCRIPTS = REPO_ROOT / "data_scripts"

# Carpeta de resultados "manuales" — difiere por algoritmo.
MANUAL_DIRNAME = {"qnodes": "manual", "geomip": "manually"}


# ── Etiquetas de nodo ──────────────────────────────────────────────────────

def labels_para(n: int) -> list[str]:
    """Genera etiquetas A..Z, AA, AB... (base-26 biyectiva)."""
    out = []
    for i in range(n):
        s, x = "", i
        while True:
            s = chr(ord("A") + x % 26) + s
            x = x // 26 - 1
            if x < 0:
                break
        out.append(s)
    return out


def letras_a_binario(etiquetas: str, n_nodos: int) -> str:
    """Convierte etiquetas de nodos (ej 'ABCEFG') a máscara binaria de longitud n_nodos."""
    etiquetas = (etiquetas or "").strip().upper()
    validas = labels_para(n_nodos)
    set_validas = set(validas)
    activos: set[str] = set()
    idx = 0
    while idx < len(etiquetas):
        matched = False
        for length in range(min(3, len(etiquetas) - idx), 0, -1):
            lbl = etiquetas[idx:idx + length]
            if lbl in set_validas:
                activos.add(lbl)
                idx += length
                matched = True
                break
        if not matched:
            idx += 1
    return "".join("1" if validas[i] in activos else "0" for i in range(n_nodos))


# ── Datasets (TPMs) ────────────────────────────────────────────────────────

def listar_datasets(algo: str | None = None) -> list[dict]:
    # Datasets compartidos desde data_scripts/ (idénticos para ambos algoritmos).
    salida = []
    for tipo, carpeta in (("binaria", "samples_binary"), ("no_binaria", "samples_no_binary")):
        d = DATA_SCRIPTS / carpeta
        if not d.exists():
            continue
        for f in sorted(d.glob("*.csv")):
            m = re.match(r"^N(\d+)", f.stem)
            salida.append({
                "archivo": f.name,
                "ruta": str(f),
                "n": int(m.group(1)) if m else None,
                "tipo": tipo,
            })
    return salida


# ── CSV de pruebas ─────────────────────────────────────────────────────────

def listar_pruebas(algo: str | None = None, n: int | None = None) -> list[dict]:
    # CSVs de pruebas compartidos desde data_scripts/Pruebas.
    d = DATA_SCRIPTS / "Pruebas"
    salida = []
    if not d.exists():
        return salida
    for f in sorted(d.glob("*.csv")):
        m = re.search(r"N(\d+)", f.stem)
        f_n = int(m.group(1)) if m else None
        if n is not None and f_n != n:
            continue
        salida.append({"archivo": f.name, "ruta": str(f), "n": f_n})
    return salida


def leer_csv_pruebas(ruta: str) -> list[dict]:
    """Lee un CSV de pruebas y devuelve filas normalizadas {num, alcance, mecanismo}."""
    filas = []
    with open(ruta, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            def _g(*nombres):
                for nombre in nombres:
                    for k, v in row.items():
                        if k and k.strip().lower() == nombre.lower():
                            return (v or "").strip()
                return ""
            filas.append({
                "num": _g("#prueba", "prueba", "num", "n") or str(i + 1),
                "alcance": _g("alcance o purview (t+1)", "alcance", "purview", "effect"),
                "mecanismo": _g("mecanismo(t)", "mecanismo", "mechanism", "cause"),
            })
    return filas


# ── Resultados guardados ───────────────────────────────────────────────────

def listar_resultados(algo: str, tipo: str) -> list[dict]:
    """tipo ∈ {manual, block}. Devuelve archivos con metadatos básicos."""
    root = ALGO_ROOTS[algo]
    salida = []
    if tipo == "manual":
        d = root / "results" / MANUAL_DIRNAME[algo]
        if d.exists():
            for f in sorted(d.glob("*.json"), reverse=True):
                salida.append({
                    "archivo": f.name,
                    "ruta": str(f),
                    "formato": "json",
                    "mtime": f.stat().st_mtime,
                })
    elif tipo == "block":
        d = root / "results" / "block"
        if d.exists():
            for f in sorted(d.rglob("*.xlsx"), reverse=True):
                salida.append({
                    "archivo": f.name,
                    "ruta": str(f),
                    "subcarpeta": f.parent.name,
                    "formato": "xlsx",
                    "mtime": f.stat().st_mtime,
                })
    return salida


def _leer_xlsx(ruta: str) -> dict:
    """Lee un XLSX de bloque: encabezados de la fila 1 y filas de datos."""
    wb = openpyxl.load_workbook(ruta, read_only=True, data_only=True)
    ws = wb.active
    filas = list(ws.iter_rows(values_only=True))
    wb.close()
    if not filas:
        return {"columnas": [], "filas": []}
    columnas = [str(c) if c is not None else "" for c in filas[0]]
    datos = []
    for fila in filas[1:]:
        datos.append({columnas[i] if i < len(columnas) else f"col{i}": fila[i]
                      for i in range(len(fila))})
    return {"columnas": columnas, "filas": datos}


def leer_detalle(ruta: str) -> dict:
    """Devuelve el contenido parseado de un resultado (JSON manual o XLSX bloque)."""
    p = Path(ruta)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {ruta}")
    if p.suffix.lower() == ".json":
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        return {"tipo": "manual", "data": data}
    if p.suffix.lower() == ".xlsx":
        return {"tipo": "block", "data": _leer_xlsx(ruta)}
    raise ValueError(f"Formato no soportado: {p.suffix}")
