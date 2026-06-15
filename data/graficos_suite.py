# -*- coding: utf-8 -*-
"""
Generación de gráficas comparativas (QNodes vs Geométrica) para la suite 2026-1.

Se invoca al final de run_suite_2026.py sobre el libro ya diligenciado. Lee TODAS
las hojas N disponibles y construye dos hojas nuevas:

  • "Datos_Graficos" : tablas auxiliares (una por k) que alimentan las gráficas.
  • "Graficos"       : las gráficas nativas de Excel.

Tres familias de comparaciones (sobre resultados de K-particiones, k=3,4,5):

  (a) Tiempo de ejecución vs TAMAÑO del subsistema (nº de nodos activos del
      alcance), GLOBAL (todas las redes juntas), un gráfico por k, series QNodes
      vs Geométrica. Muestra cómo crece el tiempo al aumentar el tamaño.
  (b) Pérdida (Φ) por subsistema: QNodes vs Geométrica, un gráfico por k, más un
      conteo de "¿se halló la misma k-partición?" (comparando las particiones
      normalizadas de ambos motores).
  (c) Variación de la pérdida tomando QNodes como REFERENCIA: Δ = Φ_Geom − Φ_QNodes
      por subsistema, un gráfico por k (barras alrededor de 0).

Los tiempos en el libro están como texto formateado ("0.0865 s", "1 min 48.70 s",
"2 h 3 min 4 s"); aquí se parsean a segundos para poder graficarlos.
"""

import re
from typing import Optional

from openpyxl.chart import ScatterChart, BarChart, LineChart, Reference, Series
from openpyxl.chart.marker import Marker
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

KS = (3, 4, 5)

_HDR_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
_BOLD = Font(bold=True)
_CENTER = Alignment(horizontal="center")


# ── Parseo de tiempo formateado → segundos ──────────────────────────────────

def parse_tiempo(valor) -> Optional[float]:
    """Convierte '0.0865 s' / '1 min 48.70 s' / '2 h 3 min 4 s' a segundos (float)."""
    if valor is None:
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    s = str(valor).strip()
    if not s:
        return None
    h = re.search(r"([\d.]+)\s*h", s)
    m = re.search(r"([\d.]+)\s*min", s)
    seg = re.search(r"([\d.]+)\s*s(?!\w)", s)
    total = 0.0
    encontrado = False
    if h:
        total += float(h.group(1)) * 3600.0; encontrado = True
    if m:
        total += float(m.group(1)) * 60.0; encontrado = True
    if seg:
        total += float(seg.group(1)); encontrado = True
    return total if encontrado else None


# ── Normalización de particiones (para comparar entre motores) ───────────────

def _canon_particion(texto) -> Optional[frozenset]:
    """
    Lleva una partición (texto de 2 líneas) a forma canónica comparable entre
    motores: frozenset de (frozenset futuros, frozenset presentes), en MAYÚSCULAS.

    Funciona para ambos formatos: QNodes (⎛ ⎞⎝ ⎠) y Geométrica (| |), porque ambos
    ponen futuros en la línea 1 y presentes en la línea 2.
    """
    if not texto:
        return None
    s = str(texto)
    if s.strip().upper().startswith("ERROR"):
        return None
    lineas = s.split("\n")
    if len(lineas) < 2:
        lineas = lineas + [""]

    def celdas(linea: str):
        t = re.sub(r"[⎛⎝⎞⎠]", "|", linea)
        return [c.strip() for c in t.split("|") if c.strip() != ""]

    fut = celdas(lineas[0])
    pres = celdas(lineas[1])
    if not fut:
        return None

    partes = []
    for i, fcell in enumerate(fut):
        fset = (frozenset() if fcell == "∅"
                else frozenset(tok.strip().upper() for tok in fcell.split(",") if tok.strip()))
        pcell = pres[i] if i < len(pres) else "∅"
        pset = (frozenset() if pcell == "∅"
                else frozenset(tok.strip().upper() for tok in pcell.split(",") if tok.strip()))
        partes.append((fset, pset))
    return frozenset(partes)


def _num(valor) -> Optional[float]:
    """Devuelve float si el valor es numérico (no ERROR/None/texto)."""
    if isinstance(valor, (int, float)):
        return float(valor)
    return None


# ── Recolección de registros desde las hojas N ───────────────────────────────

def _recolectar(wb, plan_n, cols) -> "list[dict]":
    """
    Recorre las hojas N disponibles y junta un registro por (hoja, prueba, k) con
    pérdida/tiempo de ambos motores. Solo incluye registros con datos completos.
    """
    registros = []
    for n, (hoja, _tpm) in plan_n.items():
        if hoja not in wb.sheetnames:
            continue
        ws = wb[hoja]
        fila = 6
        while True:
            alc = ws.cell(fila, 2).value
            if alc is None or str(alc).strip() == "":
                break  # fin de pruebas (las filas de resumen tienen alcance vacío)
            alc_lbl = str(alc).strip()
            size = len(set(alc_lbl.upper()))
            mec = ws.cell(fila, 3).value
            mec_lbl = str(mec).strip() if mec is not None else ""
            # Las pruebas empiezan en la fila 6 y la col 1 es =ROW(A1)=1, =ROW(A2)=2, …
            # ⇒ el nº de prueba es fila-5 (no depende de leer la fórmula de la celda).
            prueba = fila - 5
            for k in KS:
                c_qn = cols[k]["qnodes"]
                c_ge = cols[k]["geomip"]
                qn_loss = _num(ws.cell(fila, c_qn[1]).value)
                ge_loss = _num(ws.cell(fila, c_ge[1]).value)
                qn_t = parse_tiempo(ws.cell(fila, c_qn[2]).value)
                ge_t = parse_tiempo(ws.cell(fila, c_ge[2]).value)
                if None in (qn_loss, ge_loss, qn_t, ge_t):
                    continue
                qn_part_txt = ws.cell(fila, c_qn[0]).value
                ge_part_txt = ws.cell(fila, c_ge[0]).value
                qn_part = _canon_particion(qn_part_txt)
                ge_part = _canon_particion(ge_part_txt)
                misma = (qn_part is not None and ge_part is not None and qn_part == ge_part)
                registros.append({
                    "n": n, "prueba": prueba, "alcance": alc_lbl, "mecanismo": mec_lbl,
                    "size": size, "k": k,
                    "qn_loss": qn_loss, "ge_loss": ge_loss,
                    "qn_t": qn_t, "ge_t": ge_t,
                    "delta": ge_loss - qn_loss, "misma": misma,
                    "qn_part_txt": str(qn_part_txt) if qn_part_txt is not None else "",
                    "ge_part_txt": str(ge_part_txt) if ge_part_txt is not None else "",
                })
            fila += 1
    return registros


# ── Construcción de hojas + gráficas ─────────────────────────────────────────

def _escribir_encabezado(ws, fila, valores):
    for j, v in enumerate(valores, 1):
        c = ws.cell(fila, j, v)
        c.font = _BOLD
        c.fill = _HDR_FILL
        c.alignment = _CENTER


def generar_graficos(wb, plan_n, cols, log=print) -> bool:
    """
    Construye las hojas 'Datos_Graficos' y 'Graficos' en el libro `wb`.

    Devuelve True si generó gráficas; False si no había datos.
    """
    registros = _recolectar(wb, plan_n, cols)
    if not registros:
        log("  [gráficos] no hay datos de K-particiones para graficar — se omite.")
        return False

    # Hojas limpias (idempotente ante re-ejecuciones).
    for nombre in ("Graficos", "Datos_Graficos"):
        if nombre in wb.sheetnames:
            del wb[nombre]
    ws_dat = wb.create_sheet("Datos_Graficos")
    ws_g = wb.create_sheet("Graficos")

    # Columnas de la tabla auxiliar (1-indexado):
    #  1 # (nº de prueba) | 2 N | 3 Alcance | 4 Mecanismo | 5 Tamaño |
    #  6 Φ QNodes | 7 Φ Geom | 8 t QNodes | 9 t Geom | 10 Δ Φ | 11 ¿Misma? |
    # 12 Partición QNodes | 13 Partición Geom
    HEADER = ["#", "N", "Alcance", "Mecanismo", "Tamaño", "Φ QNodes", "Φ Geom",
              "t QNodes (s)", "t Geom (s)", "Δ Φ (Geom−QN)", "¿Misma part?",
              "Partición QNodes", "Partición Geom"]
    NCOL = len(HEADER)
    _PART_WRAP = Alignment(wrap_text=True, vertical="top")

    rangos = {}      # k -> (header_row, data_start, data_end)
    fila = 1
    for k in KS:
        regs_k = sorted([r for r in registros if r["k"] == k],
                        key=lambda r: (r["n"], r["size"], r["prueba"]))
        if not regs_k:
            continue
        ws_dat.cell(fila, 1, f"k = {k}").font = _BOLD
        fila += 1
        header_row = fila
        _escribir_encabezado(ws_dat, header_row, HEADER)
        fila += 1
        data_start = fila
        for r in regs_k:
            ws_dat.cell(fila, 1, r["prueba"])
            ws_dat.cell(fila, 2, r["n"])
            ws_dat.cell(fila, 3, r["alcance"])
            ws_dat.cell(fila, 4, r["mecanismo"])
            ws_dat.cell(fila, 5, r["size"])
            ws_dat.cell(fila, 6, round(r["qn_loss"], 6))
            ws_dat.cell(fila, 7, round(r["ge_loss"], 6))
            ws_dat.cell(fila, 8, round(r["qn_t"], 4))
            ws_dat.cell(fila, 9, round(r["ge_t"], 4))
            ws_dat.cell(fila, 10, round(r["delta"], 6))
            ws_dat.cell(fila, 11, "Sí" if r["misma"] else "No")
            ws_dat.cell(fila, 12, r["qn_part_txt"]).alignment = _PART_WRAP
            ws_dat.cell(fila, 13, r["ge_part_txt"]).alignment = _PART_WRAP
            fila += 1
        data_end = fila - 1
        rangos[k] = (header_row, data_start, data_end)

        # Tabla de Excel sobre el bloque → da botones de filtro por columna (incl. N).
        # Se permiten varias tablas por hoja; al filtrar N, Excel oculta las filas y
        # las gráficas (que solo grafican celdas visibles) se actualizan solas.
        ref = f"A{header_row}:{get_column_letter(NCOL)}{data_end}"
        tabla = Table(displayName=f"Datos_k{k}", ref=ref)
        tabla.tableStyleInfo = TableStyleInfo(
            name="TableStyleLight9", showRowStripes=True, showColumnStripes=False)
        ws_dat.add_table(tabla)

        fila += 2  # separación entre bloques k

    # Anchos cómodos para las columnas de texto largo (particiones, alcance/mecanismo).
    ws_dat.column_dimensions["C"].width = 16
    ws_dat.column_dimensions["D"].width = 16
    ws_dat.column_dimensions[get_column_letter(12)].width = 22
    ws_dat.column_dimensions[get_column_letter(13)].width = 22

    # Tabla resumen "¿misma partición?" por k.
    resumen_hdr = fila
    ws_dat.cell(resumen_hdr, 1, "Resumen ¿misma partición?").font = _BOLD
    fila += 1
    _escribir_encabezado(ws_dat, fila, ["k", "Iguales", "Distintas", "% iguales"])
    resumen_data = fila + 1
    fila += 1
    for k in KS:
        regs_k = [r for r in registros if r["k"] == k]
        if not regs_k:
            continue
        iguales = sum(1 for r in regs_k if r["misma"])
        distintas = len(regs_k) - iguales
        ws_dat.cell(fila, 1, k)
        ws_dat.cell(fila, 2, iguales)
        ws_dat.cell(fila, 3, distintas)
        ws_dat.cell(fila, 4, round(100.0 * iguales / len(regs_k), 1))
        fila += 1
    resumen_end = fila - 1

    # ── Gráficas ────────────────────────────────────────────────────────────
    ws_g["A1"] = "Comparativas QNodes vs Geométrica — Suite 2026-1"
    ws_g["A1"].font = Font(bold=True, size=14)
    ws_g["A2"] = ("(a) Tiempo vs tamaño de subsistema · (b) Pérdida por subsistema "
                  "· (c) Δ pérdida con QNodes como referencia")
    ws_g["A2"].font = Font(italic=True)

    fila_ancla = 4
    for k in KS:
        if k not in rangos:
            continue
        hr, ds, de = rangos[k]

        # (a) Tiempo vs tamaño — scatter, X=tamaño, series QNodes/Geom.
        sc = ScatterChart()
        sc.title = f"(a) Tiempo vs tamaño de subsistema — k={k}"
        sc.x_axis.title = "Tamaño del subsistema (nodos del alcance)"
        sc.y_axis.title = "Tiempo por prueba: prep+búsqueda (s)"
        sc.height = 9; sc.width = 17
        xref = Reference(ws_dat, min_col=5, min_row=ds, max_row=de)
        for col, nombre, color in ((8, "QNodes", "1F77B4"), (9, "Geométrica", "D62728")):
            yref = Reference(ws_dat, min_col=col, min_row=ds, max_row=de)
            serie = Series(yref, xref, title=nombre)
            serie.marker = Marker(symbol="circle", size=5)
            serie.graphicalProperties.line.noFill = True  # solo puntos
            sc.series.append(serie)
        ws_g.add_chart(sc, f"A{fila_ancla}")

        # (b) Pérdida por subsistema — líneas QNodes vs Geom.
        lc = LineChart()
        lc.title = f"(b) Pérdida Φ por subsistema — k={k}"
        lc.x_axis.title = "Subsistema (ordenado por tamaño)"
        lc.y_axis.title = "Pérdida Φ (EMD)"
        lc.height = 9; lc.width = 17
        datos_b = Reference(ws_dat, min_col=6, max_col=7, min_row=hr, max_row=de)
        cats = Reference(ws_dat, min_col=1, min_row=ds, max_row=de)
        lc.add_data(datos_b, titles_from_data=True)
        lc.set_categories(cats)
        ws_g.add_chart(lc, f"K{fila_ancla}")

        # (c) Δ pérdida (Geom − QNodes) — barras alrededor de 0.
        bc = BarChart()
        bc.type = "col"
        bc.title = f"(c) Δ pérdida (Geom − QNodes) — k={k}"
        bc.x_axis.title = "Subsistema (QNodes = referencia / 0)"
        bc.y_axis.title = "Δ Φ = Φ_Geom − Φ_QNodes"
        bc.height = 9; bc.width = 17
        datos_c = Reference(ws_dat, min_col=10, min_row=hr, max_row=de)
        bc.add_data(datos_c, titles_from_data=True)
        bc.set_categories(cats)
        ws_g.add_chart(bc, f"U{fila_ancla}")

        fila_ancla += 19

    # Resumen "¿misma partición?" — barras por k.
    if resumen_end >= resumen_data:
        rc = BarChart()
        rc.type = "col"
        rc.title = "(b) ¿Se halla la misma k-partición? (conteo por k)"
        rc.x_axis.title = "k"
        rc.y_axis.title = "Nº de subsistemas"
        rc.height = 9; rc.width = 17
        datos_r = Reference(ws_dat, min_col=2, max_col=3,
                            min_row=resumen_data - 1, max_row=resumen_end)
        cats_r = Reference(ws_dat, min_col=1, min_row=resumen_data, max_row=resumen_end)
        rc.add_data(datos_r, titles_from_data=True)
        rc.set_categories(cats_r)
        ws_g.add_chart(rc, f"A{fila_ancla}")

    log(f"  [gráficos] generadas {len(rangos)} familias × k en hoja 'Graficos' "
        f"({len(registros)} registros).")
    return True
