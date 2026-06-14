# -*- coding: utf-8 -*-
"""
Comparación batch: Fuerza Bruta k-MIP (N ≤ 6) vs KQNodes vs KGeoMIP.

Procesa ÚNICAMENTE las muestras pequeñas de Brute_Force/samples_force/ (N ≤ 6),
que es donde la fuerza bruta exhaustiva es tratable. Los archivos con N > 6 se
omiten: la fuerza bruta no se compara contra samples_binary ni ningún N grande.

Parámetros de análisis por CSV:
  estado    = "0" * N   (todos ceros)
  candidato = alcance = mecanismo = "1" * N   (sistema completo)

BruteForce y KQNodes corren en el proceso actual (KQNodes/src en sys.path).
KGeoMIP corre en subproceso aislado vía data/_worker_motor.py para evitar
el conflicto de nombres de paquete 'src' entre KGeoMIP y KQNodes.

Salida: Brute_Force/results/comparacion_fuerza_bruta_<fecha>.xlsx
  Hoja "BF vs KQNodes vs KGeoMIP"  — comparación por k para N ≤ 6
  Hoja "KQNodes vs KGeoMIP"        — comparación QN vs GEO para esos mismos N ≤ 6

Uso:
    .venv/Scripts/python Brute_Force/comparar_fuerza_bruta.py
"""

import contextlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Rutas base ────────────────────────────────────────────────────────────────
BRUTE_FORCE_ROOT = Path(__file__).resolve().parent   # AYDA_2026_1/Brute_Force
REPO_ROOT   = BRUTE_FORCE_ROOT.parent
QNODES_ROOT = REPO_ROOT / "KQNodes"
GEOMIP_ROOT = REPO_ROOT / "KGeoMIP"
DATA_ROOT   = REPO_ROOT / "data"
WORKER      = DATA_ROOT / "_worker_motor.py"
PYTHON      = sys.executable
RESULTS_DIR = BRUTE_FORCE_ROOT / "results"
SAMPLES_FRC = BRUTE_FORCE_ROOT / "samples_force"

RESULT_SENTINEL  = "@@RESULT@@"
TOLERANCIA       = 1e-9   # para considerar dos Phi "idénticos"
TOLERANCIA_COTA  = 1e-6   # margen para fallo de cota (evita falsos positivos por float32 vs float64)
N_MAX_BF         = 6      # límite práctico de la fuerza bruta

# samples_force/ → SOLO N pequeños (N ≤ 6) — BF + KQNodes + KGeoMIP.
# La fuerza bruta NO se ejecuta sobre samples_binary ni N grandes (intratable).

# ── Importar KQNodes en proceso ───────────────────────────────────────────────
if str(QNODES_ROOT) not in sys.path:
    sys.path.insert(0, str(QNODES_ROOT))

from src.models.base.application import aplicacion       # noqa: E402
from src.middlewares.profile import gestor_perfilado     # noqa: E402
from src.strategies.force import BruteForceKMIP          # noqa: E402
from src.strategies.q_nodes import QNodes                # noqa: E402

aplicacion.desactivar_profiling()
gestor_perfilado.enabled = False


# ── Supresión de output de los motores ────────────────────────────────────────
@contextlib.contextmanager
def _silenciar():
    devnull = open(os.devnull, "w", encoding="utf-8")
    nivel_previo = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
    finally:
        devnull.close()
        logging.disable(nivel_previo)


# ── Utilidades ────────────────────────────────────────────────────────────────
def sep(titulo: str) -> None:
    print(f"\n{'═' * 68}")
    print(f"  {titulo}")
    print("═" * 68)


def cargar_tpm(csv_path: Path) -> tuple:
    with open(csv_path, "r", encoding="utf-8") as fh:
        tpm = np.genfromtxt(fh, delimiter=",")
    return tpm, int(tpm.shape[1])


def comparar_phi(phi_bf: float, phi_heur: float) -> tuple:
    """Devuelve (igual, cota_ok, err_rel) comparando heurística contra bruta.

    cota_ok usa TOLERANCIA_COTA (1e-6) para evitar falsos positivos por
    diferencias de precisión float32/float64 en la acumulación de la EMD.
    """
    igual    = abs(phi_bf - phi_heur) <= TOLERANCIA
    cota_ok  = phi_heur >= phi_bf - TOLERANCIA_COTA
    err_rel  = abs(phi_bf - phi_heur) / phi_bf if phi_bf > TOLERANCIA else 0.0
    return igual, cota_ok, round(err_rel, 9)


# ── Motores ───────────────────────────────────────────────────────────────────

def correr_bruta(tpm: np.ndarray, n: int, estado: str,
                 k: Optional[int], permitir_vacio: bool) -> dict:
    """Fuerza bruta exhaustiva k-MIP (BruteForceKMIP, solo N ≤ 6)."""
    candidato = alcance = mecanismo = "1" * n
    try:
        t0 = time.time()
        with _silenciar():
            bf  = BruteForceKMIP(tpm)
            sol = bf.aplicar_estrategia(
                estado_inicial=estado,
                condicion=candidato,
                alcance=alcance,
                mecanismo=mecanismo,
                k=k,
                permitir_presente_vacio=permitir_vacio,
                umbral_configuraciones=500_000,
                max_futuros=N_MAX_BF,
            )
        return {
            "ok": True,
            "perdida": float(sol.perdida),
            "particion": str(sol.particion),
            "tiempo": float(getattr(sol, "tiempo_total", time.time() - t0)),
            "optimos_por_k": {kk: round(float(v[0]), 9)
                               for kk, v in bf.optimos_por_k.items()},
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def correr_qnodes(tpm: np.ndarray, n: int, estado: str,
                  k: Optional[int], permitir_vacio: bool) -> dict:
    """KQNodes en proceso."""
    candidato = alcance = mecanismo = "1" * n
    try:
        t0 = time.time()
        with _silenciar():
            q   = QNodes(tpm)
            sol = q.aplicar_estrategia(
                estado_inicial=estado,
                condicion=candidato,
                alcance=alcance,
                mecanismo=mecanismo,
                k=k,
                permitir_presente_vacio=permitir_vacio,
            )
        return {
            "ok": True,
            "perdida": float(sol.perdida),
            "particion": str(sol.particion),
            "tiempo": float(getattr(sol, "tiempo_total", time.time() - t0)),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def correr_geomip(csv_path: Path, n: int, estado: str,
                  k: Optional[int], permitir_vacio: bool) -> dict:
    """KGeoMIP en subproceso aislado (evita conflicto de paquetes src)."""
    candidato = alcance = mecanismo = "1" * n
    cfg = {
        "engine": "geomip",
        "root": str(GEOMIP_ROOT),
        "tpm": str(csv_path),
        "estado": estado,
        "candidato": candidato,
        "permitir_presente_vacio": permitir_vacio,
        "k": k,
        "tests": [{"row": 1, "alcance": alcance, "mecanismo": mecanismo}],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json",
                                     delete=False, encoding="utf-8") as tf:
        json.dump(cfg, tf, ensure_ascii=False)
        lote = tf.name
    try:
        proc = subprocess.Popen(
            [PYTHON, str(WORKER), lote],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
        )
        resultado: dict = {"ok": False, "error": "sin respuesta del worker"}
        for linea in proc.stdout:
            if not linea.startswith(RESULT_SENTINEL):
                continue
            payload = json.loads(linea[len(RESULT_SENTINEL):])
            if "fatal" in payload:
                return {"ok": False, "error": payload["fatal"]}
            if "ready" in payload:
                continue
            resultado = payload
        proc.wait()
    finally:
        try:
            os.unlink(lote)
        except OSError:
            pass

    if resultado.get("ok"):
        resultado["tiempo"] = float(resultado.get("tiempo", 0.0))
    return resultado


# ── Comparación QN vs GEO (sin fuerza bruta) ─────────────────────────────────

def procesar_solo_comp(csv_path: Path, tpm: np.ndarray, n: int,
                       nombre: str, estado: str, k: Optional[int],
                       permitir_vacio: bool, filas_comp: list) -> None:
    k_lbl = "libre" if k is None else str(k)
    print(f"    QNodes  k={k_lbl}...", end=" ", flush=True)
    qn_res = correr_qnodes(tpm, n, estado, k, permitir_vacio)
    print("OK" if qn_res.get("ok") else "ERR")

    print(f"    KGeoMIP k={k_lbl}...", end=" ", flush=True)
    geo_res = correr_geomip(csv_path, n, estado, k, permitir_vacio)
    print("OK" if geo_res.get("ok") else "ERR")

    phi_qn  = float(qn_res["perdida"])  if qn_res.get("ok")  else None
    phi_geo = float(geo_res["perdida"]) if geo_res.get("ok") else None
    t_qn    = round(float(qn_res.get("tiempo", 0)), 4)  if qn_res.get("ok")  else None
    t_geo   = round(float(geo_res.get("tiempo", 0)), 4) if geo_res.get("ok") else None

    if phi_qn is not None and phi_geo is not None:
        diff_abs = abs(phi_qn - phi_geo)
        mn = min(phi_qn, phi_geo)
        err_rel  = round(diff_abs / mn, 9) if mn > TOLERANCIA else 0.0
        coinciden = diff_abs <= TOLERANCIA
    else:
        diff_abs = err_rel = None
        coinciden = False

    filas_comp.append({
        "archivo": nombre, "n": n, "k": k if k is not None else "libre",
        "phi_qn":  round(phi_qn, 9)  if phi_qn  is not None else "ERROR",
        "phi_geo": round(phi_geo, 9) if phi_geo is not None else "ERROR",
        "coinciden": coinciden,
        "diff_abs": round(diff_abs, 9) if diff_abs is not None else None,
        "err_rel":  err_rel,
        "t_qn": t_qn, "t_geo": t_geo,
        "nota_qn":  "" if qn_res.get("ok")  else qn_res.get("error", ""),
        "nota_geo": "" if geo_res.get("ok") else geo_res.get("error", ""),
    })


# ── Exportación XLSX ──────────────────────────────────────────────────────────

def _escribir_cabeceras(ws, cabeceras: list) -> None:
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    hf   = Font(bold=True, color="FFFFFF")
    hfll = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    ctr  = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for col, titulo in enumerate(cabeceras, 1):
        c = ws.cell(row=1, column=col, value=titulo)
        c.font, c.fill, c.alignment = hf, hfll, ctr
        ws.column_dimensions[get_column_letter(col)].width = max(16, len(titulo) + 2)
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 30


def exportar_xlsx(filas_bf: list, filas_comp: list) -> Path:
    import openpyxl
    from openpyxl.styles import PatternFill, Alignment

    verde   = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    naranja = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    rojo    = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    ctr     = Alignment(horizontal="center", vertical="center")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ruta  = RESULTS_DIR / f"comparacion_fuerza_bruta_{fecha}.xlsx"
    wb    = openpyxl.Workbook()

    # ── Hoja 1: BF vs KQNodes vs KGeoMIP ─────────────────────────────────────
    ws1 = wb.active
    ws1.title = "BF vs KQNodes vs KGeoMIP"
    hdrs1 = [
        "Archivo", "N", "k",
        "Φ_FuerzaBruta", "Φ_KQNodes", "Φ_KGeoMIP",
        "QN_≥_BF", "GEO_≥_BF",
        "err_rel_QN", "err_rel_GEO",
        "Φ_igual_QN", "Φ_igual_GEO",
        "t_QNodes(s)", "t_KGeoMIP(s)",
        "Nota_QN", "Nota_GEO",
    ]
    _escribir_cabeceras(ws1, hdrs1)

    for ri, f in enumerate(filas_bf, 2):
        row = [
            f.get("archivo"), f.get("n"), f.get("k"),
            f.get("phi_bf"), f.get("phi_qn"), f.get("phi_geo"),
            f.get("qn_cota_ok"), f.get("geo_cota_ok"),
            f.get("err_rel_qn"), f.get("err_rel_geo"),
            f.get("phi_igual_qn"), f.get("phi_igual_geo"),
            f.get("t_qn"), f.get("t_geo"),
            f.get("nota_qn", ""), f.get("nota_geo", ""),
        ]
        if f.get("qn_cota_ok") is False or f.get("geo_cota_ok") is False:
            fill = rojo
        elif f.get("phi_igual_qn") and f.get("phi_igual_geo"):
            fill = verde
        else:
            fill = naranja
        for ci, val in enumerate(row, 1):
            c = ws1.cell(row=ri, column=ci, value=val)
            c.alignment = ctr
            c.fill = fill

    # ── Hoja 2: KQNodes vs KGeoMIP ───────────────────────────────────────────
    ws2 = wb.create_sheet("KQNodes vs KGeoMIP")
    hdrs2 = [
        "Archivo", "N", "k",
        "Φ_KQNodes", "Φ_KGeoMIP",
        "Coinciden", "diff_abs", "err_rel",
        "t_QNodes(s)", "t_KGeoMIP(s)",
        "Nota_QN", "Nota_GEO",
    ]
    _escribir_cabeceras(ws2, hdrs2)

    for ri, f in enumerate(filas_comp, 2):
        row = [
            f.get("archivo"), f.get("n"), f.get("k", "libre"),
            f.get("phi_qn"), f.get("phi_geo"),
            f.get("coinciden"),
            f.get("diff_abs"), f.get("err_rel"),
            f.get("t_qn"), f.get("t_geo"),
            f.get("nota_qn", ""), f.get("nota_geo", ""),
        ]
        if f.get("coinciden"):
            fill = verde
        elif (f.get("err_rel") or 0) < 0.01:
            fill = naranja
        else:
            fill = rojo
        for ci, val in enumerate(row, 1):
            c = ws2.cell(row=ri, column=ci, value=val)
            c.alignment = ctr
            c.fill = fill

    wb.save(ruta)
    return ruta


# ── Punto de entrada ──────────────────────────────────────────────────────────

def main() -> None:
    sep("Comparación: Fuerza Bruta vs KQNodes vs KGeoMIP")

    # samples_force/ = N pequeños (N≤6) → comparación completa con fuerza bruta.
    archivos_frc = sorted(SAMPLES_FRC.glob("N*.csv"))

    if not archivos_frc:
        print(f" No se encontraron CSV en {SAMPLES_FRC}.")
        sys.exit(1)

    print(f"\n  samples_force  : {len(archivos_frc)} archivo(s)  [BF + KQNodes + KGeoMIP, N ≤ {N_MAX_BF}]")
    for f in archivos_frc:
        print(f"    {f.name}")

    opcion = input(
        "\n ¿Permitir mecanismo vacío (∅) en las partes?\n"
        "   1. Sí (recomendado) — lower bound real para la bruta\n"
        "   2. No\n"
        " Seleccione (1/2) o [ENTER] para Sí: "
    ).strip()
    permitir_vacio = (opcion != "2")
    print(f" permitir_presente_vacio = {permitir_vacio}")

    filas_bf: list   = []
    filas_comp: list = []

    # ── samples_force: BF + QNodes + KGeoMIP por cada k (N ≤ 6) ─────────────
    sep("samples_force — BruteForce + KQNodes + KGeoMIP (N ≤ 6)")

    for csv_path in archivos_frc:
        tpm, n = cargar_tpm(csv_path)
        nombre = csv_path.name
        estado = "0" * n
        print(f"\n  [{nombre}]  N={n}")

        if n > N_MAX_BF:
            print(f"    N={n} > {N_MAX_BF} — OMITIDO (la fuerza bruta solo compara N ≤ {N_MAX_BF}).")
            continue

        # BF una sola vez con k=None → obtiene todos los k de golpe
        print(f"    BruteForce k=None...", end=" ", flush=True)
        bf_res = correr_bruta(tpm, n, estado, None, permitir_vacio)
        print("OK" if bf_res.get("ok") else f"ERROR: {bf_res.get('error','?')}")

        if not bf_res.get("ok") or not bf_res.get("optimos_por_k"):
            print(f"    ⚠ BruteForce falló — solo QN vs GEO")
            procesar_solo_comp(csv_path, tpm, n, nombre, estado,
                               None, permitir_vacio, filas_comp)
            continue

        optimos_por_k = bf_res["optimos_por_k"]

        for k_val in sorted(optimos_por_k.keys()):
            phi_bf = optimos_por_k[k_val]

            print(f"    QNodes  k={k_val}...", end=" ", flush=True)
            qn_res = correr_qnodes(tpm, n, estado, k_val, permitir_vacio)
            print("OK" if qn_res.get("ok") else "ERR")

            print(f"    KGeoMIP k={k_val}...", end=" ", flush=True)
            geo_res = correr_geomip(csv_path, n, estado, k_val, permitir_vacio)
            print("OK" if geo_res.get("ok") else "ERR")

            phi_qn  = float(qn_res["perdida"])  if qn_res.get("ok")  else None
            phi_geo = float(geo_res["perdida"]) if geo_res.get("ok") else None
            t_qn    = round(float(qn_res.get("tiempo", 0)), 4)  if phi_qn  is not None else None
            t_geo   = round(float(geo_res.get("tiempo", 0)), 4) if phi_geo is not None else None

            qn_igual = qn_cota_ok = err_qn = None
            geo_igual = geo_cota_ok = err_geo = None
            if phi_qn is not None:
                qn_igual, qn_cota_ok, err_qn = comparar_phi(phi_bf, phi_qn)
            if phi_geo is not None:
                geo_igual, geo_cota_ok, err_geo = comparar_phi(phi_bf, phi_geo)

            filas_bf.append({
                "archivo": nombre, "n": n, "k": k_val,
                "phi_bf":  round(phi_bf, 9),
                "phi_qn":  round(phi_qn, 9)  if phi_qn  is not None else "ERROR",
                "phi_geo": round(phi_geo, 9) if phi_geo is not None else "ERROR",
                "qn_cota_ok": qn_cota_ok, "geo_cota_ok": geo_cota_ok,
                "err_rel_qn": err_qn, "err_rel_geo": err_geo,
                "phi_igual_qn": qn_igual, "phi_igual_geo": geo_igual,
                "t_qn": t_qn, "t_geo": t_geo,
                "nota_qn":  "" if qn_res.get("ok")  else qn_res.get("error", ""),
                "nota_geo": "" if geo_res.get("ok") else geo_res.get("error", ""),
            })

            # También en la hoja comparativa
            if phi_qn is not None and phi_geo is not None:
                diff_abs = abs(phi_qn - phi_geo)
                mn = min(phi_qn, phi_geo)
                filas_comp.append({
                    "archivo": nombre, "n": n, "k": k_val,
                    "phi_qn": round(phi_qn, 9), "phi_geo": round(phi_geo, 9),
                    "coinciden": diff_abs <= TOLERANCIA,
                    "diff_abs": round(diff_abs, 9),
                    "err_rel":  round(diff_abs / mn, 9) if mn > TOLERANCIA else 0.0,
                    "t_qn": t_qn, "t_geo": t_geo,
                    "nota_qn": "", "nota_geo": "",
                })

    # ── Resumen por consola ───────────────────────────────────────────────────
    sep("Resumen")
    if filas_bf:
        total = len(filas_bf)
        iguales_qn  = sum(1 for f in filas_bf if f.get("phi_igual_qn"))
        iguales_geo = sum(1 for f in filas_bf if f.get("phi_igual_geo"))
        fallos_qn   = sum(1 for f in filas_bf if f.get("qn_cota_ok") is False)
        fallos_geo  = sum(1 for f in filas_bf if f.get("geo_cota_ok") is False)
        print(f"  BF vs Frameworks — {total} combinaciones (archivo × k):")
        print(f"    KQNodes  — Φ exacto: {iguales_qn}/{total}   Fallos cota: {fallos_qn}")
        print(f"    KGeoMIP  — Φ exacto: {iguales_geo}/{total}   Fallos cota: {fallos_geo}")
        if fallos_qn or fallos_geo:
            print("    ⚠  Hay fallos de cota inferior — revisar implementación.")
    else:
        print("  No se generaron filas de comparación BF vs Frameworks.")

    if filas_comp:
        total_c = len(filas_comp)
        coinciden = sum(1 for f in filas_comp if f.get("coinciden"))
        print(f"\n  KQNodes vs KGeoMIP — {total_c} combinaciones:")
        print(f"    Coinciden exactamente: {coinciden}/{total_c}")

    # ── Exportar XLSX ─────────────────────────────────────────────────────────
    sep("Exportando XLSX")
    try:
        ruta = exportar_xlsx(filas_bf, filas_comp)
        print(f"  ✓ Guardado en: {ruta}")
    except Exception as exc:
        print(f"  ✗ Error al guardar XLSX: {exc}")


if __name__ == "__main__":
    main()
