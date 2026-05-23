# -*- coding: utf-8 -*-
"""
Script de ejecución interactivo para KGeoMIP — k-Partición de Mínima Información (ÓPTIMO GLOBAL)

Permite al usuario seleccionar el dataset a través de una ventana de exploración
de archivos nativa y declara su estado inicial. Realiza la evaluación sobre todos los
valores posibles de particiones (k=2..N) para extraer la partición de mínima pérdida.
"""

import sys
import random
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import numpy as np
import json
import re
from datetime import datetime

from src.models.base.application import aplicacion
from src.controllers.manager import Manager
from src.controllers.strategies.kgeomip import KGeoMIP
from src.controllers.strategies.kgeomip import _serializar_particion
from src.lazy_tpm import cargar_tpm

GEOMIP_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = GEOMIP_ROOT / "data" / "samples"


def sep(titulo: str):
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print('='*60)


def main():
    aplicacion.profiler_habilitado = False

    sep(" Ejecución GUI/Interactiva de KGeoMIP Óptimo Global")

    # ── 1. Selección del archivo (TPM) mediante Ventana ───────────────────
    print(" Abriendo explorador de archivos en la carpeta de datasets...")
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        ruta_archivo_str = filedialog.askopenfilename(
            initialdir=str(SAMPLES_DIR),
            title="Seleccionar matriz de probabilidad (TPM) CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        root.destroy()
    except Exception as e:
        print(f" No se pudo cargar el explorador: {e}")
        archivo = input(" Ingrese manualmente el nombre del archivo (ej. N3A.csv): ").strip()
        ruta_archivo_str = str(SAMPLES_DIR / archivo)

    if not ruta_archivo_str:
        print(" Ejecución cancelada: No se seleccionó ningún archivo.")
        sys.exit(1)

    ruta_archivo = Path(ruta_archivo_str)
    if not ruta_archivo.exists():
        print(f" Error: El archivo {ruta_archivo} no fue encontrado.")
        sys.exit(1)

    try:
        tpm = cargar_tpm(ruta_archivo)
        n_nodos = tpm.shape[1] if len(tpm.shape) > 1 else int(np.log2(tpm.shape[0]))
    except Exception as e:
        print(f" Error: Ocurrió un error al intentar leer el fichero TPM seleccionado: {e}")
        sys.exit(1)

    # ── 2. Selección del Sistema Candidato ────────────────────────────────
    print(f"\n La red '{ruta_archivo.name}' consta de {n_nodos} variables lógicas asociadas.")
    candidato_input = input(
        f" Ingrese el SISTEMA CANDIDATO en binario (longitud: {n_nodos} bits)\n"
        f"   O presione [ENTER] para tomar TODO el sistema: "
    ).strip()

    if not candidato_input:
        bits_candidato = "1" * n_nodos
        print(f" Seleccionado el sistema completo: {bits_candidato}")
    else:
        if set(candidato_input).issubset({"0", "1"}) and len(candidato_input) == n_nodos:
            bits_candidato = candidato_input
        else:
            print(f" Error: La máscara debe ser cadena de unos y ceros y coincidir con longitud {n_nodos}.")
            sys.exit(1)

    n_candidatos = bits_candidato.count("1")
    if n_candidatos < 2:
        print(" Error: El sistema candidato debe tener al menos 2 variables (1s) para ser particionado.")
        sys.exit(1)

    # ── 3. Selección del Estado Inicial ───────────────────────────────────
    estado_input = input(
        f"\n Ingrese el ESTADO INICIAL en binario (longitud: {n_candidatos} bits)\n"
        f"   O presione [ENTER] para uno RANDOM: "
    ).strip()

    if not estado_input:
        estado_candidato = "".join(str(random.randint(0, 1)) for _ in range(n_candidatos))
        print(f" Estado aleatorio auto-generado: {estado_candidato}")
    else:
        if set(estado_input).issubset({"0", "1"}) and len(estado_input) == n_candidatos:
            estado_candidato = estado_input
        else:
            print(" Error: El estado inicial debe ser estrictamente una cadena de unos y ceros.")
            sys.exit(1)

    # Reconstruir estado completo rellenando con 0s
    estado_completo_list = ["0"] * n_nodos
    idx_cand = 0
    for i, bit in enumerate(bits_candidato):
        if bit == "1":
            estado_completo_list[i] = estado_candidato[idx_cand]
            idx_cand += 1
    estado = "".join(estado_completo_list)

    # ── 4. Selección del Alcance (Purview t+1) ────────────────────────────
    print(f"\n Ingrese el ALCANCE (Purview t+1) en binario (longitud: {n_nodos} bits)")
    print(f"   Referencia — Candidato activo: {bits_candidato}")
    print(f"   1 = observar ese nodo en t+1  |  0 = ignorarlo (marginalizarlo)")
    alcance_input = input(
        f"   O presione [ENTER] para usar TODO el candidato como alcance: "
    ).strip()

    if not alcance_input:
        bits_alcance = bits_candidato
        print(f" Alcance: {bits_alcance} (igual al candidato)")
    else:
        if set(alcance_input).issubset({"0", "1"}) and len(alcance_input) == n_nodos:
            # Validar que el alcance no active nodos fuera del candidato
            invalidos = [
                i for i, (a, c) in enumerate(zip(alcance_input, bits_candidato))
                if a == "1" and c == "0"
            ]
            if invalidos:
                print(f" Error: El alcance activa nodos {invalidos} que NO están en el candidato.")
                sys.exit(1)
            bits_alcance = alcance_input
        else:
            print(f" Error: Debe ser una cadena de 0s y 1s de longitud {n_nodos}.")
            sys.exit(1)

    # ── 5. Selección del Mecanismo (t) ────────────────────────────────────
    print(f"\n Ingrese el MECANISMO (t) en binario (longitud: {n_nodos} bits)")
    print(f"   Referencia — Candidato activo: {bits_candidato}")
    print(f"   1 = fijar ese nodo en su estado inicial  |  0 = marginalizarlo")
    mecanismo_input = input(
        f"   O presione [ENTER] para usar TODO el candidato como mecanismo: "
    ).strip()

    if not mecanismo_input:
        bits_mecanismo = bits_candidato
        print(f" Mecanismo: {bits_mecanismo} (igual al candidato)")
    else:
        if set(mecanismo_input).issubset({"0", "1"}) and len(mecanismo_input) == n_nodos:
            # Validar que el mecanismo no active nodos fuera del candidato
            invalidos = [
                i for i, (m, c) in enumerate(zip(mecanismo_input, bits_candidato))
                if m == "1" and c == "0"
            ]
            if invalidos:
                print(f" Error: El mecanismo activa nodos {invalidos} que NO están en el candidato.")
                sys.exit(1)
            bits_mecanismo = mecanismo_input
        else:
            print(f" Error: Debe ser una cadena de 0s y 1s de longitud {n_nodos}.")
            sys.exit(1)

    # ── 6. Selección de K (número de particiones) ───────────────────────
    print(f"\n Número máximo de particiones posibles: {n_candidatos}")
    k_input = input(
        f" Ingrese el valor de K (particiones a evaluar) [2-{n_candidatos}]\n"
        f"   O presione [ENTER] para evaluar TODAS las k posibles: "
    ).strip()

    if not k_input:
        k_a_evaluar = None
        print(f" Se evaluarán TODAS las k posibles (2 a {min(5, n_candidatos)})")
    else:
        try:
            k_val = int(k_input)
            if k_val < 2 or k_val > n_candidatos:
                print(f" Error: K debe estar entre 2 y {n_candidatos}.")
                sys.exit(1)
            k_a_evaluar = k_val
            print(f" Se evaluará K = {k_a_evaluar}")
        except ValueError:
            print(" Error: K debe ser un número entero.")
            sys.exit(1)

    # ── Resumen antes de ejecutar ─────────────────────────────────────────
    sep(" Iniciando Algoritmo — KGeoMIP Partición Óptima Global")
    print(f"   Dataset Cargado : {ruta_archivo.name}")
    print(f"   Sist. Candidato : {bits_candidato} ({n_candidatos} nodos)")
    print(f"   Estado Central  : {estado}")
    print(f"   Alcance (t+1)   : {bits_alcance}")
    print(f"   Mecanismo (t)   : {bits_mecanismo}")
    if k_a_evaluar:
        print(f"   K a evaluar     : {k_a_evaluar}\n")
    else:
        print(f"   K a evaluar     : Todas (2..{min(5, n_candidatos)})\n")

    # ── 7. Instanciar y ejecutar KGeoMIP ──────────────────────────────────
    manager_inst = Manager(estado)
    kgeomip_inst = KGeoMIP(manager_inst)
    solucion_geomip = kgeomip_inst.aplicar_estrategia(
        condicion=bits_candidato,
        alcance=bits_alcance,
        mecanismo=bits_mecanismo,
        tpm=tpm,
        k=k_a_evaluar
    )

    print(solucion_geomip)

    # ── 7. Guardar los resultados en la carpeta 'results' ─────────────────
    results_dir = GEOMIP_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    estrateg_str = getattr(solucion_geomip, "estrategia", "kgeomip")
    match_k = re.search(r'K=(\d+)', estrateg_str)
    k_val_str = match_k.group(1) if match_k else "optimal"

    fecha_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nombre_archivo = f"{fecha_actual}_resultado_{ruta_archivo.stem}_k{k_val_str}_global.json"
    ruta_salida = results_dir / nombre_archivo

    res_data = {
        "dataset": ruta_archivo.name,
        "estado_inicial": estado,
        "sistema_candidato": bits_candidato,
        "alcance": bits_alcance,
        "mecanismo": bits_mecanismo,
        "estrategia": estrateg_str,
        "perdida_phi": float(solucion_geomip.perdida),
        "fundamento_eleccion": (
            "Se seleccionó esta k-partición porque presenta la pérdida de información "
            "(EMD) mínima global de todo el abanico evaluado."
        ),
        "distribucion_subsistema": (
            solucion_geomip.distribucion_subsistema.tolist()
            if hasattr(solucion_geomip.distribucion_subsistema, 'tolist')
            else solucion_geomip.distribucion_subsistema
        ),
        "distribucion_particion": (
            solucion_geomip.distribucion_particion.tolist()
            if hasattr(solucion_geomip.distribucion_particion, 'tolist')
            else solucion_geomip.distribucion_particion
        ),
        "particion": getattr(kgeomip_inst, "particion_optima_serializada", str(solucion_geomip.particion)),

        "historico_comparaciones": getattr(kgeomip_inst, "historico_particiones", []),
        "tiempo_total": float(
            getattr(solucion_geomip, "tiempo_ejecucion",
                    getattr(solucion_geomip, "tiempo_total", 0.0))
        )
    }

    with open(ruta_salida, 'w', encoding='utf-8') as f:
        json.dump(res_data, f, indent=4, ensure_ascii=False)

    print(f"\n ✓ Resultados guardados exitosamente en: {ruta_salida.relative_to(GEOMIP_ROOT)}")


if __name__ == '__main__':
    main()
