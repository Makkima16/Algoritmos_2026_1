"""
Script de ejecución interactivo para KGeoMIP — k-Partición de Mínima Información.

Permite al usuario seleccionar el dataset a través de una ventana de exploración
de archivos nativa (apuntando a la carpeta de dependencias `data/samples`), 
declarar o generar aleatoriamente su estado inicial (según la longitud de su red), 
y el valor de la partición 'k' a efectuar.

Ejemplo de uso:x
    python exec_kgeomip.py
"""

import sys
import random
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import numpy as np

from src.models.base.application import aplicacion
from src.controllers.manager import Manager
from src.controllers.strategies.kgeomip import KGeoMIP

GEOMIP_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = GEOMIP_ROOT / "data" / "samples"


def cargar_tpm(ruta: Path) -> np.ndarray:
    return np.genfromtxt(ruta, delimiter=",")


def sep(titulo: str):
    print(f"\n{'═'*60}")
    print(f"  {titulo}")
    print('═'*60)


def main():
    aplicacion.profiler_habilitado = False
    
    sep("🧠 Ejecución GUI/Interactiva de KGeoMIP")

    # 1. Selección del archivo (TPM) mediante Ventana
    print("📂 Abriendo explorador de archivos en la carpeta de datasets...")
    try:
        # Ocultar la ventana de fondo de tkinter
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True) # Asegurar que la ventana salga por delante
        ruta_archivo_str = filedialog.askopenfilename(
            initialdir=str(SAMPLES_DIR),
            title="Seleccionar matriz de probabilidad (TPM) CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        root.destroy()
    except Exception as e:
        print(f"⚠️ No se pudo cargar el explorador de archivos: {e}")
        # Fallback en consola en caso de fallo gráfico
        archivo = input("📝 Ingrese manualmente el nombre del archivo (ej. N3A.csv): ").strip()
        ruta_archivo_str = str(SAMPLES_DIR / archivo)

    if not ruta_archivo_str:
        print("❌ Ejecución cancelada: No se seleccionó ningún archivo.")
        sys.exit(1)

    ruta_archivo = Path(ruta_archivo_str)
    if not ruta_archivo.exists():
        print(f"❌ Error: El archivo {ruta_archivo} no fue encontrado.")
        sys.exit(1)

    try:
        tpm = cargar_tpm(ruta_archivo)
        n_nodos = tpm.shape[1] if len(tpm.shape) > 1 else int(np.log2(tpm.shape[0]))
    except Exception as e:
        print(f"❌ Ocurrió un error al intentar leer el fichero TPM seleccionado: {e}")
        sys.exit(1)

    # 2. Selección del Sistema Candidato
    print(f"\n✔ La red '{ruta_archivo.name}' consta de {n_nodos} variables lógicas asociadas.")
    candidato_input = input(f"📝 Ingrese el SISTEMA CANDIDATO en binario (longitud: {n_nodos} bits, 1=incluido, 0=marginalizado)\n"
                            f"   O presione [ENTER] para tomar TODO el sistema: ").strip()

    if not candidato_input:
        bits_candidato = "1" * n_nodos
        print(f"✨ Seleccionado el sistema completo: {bits_candidato}")
    else:
        if set(candidato_input).issubset({"0", "1"}) and len(candidato_input) == n_nodos:
            bits_candidato = candidato_input
        else:
            print(f"❌ Error: La máscara del sistema candidato debe ser una cadena de unos y ceros y coincidir con la longitud de la red ({n_nodos}).")
            sys.exit(1)

    n_candidatos = bits_candidato.count("1")
    if n_candidatos < 2:
        print("❌ Error: El sistema candidato debe tener al menos 2 variables (1s) para ser particionado.")
        sys.exit(1)

    # 3. Selección del estado inicial del candidato
    estado_input = input(f"\n📝 Ingrese el ESTADO INICIAL del sistema candidato en binario (longitud: {n_candidatos} bits)\n"
                         f"   O simplemente presione [ENTER] para generar uno RÁNDOM: ").strip()

    if not estado_input:
        estado_candidato = "".join(str(random.randint(0, 1)) for _ in range(n_candidatos))
        print(f"🎲 ¡Estado aleatorio auto-generado validado para el candidato! => {estado_candidato}")
    else:
        if set(estado_input).issubset({"0", "1"}) and len(estado_input) == n_candidatos:
            estado_candidato = estado_input
        else:
            print(f"❌ Error: El estado inicial debe ser estrictamente una cadena de unos y ceros y tener exactamente {n_candidatos} bits de longitud.")
            sys.exit(1)

    # Reconstruir estado completo (rellenando con 0s el background marginalizado)
    # El Manager interno necesita una cadena de tamaño completo N, aunque los 0s impuestos acá serán 
    # ignorados matemáticamente porque la máscara `bits_candidato` indica que no están condicionados.
    estado_completo_list = ["0"] * n_nodos
    idx_cand = 0
    for i, bit in enumerate(bits_candidato):
        if bit == "1":
            estado_completo_list[i] = estado_candidato[idx_cand]
            idx_cand += 1
    estado = "".join(estado_completo_list)

    # 4. Solicitud variable K 
    while True:
        k_str = input(f"\n📝 Ingrese el valor de la partición (k) que desea agrupar para este candidato (Min: 2, Max: {n_candidatos}): ").strip()
        try:
            k_val = int(k_str)
            if 2 <= k_val <= n_candidatos:
                break
            else:
                print(f"⚠️ 'k' fuera de tolerancia. Debe estar en el rango de 2 a {n_candidatos}.")
        except ValueError:
            print("⚠️ Ingrese un valor numérico entero y natural.")

    sep(f"🚀 Iniciando Algoritmo · KGeoMIP k={k_val}")
    print(f"  ▶ Dataset Cargado : {ruta_archivo.name}")
    print(f"  ▶ Sist. Candidato : {bits_candidato} ({n_candidatos} nodos)")
    print(f"  ▶ Estado Central  : {estado}")
    print(f"  ▶ N° Particiones  : {k_val}\n")

    # 5. Instanciar ejecución y despliegue del motor modificado
    kgeomip_inst = KGeoMIP(Manager(estado), k=k_val)
    solucion = kgeomip_inst.aplicar_estrategia(
        condicion=bits_candidato,
        alcance=bits_candidato,
        mecanismo=bits_candidato,
        tpm=tpm
    )

    print(solucion)

    # 5. Guardar los resultados en la carpeta 'results'
    import json
    
    results_dir = GEOMIP_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    nombre_archivo = f"resultado_{ruta_archivo.stem}_k{k_val}.json"
    ruta_salida = results_dir / nombre_archivo
    
    # Extraemos y formateamos la información del objeto Solution
    res_data = {
        "dataset": ruta_archivo.name,
        "estado_inicial": estado,
        "k_particiones": k_val,
        "estrategia": getattr(solucion, "estrategia", "kgeomip"),
        "perdida_phi": float(solucion.perdida),
        "distribucion_subsistema": solucion.distribucion_subsistema.tolist() if hasattr(solucion.distribucion_subsistema, 'tolist') else solucion.distribucion_subsistema,
        "distribucion_particion": solucion.distribucion_particion.tolist() if hasattr(solucion.distribucion_particion, 'tolist') else solucion.distribucion_particion,
        "particion": str(solucion.particion),
        "tiempo_total": float(getattr(solucion, "tiempo_total", 0.0))
    }
    
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        json.dump(res_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n💾 ¡Resultados guardados exitosamente en: {ruta_salida.relative_to(GEOMIP_ROOT)}")

if __name__ == "__main__":
    main()