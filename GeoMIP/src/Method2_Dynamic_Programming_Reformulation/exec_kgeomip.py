"""
Script de ejecución interactivo para KGeoMIP — k-Partición de Mínima Información.

Permite al usuario seleccionar el dataset a través de una ventana de exploración
de archivos nativa (apuntando a la carpeta de dependencias `data/samples`), 
declarar o generar aleatoriamente su estado inicial (según la longitud de su red), 
y el valor de la partición 'k' a efectuar.

Ejemplo de uso:
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

    # 2. Selección del estado inicial
    print(f"\n✔ La red '{ruta_archivo.name}' consta de {n_nodos} variables lógicas asociadas.")
    estado_input = input(f"📝 Ingrese el estado inicial del sistema en binario (longitud: {n_nodos} bits)\n"
                         f"   O simplemente presione [ENTER] para generar uno RÁNDOM: ").strip()

    if not estado_input:
        estado = "".join(str(random.randint(0, 1)) for _ in range(n_nodos))
        print(f"🎲 ¡Estado aleatorio auto-generado validado! => {estado}")
    else:
        if set(estado_input).issubset({"0", "1"}) and len(estado_input) == n_nodos:
            estado = estado_input
        else:
            print(f"❌ Error: El estado debe ser estrictamente una cadena de unos y ceros y coincidir con la longitud de la red ({n_nodos}).")
            sys.exit(1)

    # 3. Solicitud variable K 
    while True:
        k_str = input(f"\n📝 Ingrese el valor de la '{n_nodos}-partición' (k) que desea agrupar (Min: 2, Max: {n_nodos}): ").strip()
        try:
            k_val = int(k_str)
            if 2 <= k_val <= n_nodos:
                break
            else:
                print(f"⚠️ 'k' fuera de tolerancia. Debe estar en el rango de 2 a {n_nodos}.")
        except ValueError:
            print("⚠️ Ingrese un valor numérico entero y natural.")

    # Alcance de todo el marco condicional
    bits_integrales = "1" * n_nodos

    sep(f"🚀 Iniciando Algoritmo · KGeoMIP k={k_val}")
    print(f"  ▶ Dataset Cargado : {ruta_archivo.name}")
    print(f"  ▶ Estado Central  : {estado}")
    print(f"  ▶ N° Particiones  : {k_val}\n")

    # 4. Instanciar ejecución y despliegue del motor modificado
    kgeomip_inst = KGeoMIP(Manager(estado), k=k_val)
    solucion = kgeomip_inst.aplicar_estrategia(
        condicion=bits_integrales,
        alcance=bits_integrales,
        mecanismo=bits_integrales,
        tpm=tpm
    )

    print(solucion)


if __name__ == "__main__":
    main()