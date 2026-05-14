import sys
import os
import tkinter as tk
from tkinter import filedialog

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.controllers.manager import Manager
from src.strategies.dynamic import DynamicPartition

def seleccionar_archivo() -> str:
    """Abre un diálogo visual para elegir el archivo de muestras"""
    root = tk.Tk()
    root.withdraw() # Oculta la ventana principal
    ruta = filedialog.askopenfilename(
        title="Selecciona el archivo TPM (.csv)",
        filetypes=[("Archivos CSV", "*.csv")]
    )
    return ruta

def iniciar():
    """Punto de entrada"""
    # 1. Elegimos el archivo manualmente (estilo GeoMIP)
    archivo_seleccionado = seleccionar_archivo()
    
    if not archivo_seleccionado:
        print("No se seleccionó ningún archivo. Saliendo...")
        return
    
    # Pedir estado inicial
    estado_inicial = input("Ingresa el estado inicial (ej. 10000): ").strip()
    
    # Pedir sistema candidato (máscara binaria)
    candidato_input = input("Ingresa el sistema candidato en binario (ej. 11111): ").strip()

    print(f"\nGenerando análisis para:")
    print(f" - TPM: {archivo_seleccionado}")
    print(f" - Estado inicial: {estado_inicial}")
    print(f" - Sistema candidato (binario): {candidato_input}\n")
    
    gestor_redes = Manager(estado_inicial)
    
    # Cargar la red
    mpt = gestor_redes.cargar_red(ruta_archivo_csv=archivo_seleccionado)

    print("\nIniciando búsqueda dinámica de la k-partición óptima...")
    
    # Creamos y ejecutamos la estrategia dinámica (DP / Caché)
    analizador_dinamico = DynamicPartition(mpt)
    
    # El DynamicPartition espera el estado inicial y la máscara binaria del candidato
    solucion = analizador_dinamico.evaluar_todas_las_k_particiones(
        estado_inicial=estado_inicial,
        sistema_candidato=candidato_input
    )
    
    # Imprimir los resultados
    print(f"\n=====================================")
    print(f"Mejor partición encontrada (k={len(solucion.particion.split('|'))}):")
    print(f"Configuración: {solucion.particion}")
    print(f"Pérdida (EMD): {solucion.perdida}")
    print(f"Tiempo total : {solucion.tiempo_ejecucion:.4f}s")
    print(f"=====================================")


if __name__ == "__main__":
    iniciar()