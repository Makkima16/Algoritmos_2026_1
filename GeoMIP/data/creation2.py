import numpy as np
import os
from pathlib import Path

def generate_modular_tpm(n_nodes, blocks, noise=0.0):
    """
    Genera una TPM donde las variables están fuertemente conectadas 
    solo con los miembros de su propio bloque.
    
    n_nodes: Número total de variables (N).
    blocks: Lista de listas, donde cada sublista contiene los índices de las variables de ese bloque.
            Ej: [[0, 1], [2, 3]] para N=4.
    noise: Probabilidad de ruido para evitar que sea 100% determinista (opcional).
    """
    # 2^N estados posibles
    n_states = 2 ** n_nodes
    tpm = np.zeros((n_states, n_nodes))
    
    # Para cada bloque, generamos una tabla de verdad aleatoria 
    # que dependa solo de las variables de ese bloque.
    block_logic = {}
    for block in blocks:
        n_block_states = 2 ** len(block)
        # Probabilidades aleatorias (muy marcadas: o cerca de 1 o cerca de 0) para fuerte fuerte interdependencia
        probs = np.random.choice([0.05, 0.95], size=(n_block_states, len(block)))
        block_logic[tuple(block)] = probs
        
    for state_idx in range(n_states):
        # Convertir estado a su representación binaria
        bin_str = format(state_idx, f'0{n_nodes}b')
        state_arr = np.array([int(b) for b in bin_str])
        
        for block in blocks:
            # Extraer el subestado del bloque
            sub_state = state_arr[block]
            # Índice del subestado
            sub_idx = int("".join(sub_state.astype(str)), 2)
            
            # Asignar probabilidades a las variables del bloque
            for i, var_idx in enumerate(block):
                prob = block_logic[tuple(block)][sub_idx, i]
                # Aplicamos algo de ruido aleatorio uniforme si se desea
                if noise > 0:
                    prob = prob * (1 - noise) + (1 - prob) * noise
                tpm[state_idx, var_idx] = prob
                
    return tpm

def main():
    # ==== CONFIGURACIÓN ====
    N = 6
    # Vamos a forzar 3 grupos de 2 nodos: (A,B), (C,D), (E,F)
    # Índices: 0=A, 1=B, 2=C, 3=D, 4=E, 5=F
    bloques = [[0, 1], [2, 3], [4, 5]]
    
    # Nombre del archivo de salida
    filepath = Path(__file__).parent / "samples" / "N6A_Fuerte.csv"
    filepath.parent.mkdir(exist_ok=True)
    
    print(f"Generando TPM para N={N} con bloques fuertemente conectados: {bloques}")
    tpm = generate_modular_tpm(N, bloques, noise=0.01)
    
    # Guardar a CSV
    np.savetxt(filepath, tpm, delimiter=",", fmt="%.4f")
    print(f"✅ Archivo guardado exitosamente en:\n {filepath}")
    print("\n¡Ejecuta exec_kgeomip.py y selecciona este archivo para ver cómo")
    print("el algoritmo ahora particiona agrupando obligatoriamente estos bloques!")

if __name__ == "__main__":
    main()
