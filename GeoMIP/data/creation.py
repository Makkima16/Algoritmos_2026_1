import numpy as np
import time
import os
import sys


class SystemCreator:
    def __init__(self, N: int):
        self.N = N
        self.num_states = 2**N

        total_size_gb = (self.num_states * N) / (1024**3)
        print(f'\nTamaño estimado: {total_size_gb:.6f} GB')
        if total_size_gb > 1:
            confirm = input('El sistema ocupará más de 1GB. ¿Desea continuar? (s/n): ')
            if confirm.lower() != 's':
                sys.exit('Operación cancelada por el usuario')

        estimated_time = total_size_gb * 2
        print(f'Tiempo estimado: {estimated_time:.1f} segundos ({estimated_time/60:.1f} minutos)')

        print('Generando estados (probabilidades correlacionadas)...')
        start_time = time.time()
        
        # 1. Generar matriz de probabilidades base evitando el ruido blanco puro (0.5 puro)
        self.states = np.random.uniform(0.2, 0.8, size=(self.num_states, N))
        
        # 2. Introducir Correlación: Dependencia del valor j respecto a los bits del estado actual
        # Obtenemos la representación binaria de cada fila (estado)
        state_indices = np.arange(self.num_states)[:, np.newaxis]
        shifts = np.arange(N - 1, -1, -1)
        binary_states = (state_indices >> shifts) & 1
        
        # Hacemos que cada nodo dependa de sus vecinos en la cadena de bits para crear una alta integración
        left_neighbors = np.roll(binary_states, shift=1, axis=1)
        right_neighbors = np.roll(binary_states, shift=-1, axis=1)
        
        # Ajustamos las probabilidades base según la activación de los nodos circundantes
        self.states += 0.15 * left_neighbors
        self.states -= 0.10 * right_neighbors
        
        # Garantizamos que las probabilidades se mantengan en el rango (0, 1) excluyente para emd
        self.states = np.clip(self.states, 0.000001, 0.999999)
        
        elapsed = time.time() - start_time
        print(f'Generación completada en {elapsed:.2f} segundos')

    def marginalize(self, dimension: int) -> np.ndarray:
        if dimension < 1 or dimension >= self.N:
            raise ValueError(f'La dimensión debe estar en [1, {self.N-1})')
        return self.states[:, dimension]

    def save_to_csv(self, filename: str = None):
        filename = f'Sys{self.N}.csv' if filename is None else filename

        os.makedirs('GeoMIP\.assets', exist_ok=True)
        filepath = os.path.join('GeoMIP\.assets', filename)
        print(f'\nGuardando estados en {filepath}...')

        start_time = time.time()

        # Guardar solo la data, sin header, y con precisión flotante (Probabilidades)
        np.savetxt(filepath, self.states, delimiter=',', fmt='%.6f')

        elapsed = time.time() - start_time
        file_size_gb = os.path.getsize(filepath) / (1024**3)
        print(f'Archivo guardado: {file_size_gb:.6f} GB')
        print(f'Tiempo de guardado: {elapsed:.2f} segundos')


def generate_and_save(N: int):
    print(f'\nGenerando sistema con N={N}...')
    start_total = time.time()

    system = SystemCreator(N)
    system.save_to_csv()

    total_time = time.time() - start_total
    print(f'\nTiempo total del proceso: {total_time:.2f} segundos ({total_time/60:.2f} minutos)')
    return system


if __name__ == '__main__':
    try:
        # Generar un sistema que sea un reto real computacional sin saturar el cálculo de EMD.
        system = generate_and_save(10)
    except KeyboardInterrupt:
        print('\nOperación cancelada por el usuario')
    except Exception as e:
        print(f'\nError: {str(e)}')
