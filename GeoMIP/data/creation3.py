import numpy as np
import time
import os
import sys

class DeterministicSystemCreator:
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

        print('Generando estados (puramente deterministas 0 o 1)...')
        start_time = time.time()
        
        # Generar matriz determinista (1 o 0 al azar)
        self.states = np.random.choice([0.0, 1.0], size=(self.num_states, N))
        
        elapsed = time.time() - start_time
        print(f'Generación completada en {elapsed:.2f} segundos')

    def save_to_csv(self, filename: str = None):
        # Utilizar la misma carpeta que el creation original
        target_dir = os.path.join(os.path.dirname(__file__), 'samples')
        os.makedirs(target_dir, exist_ok=True)

        if filename is None:
            # Buscar la siguiente letra disponible A, B, C... pero con sufijo Determinista
            for i in range(26):
                letter = chr(65 + i)
                filename = f'N{self.N}{letter}_Determinista.csv'
                filepath = os.path.join(target_dir, filename)
                if not os.path.exists(filepath):
                    break
        else:
            filepath = os.path.join(target_dir, filename)

        print(f'\nGuardando estados en {filepath}...')
        start_time = time.time()

        # Guardar como flotante con un solo decimal (0.0 o 1.0)
        np.savetxt(filepath, self.states, delimiter=',', fmt='%.1f')

        elapsed = time.time() - start_time
        file_size_gb = os.path.getsize(filepath) / (1024**3)
        print(f'Archivo guardado: {file_size_gb:.6f} GB')
        print(f'Tiempo de guardado: {elapsed:.2f} segundos')

def generate_and_save(N: int):
    print(f'\nGenerando sistema determinista con N={N}...')
    start_total = time.time()

    system = DeterministicSystemCreator(N)
    system.save_to_csv()

    total_time = time.time() - start_total
    print(f'\nTiempo total del proceso: {total_time:.2f} segundos ({total_time/60:.2f} minutos)')
    return system

if __name__ == '__main__':
    try:
        n_str = input('\nIngrese el número de variables (N) para el sistema: ').strip()
        n_val = int(n_str)
        if n_val < 1:
            raise ValueError()
        system = generate_and_save(n_val)
    except ValueError:
        print('\nError: Por favor ingrese un número entero válido y positivo.')
    except KeyboardInterrupt:
        print('\nOperación cancelada por el usuario')
    except Exception as e:
        print(f'\nError: {str(e)}')
