import numpy as np
from numpy.typing import NDArray
from pyemd import emd

from src.models.enums.notation import Notation

from src.models.base.application import aplicacion
from src.constants.base import ABC_START


# @cache
def get_labels(n: int) -> tuple[str, ...]:
    def get_excel_column(n: int) -> str:
        if n <= 0:
            return ""
        return get_excel_column((n - 1) // 26) + chr((n - 1) % 26 + ord(ABC_START))

    return tuple([get_excel_column(i) for i in range(1, n + 1)])


ABECEDARY = get_labels(40)
LOWER_ABECEDARY = [letter.lower() for letter in ABECEDARY]


_HAMMING_CACHE = {}

def get_hamming_matrix(n: int) -> NDArray[np.float64]:
    """
    Pre-calculations mechanism for the Hamming distance cost matrix.
    Avoids calculating the distance repeatedly during Search.
    
    Args:
        n (int): The dimensions of the distribution (2^n states typically, but could be number of bins).
    """
    # Para N >= 13 nodos, la distribución conjunta tiene 2^N >= 8192 estados
    # y la matriz (2^N × 2^N) supera 512 MB. Rechazar para forzar uso de EMD rápida.
    if n > 4096:
        raise MemoryError(f"No se puede evaluar EMD clásica con tamaño {n}x{n} por memoria.")
        
    if n not in _HAMMING_CACHE:
        costs = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            costs[i, :i] = [hamming_distance(i, j) for j in range(i)]
            costs[:i, i] = costs[i, :i]  # Reflejo simétrico
        np.fill_diagonal(costs, 0.0)
        _HAMMING_CACHE[n] = costs
    return _HAMMING_CACHE[n]


def emd_causal(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    """
    Calculate the Earth Mover's Distance (EMD) between two probability distributions u and v.
    The Hamming distance was used as the ground metric.
    """
    if not all(isinstance(arr, np.ndarray) for arr in [u, v]):
        raise TypeError("u and v must be numpy arrays.")

    n: int = u.size
    cost_mat = get_hamming_matrix(n)
    return emd(u, v, cost_mat)


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def reindexar(N: int):
    notaciones = {
        Notation.BIG_ENDIAN.value: range(N),
        Notation.LIL_ENDIAN.value: lil_endian(N),
        # ...otras
    }
    return notaciones[aplicacion.notacion]


def seleccionar_subestado(subestado):
    # posible in-deducción por acceso inverso
    notaciones = {
        Notation.BIG_ENDIAN.value: subestado,
        Notation.LIL_ENDIAN.value: subestado[::-1],
        # ...otras
    }
    return notaciones[aplicacion.notacion]


def lil_endian(n: int) -> np.ndarray:
    """
    Implementación final optimizada para generación de little endian.
    Combina las mejores prácticas encontradas en nuestras pruebas.
    """
    if n <= 0:
        return np.array([0], dtype=np.uint32)  # Caso especial para n=0

    size = 1 << n
    result = np.zeros(size, dtype=np.uint32)

    # Optimización de parámetros basada en n
    block_bits = max(12, min(16, 28 - int(np.log2(n))))
    block_size = 1 << block_bits

    # Precomputar shifts de una vez
    shifts = np.array([n - i - 1 for i in range(n)], dtype=np.uint32)

    # Pre-alocar buffer para procesamientos por bloque
    block_result = np.zeros(block_size, dtype=np.uint32)

    # Determinar tamaño óptimo de grupo de bits
    bit_group_size = 6 if n > 24 else 4  # Ajuste basado en pruebas empíricas

    for start in range(0, size, block_size):
        end = min(start + block_size, size)
        current_size = end - start

        # Reset eficiente del buffer
        block_result[:current_size] = 0
        block_indices = np.arange(start, end, dtype=np.uint32)

        # Procesar bits en grupos optimizados
        for base_bit in range(0, n, bit_group_size):
            bits_remaining = min(bit_group_size, n - base_bit)
            if bits_remaining <= 0:
                break

            # Optimización: procesamiento de múltiples bits de una vez
            group_mask = np.uint32((1 << bits_remaining) - 1)
            group_values = (block_indices >> base_bit) & group_mask

            for j in range(bits_remaining):
                shift = shifts[base_bit + j]
                bit_value = (group_values >> j) & np.uint32(1)
                block_result[:current_size] |= bit_value << shift

        result[start:end] = block_result[:current_size]

    return result
