import numpy as np

from src.constants.base import ABC_START
from src.models.base.application import aplicacion
from src.models.enums.notation import Notation

# Umbral usado por QNodes para adaptar el número de iteraciones de ILS/refinamiento
# según N. No cambia la métrica: _emd_bloques usa siempre la suma L1 marginal exacta.
HAMMING_EMD_MAX_N = 12


# @cache
def get_labels(n: int) -> tuple[str, ...]:
    """
    Genera las etiquetas STR en formato Excel para un sistema de n nodos.

    Args:
        n (int): El número de nodos.

    Returns:
        tuple[str, ...]: Las etiquetas como String en formato Excel para un sistema de n nodos.
    """

    def get_excel_column(n: int) -> str:
        if n <= 0:
            return ""
        return get_excel_column((n - 1) // 26) + chr((n - 1) % 26 + ord(ABC_START))

    return tuple([get_excel_column(i) for i in range(1, n + 1)])


ABECEDARY = get_labels(40)
LOWER_ABECEDARY = [letter.lower() for letter in ABECEDARY]


def reindexar(n: int) -> np.ndarray:
    """
    Genera una secuencia de números en una notación específica.
    """
    notaciones = {
        Notation.BIG_ENDIAN.value: big_endian(n),
        Notation.LIL_ENDIAN.value: lil_endian(n),
        # ...otras
    }
    notacion = (
        aplicacion.notacion_indexado.value
        if isinstance(aplicacion.notacion_indexado, Notation)
        else str(aplicacion.notacion_indexado)
    )
    if notacion not in notaciones:
        opciones = ", ".join(sorted(notaciones.keys()))
        raise ValueError(
            f"Notación de indexado no soportada: '{notacion}'. "
            f"Opciones disponibles: {opciones}"
        )
    return notaciones[notacion]


def seleccionar_estado(subestado: np.ndarray) -> np.ndarray:
    """
    Selecciona el estado de un n-cubo según la notación específica utilizada en el sistema.
    """
    # posible in-deducción por acceso inverso
    notaciones = {
        Notation.BIG_ENDIAN.value: subestado,
        Notation.LIL_ENDIAN.value: subestado[::-1],
        # ...otras
    }
    notacion = (
        aplicacion.notacion_indexado.value
        if isinstance(aplicacion.notacion_indexado, Notation)
        else str(aplicacion.notacion_indexado)
    )
    if notacion not in notaciones:
        opciones = ", ".join(sorted(notaciones.keys()))
        raise ValueError(
            f"Notación de estado no soportada: '{notacion}'. "
            f"Opciones disponibles: {opciones}"
        )
    return notaciones[notacion]


def big_endian(n: int) -> np.ndarray:
    """
    Implementación para generación de números en notación big endian.
    """
    return np.array(range(n), dtype=np.uint32)


def lil_endian(n: int) -> np.ndarray:
    """
    Implementación final optimizada para generación de números en notación little endian.
    """
    if n <= 0:
        # Caso especial para n=0
        return np.array([0], dtype=np.uint32)

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
