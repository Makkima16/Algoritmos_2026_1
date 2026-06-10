"""
Optimizaciones de EMD para N > 12 (donde construir la matriz de Hamming 2^N x 2^N es inviable).

La métrica EMD sobre los grafos de Hamming (que es el cálculo real para distribuciones en 
sistemas lógicos o booleanos) cuenta con la propiedad matemática de que: 
Si dos distribuciones de probabilidad multivariada están compuestas de nodos condicionalmente 
independientes, el EMD en el hipercubo sobre sus distribuciones conjuntas es EXACTAMENTE igual 
a la suma del EMD 1D de sus distribuciones marginales nodo por nodo.

EMD(P, Q) = Sum_i | P(nodo_i = 1) - Q(nodo_i = 1) |

Esta reformulación permite que el costo en tiempo pase de O(2^N) a O(N), y la memoria sea de 
unos bytes en vez de Gigabytes.
"""

import numpy as np


def _emd_marginal_sum(u: np.ndarray, v: np.ndarray) -> float:
    """
    Suma absoluta de EMDs marginales. 
    Aplica para distribuciones pre-factorizadas como conjuntos de variables independientes.
    """
    return float(np.sum(np.abs(u - v)))


def emd_causal_fast_partition(
    dist_original: np.ndarray,
    dist_reconstruida: np.ndarray,
    n_nodos: int,
    use_marginal: bool = True
) -> float:
    """
    Versión optimizada para evaluación de particiones en GeoMIP.
    
    A través de IIT, se evalúa las diferencias en probabilidades marginales, en
    vez de expandir a distribuciones de 2^N elementos.
    
    Args:
        dist_original: Probabilidades marginales originales (shape N)
        dist_reconstruida: Probabilidades marginales particionadas (shape N)
        n_nodos: Número de nodos del sistema
        use_marginal: Flag legacy, actualmente asume False eval para fallbacks
    
    Returns:
        Distancia EMD en float (MIP exacto/Phi)
    """
    if use_marginal or len(dist_original) == n_nodos:
        return _emd_marginal_sum(dist_original, dist_reconstruida)
    
    # Excepción explícita si entra una conjuta.
    # El flujo en KGeoMIP para N > 12 debe siempre pasar las marginales pre calculadas.
    n_total = len(dist_original)
    if n_total <= 4096:  # O sea N <= 12
        from funcs.base import emd_causal as emd_causal_exact
        return emd_causal_exact(dist_original, dist_reconstruida)

    raise ValueError(
        f"Para evaluar N={n_nodos} en GeoMIP, pase el vector marginal (tamaño {n_nodos}) "
        f"y no la conjunta ensamblada de tamaño {n_total}."
    )

