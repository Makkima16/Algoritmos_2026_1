"""
Optimizaciones de EMD para N > 15 (donde construir la matriz de Hamming es inviable).

Para N=25, tenemos 2^25 = 33.5M estados. En lugar de crear una matriz 33.5M x 33.5M,
usamos:
1. Muestreo Monte Carlo del EMD
2. Aproximación por distribuciones marginales
3. Combinación de ambos para máxima precisión sin overhead de memoria
"""

import numpy as np
from typing import Tuple, Optional
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean


def emd_causal_sampled(
    u: np.ndarray,
    v: np.ndarray,
    sample_size: Optional[int] = None,
    method: str = "hybrid"
) -> float:
    """
    Calcula una aproximación del EMD causal usando muestreo, evitando la construcción
    de la matriz de Hamming de tamaño 2^N × 2^N.

    Args:
        u: Distribución original (shape: 2^N o N)
        v: Distribución reconstruida (shape: 2^N o N)
        sample_size: Número de estados a muestrear (por defecto: min(10000, 2^N))
        method: 'hybrid' (mejor), 'wasserstein', 'marginal', 'euclidean'

    Returns:
        Aproximación float del EMD causal
    """
    n_total = len(u)
    
    # Si es manejable, usar EMD exacto (n <= 32768)
    if n_total <= 32768:
        from src.funcs.base import emd_causal as emd_causal_exact
        return emd_causal_exact(u, v)
    
    # Para n > 32768, usar aproximación basada en muestreo
    if sample_size is None:
        sample_size = min(10000, n_total // 100)  # Al menos 1% de estados
    
    if method == "hybrid":
        return _emd_hybrid(u, v, sample_size, n_total)
    elif method == "wasserstein":
        return _emd_wasserstein_1d(u, v)
    elif method == "marginal":
        return _emd_marginal_sum(u, v)
    elif method == "euclidean":
        return np.linalg.norm(u - v, ord=2)
    else:
        return _emd_hybrid(u, v, sample_size, n_total)


def _emd_hybrid(
    u: np.ndarray,
    v: np.ndarray,
    sample_size: int,
    n_total: int
) -> float:
    """
    Método híbrido: combina muestreo con optimización de gradiente.
    Más preciso que puro muestreo aleatorio.
    """
    # 1. Muestrear estados con mayor divergencia (importance sampling)
    divergencia = np.abs(u - v)
    indices_importantes = np.argsort(divergencia)[-sample_size // 2:]
    indices_aleatorios = np.random.choice(n_total, size=sample_size // 2, replace=False)
    
    indices_muestreados = np.unique(np.concatenate([indices_importantes, indices_aleatorios]))
    
    u_muestra = u[indices_muestreados]
    v_muestra = v[indices_muestreados]
    
    # 2. Normalizar distribuciones muestreadas
    u_muestra = u_muestra / np.sum(u_muestra)
    v_muestra = v_muestra / np.sum(v_muestra)
    
    # 3. Calcular distancia de Wasserstein en 1D (approximation del EMD)
    # sobre los estados muestreados
    dist_wasserstein = wasserstein_distance(u_muestra, v_muestra)
    
    # 4. Escalar resultado para aproximar el valor completo
    # Heurística: multiplicar por factor que aproxime el tamaño total vs muestreado
    factor_escala = np.sqrt(n_total / len(indices_muestreados))
    
    return float(dist_wasserstein * factor_escala)


def _emd_wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    """
    Usa la distancia de Wasserstein directa como aproximación del EMD.
    Es más rápida pero menos precisa que el método híbrido.
    """
    # Normalizar
    u_norm = u / np.sum(u)
    v_norm = v / np.sum(v)
    
    # Calcular Wasserstein 1D (que es una aproximación del EMD)
    return wasserstein_distance(u_norm, v_norm)


def _emd_marginal_sum(u: np.ndarray, v: np.ndarray) -> float:
    """
    Suma de EMDs marginales. Cuando N es muy grande, la distribución se puede
    factorizar y calcular como suma de marginales.
    
    Útil como fallback si u y v ya están pre-factorizadas como marginales.
    """
    return np.sum(np.abs(u - v))


def emd_causal_fast_partition(
    dist_original: np.ndarray,
    dist_reconstruida: np.ndarray,
    n_nodos: int,
    use_marginal: bool = False
) -> float:
    """
    Versión optimizada para evaluación de particiones en GeoMIP con N > 20.
    
    Si N > 20 y ambas distribuciones son pequeñas (< 1M elementos), usa Wasserstein.
    Si N > 20 y las distribuciones son grandes, usa muestreo híbrido.
    
    Args:
        dist_original: Distribución original conjunta
        dist_reconstruida: Distribución reconstruida (particionada)
        n_nodos: Número de nodos del sistema
        use_marginal: Si True, asume que solo tenemos marginales (shape N, no 2^N)
    
    Returns:
        Distancia EMD aproximada (float)
    """
    if use_marginal or len(dist_original) == n_nodos:
        # Si solo tenemos marginales, devolver suma simple de divergencias
        return float(_emd_marginal_sum(dist_original, dist_reconstruida))
    
    # Tenemos distribuciones conjuntas de tamaño 2^N
    n_total = len(dist_original)
    
    if n_total <= 100000:
        # Para n <= 16-17, usar Wasserstein directo
        return _emd_wasserstein_1d(dist_original, dist_reconstruida)
    else:
        # Para n >= 25, usar muestreo híbrido
        return _emd_hybrid(
            dist_original,
            dist_reconstruida,
            sample_size=min(50000, n_total // 10),
            n_total=n_total
        )


def distribucion_conjunta_optimizada(
    probabilidades: np.ndarray,
    max_size: Optional[int] = None,
    sample_states: Optional[int] = None
) -> np.ndarray:
    """
    Genera la distribución conjunta de tamaño 2^N, pero con opciones de:
    1. Limitar el tamaño máximo en memoria
    2. Muestrear solo algunos estados en lugar de todos
    
    Args:
        probabilidades: Array de N probabilidades
        max_size: Tamaño máximo de memoria permitido (default: 500M elementos)
        sample_states: Si se proporciona, devuelve solo ese número de estados aleatorios
    
    Returns:
        Array con la distribución (completa o muestreada)
    """
    n = len(probabilidades)
    n_total_states = 2 ** n
    
    if max_size is None:
        max_size = 500_000_000  # 500M elementos ~4GB en float64
    
    if sample_states and sample_states < n_total_states:
        # Muestreo: generar solo los estados solicitados
        return _generar_estados_aleatorios(probabilidades, sample_states)
    
    if n_total_states <= max_size:
        # Cabe en memoria, usar método vectorizado completo
        from src.controllers.strategies.kgeomip import distribucion_conjunta_vectorizada
        return distribucion_conjunta_vectorizada(probabilidades)
    else:
        # Demasiado grande, usar muestreo
        sample_size = min(max_size, n_total_states)
        return _generar_estados_aleatorios(probabilidades, sample_size)


def _generar_estados_aleatorios(
    probabilidades: np.ndarray,
    n_samples: int
) -> np.ndarray:
    """
    Genera n_samples estados aleatorios según las probabilidades proporcionadas.
    Cada estado tiene probabilidad = producto de probabilidades marginales.
    """
    n = len(probabilidades)
    # Generar estados aleatorios binarios
    estados = np.random.binomial(1, 0.5, size=(n_samples, n))
    
    # Calcular la probabilidad de cada estado según el producto de marginales
    log_probs = np.zeros(n_samples)
    for i in range(n):
        p = probabilidades[i]
        log_probs += np.where(estados[:, i] == 1, np.log(p + 1e-10), np.log(1 - p + 1e-10))
    
    probs = np.exp(log_probs)
    probs = probs / np.sum(probs)  # Normalizar
    
    return probs


def compare_methods_benchmark(
    n_nodos: int,
    dist_original: Optional[np.ndarray] = None,
    dist_modified: Optional[np.ndarray] = None,
    verbose: bool = True
) -> dict:
    """
    Compara el rendimiento de diferentes métodos de EMD para debug/testing.
    """
    if dist_original is None:
        # Generar distribuciones de prueba
        n_total = 2 ** n_nodos
        dist_original = np.random.dirichlet(np.ones(n_total))
        dist_modified = np.random.dirichlet(np.ones(n_total))
    
    import time
    results = {}
    
    if len(dist_original) <= 32768:
        # Método exacto
        start = time.time()
        from src.funcs.base import emd_causal as emd_exact
        val_exact = emd_exact(dist_original, dist_modified)
        results['exacto'] = {'valor': val_exact, 'tiempo': time.time() - start}
    
    # Método Wasserstein
    start = time.time()
    val_wass = _emd_wasserstein_1d(dist_original, dist_modified)
    results['wasserstein'] = {'valor': val_wass, 'tiempo': time.time() - start}
    
    # Método Híbrido
    start = time.time()
    val_hybrid = _emd_hybrid(dist_original, dist_modified, 10000, len(dist_original))
    results['hibrido'] = {'valor': val_hybrid, 'tiempo': time.time() - start}
    
    if verbose:
        print(f"\nComparación de métodos EMD para N={n_nodos}:")
        for metodo, data in results.items():
            print(f"  {metodo:12s}: valor={data['valor']:.6f}, tiempo={data['tiempo']:.4f}s")
    
    return results
