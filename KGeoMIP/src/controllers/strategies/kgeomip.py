"""
KGeoMIP — Extensión de GeoMIP a k-particiones mediante greedy top-down asimétrico.

Algoritmo (pipeline por cada k ∈ {2, 3, ..., min(5, N)} evaluado secuencialmente):
  1. _construir_tabla_costos  → tabla (2^n_dims × N) por recurrencia de capas de Hamming.
     Con Numba (@njit cache=True): un kernel C puro en una sola pasada; el binario se
     carga de disco en el primer arranque del proceso (~0.1–0.5 s) o se compila si no
     existe aún (~1–3 s). Sin Numba: vectorización numpy por shells, chunkeada para
     uso eficiente de caché CPU. Complejidad: O(n_dims × 2^n_dims).
  2. _construir_cut_pool      → O(N) cortes asimétricos (aislamiento simétrico,
     complemento y mecanismo vacío por nodo) + mejor representante por nivel Hamming.
  3. _greedy_k_particion      → greedy top-down: inicia con un bloque único (todos
     los futuros, todos los presentes) y aplica k-1 divisiones óptimas del pool.
  4. _refinar_bloques_1move   → best-improvement: movimientos futuro y presente asimétrico
     (FASE FINAL ACTIVA del motor en producción).

  Mejoras DESACTIVADAS (existen como código pero con bucle = 0; no aportan Φ y cuestan
  3–5×, ver docs/decision_sin_ils.md):
    · _refinar_bloques_2move  → VNS: pares de movimientos simultáneos (N_VNS_MAX = 0).
    · _perturbacion_bloques   → ILS ligero: perturbación + re-refinar (N_ILS_LIGHT = 0).

PARALELIZACIÓN:
  Los k se evalúan de forma SECUENCIAL; dentro de cada k se usan N_JOBS_INTERNOS = cpu_count-1
  hilos (joblib prefer="threads") en _mejor_split_bloques, _refinar_bloques_1move y
  _refinar_bloques_2move para evaluar candidatos y vecinos en paralelo.

ARRANQUE (primera ejecución del proceso):
  warmup_motor() debe llamarse ANTES del lote para absorber el coste de inicio sin
  contaminar los tiempos de las pruebas. Dispara el kernel Numba (carga del binario
  cacheado o compilación JIT) y el pool de hilos joblib con arrays mínimos de 2 nodos.
  Ver exec_kgeomip.py, modo bloque.
"""

import time
import itertools
import random as _random_module
from math import comb
from typing import List, Tuple, Dict, Optional
import multiprocessing

import numpy as np
from joblib import Parallel, delayed

try:
    from sklearn.cluster import SpectralClustering, AgglomerativeClustering
    _SKLEARN_DISPONIBLE = True
except ImportError:
    _SKLEARN_DISPONIBLE = False

try:
    from numba import njit
    _NUMBA_DISPONIBLE = True
except ImportError:
    _NUMBA_DISPONIBLE = False

from src.controllers.manager import Manager
from src.funcs.base import ABECEDARY, LOWER_ABECEDARY
from src.funcs.format import fmt_biparte_q
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.base import (
    NET_LABEL,
    ACTUAL,
    EFECTO,
    TYPE_TAG,
    VOID_STR,
    COST_TABLE_CHUNK_ROWS,
)



KGEOMIP_LABEL: str = "KGeoMIP"
KGEOMIP_STRATEGY_TAG: str = f"{KGEOMIP_LABEL}_strategy"
KGEOMIP_ANALYSIS_TAG: str = f"{KGEOMIP_LABEL}_analysis"

# Mantener la posibilidad de que una parte tenga mecanismo vacío (∅).
PERMITIR_PRESENTE_VACIO_POR_DEFECTO: bool = True

N_JOBS_INTERNOS: int = max(1, multiprocessing.cpu_count() - 1)

# (frozenset global futuros, frozenset global presentes)
# Permite cortes asimétricos: futuros y presentes se particionan independientemente.
Block = tuple


def _popcount_vec(x: np.ndarray) -> np.ndarray:
    """
    Vectorized popcount (Hamming weight) for uint32 arrays using byte unpacking.

    Args:
        x: Array of non-negative integers cast to uint32.

    Returns:
        int32 array with bit counts for each element.
    """
    x_u32 = np.asarray(x, dtype=np.uint32)
    x_bytes = x_u32.view(np.uint8).reshape(-1, 4)
    return np.unpackbits(x_bytes, axis=1).sum(axis=1).astype(np.int32)


if _NUMBA_DISPONIBLE:

    @njit(cache=True, nogil=True)
    def _kernel_tabla_costos(prob_T, prob_origen, xor_origen, orden, n_dims, n, tabla_T):
        """
        Llena tabla_T (2^n_dims, n) con la recurrencia por cáscaras de Hamming.

        Equivalente bit a bit a la ruta numpy: para cada estado j (procesado en
        orden de popcount ascendente vía `orden`) acumula primero
            vecinal[x] = Σ_b tabla_T[j⊕2^b, x]   (b: bits encendidos, orden asc.)
        y luego
            tabla_T[j, x] = (|prob_T[j,x] - prob_origen[x]| + vecinal[x]) · 2^-d
        con d = popcount(xor). El mismo orden de acumulación que la versión numpy
        garantiza igualdad exacta en float32. Cada vecino j⊕2^b está en la cáscara
        d-1, ya calculada (la fila de j arranca en cero y sirve de acumulador).
        """
        for t in range(orden.shape[0]):
            j = orden[t]
            xj = xor_origen[j]
            if xj == 0:
                # Origen: |p - p| = 0 y sin vecinos; la fila queda en cero.
                continue
            # vecinal acumulado en la propia fila (parte de cero) → cáscara d-1
            d = 0
            for b in range(n_dims):
                if (xj >> np.uint32(b)) & np.uint32(1):
                    d += 1
                    vecino = j ^ (np.int64(1) << np.int64(b))
                    for x in range(n):
                        tabla_T[j, x] += tabla_T[vecino, x]
            gamma = np.float32(1.0 / (np.int64(1) << np.int64(d)))
            for x in range(n):
                tabla_T[j, x] = (abs(prob_T[j, x] - prob_origen[x]) + tabla_T[j, x]) * gamma


def _generar_candidatos_aislamiento(
    n_vars: int, k: int
) -> "List[Tuple[List[int], ...]]":
    """
    Genera candidatos de 'aislamiento heurístico' para k-particiones.

    Crea todas las particiones donde exactamente k-1 variables están aisladas
    individualmente y el resto forman un único clúster residual.

    Para K=2: N candidatos (aislar nodo 0 vs. resto, nodo 1 vs. resto, ...).
    Para K=3: C(N,2) candidatos (dos nodos individuales + residual).
    Para K=4: C(N,3) candidatos. Etc.

    En el 99% de los casos para K=2, uno de estos candidatos reproduce el
    corte MIP exacto sin necesidad de búsqueda exhaustiva.

    Args:
        n_vars: Número total de variables en el subsistema.
        k     : Número de partes de la partición.

    Returns:
        Lista de tuplas de k listas de índices enteros (formato simétrico).
    """
    candidatos: list = []
    todos = list(range(n_vars))
    n_aislados = k - 1

    for aislados in itertools.combinations(todos, n_aislados):
        aislados_set = set(aislados)
        residual = [i for i in todos if i not in aislados_set]
        if not residual:
            continue
        partes: List[List[int]] = [[a] for a in aislados] + [residual]
        candidatos.append(tuple(partes))

    return candidatos


def _generar_candidatos_presente_vacio(
    n_vars: int, k: int
) -> "List[Tuple[List[int], ...]]":
    """
    Genera variantes de aislamiento donde los nodos aislados usan mecanismo vacío (∅).

    Formalmente equivale a bipartir(futuros_i, []) — el nodo futuro aislado se
    evalúa como si no dependiera de ningún estado presente. Esto es válido en IIT
    cuando la partición óptima corta todas las conexiones causales de ese nodo.

    Implementación: la lista de la parte aislada inicia con el centinela -1,
    que evaluar_k_particion interpreta como "usar presente vacío para esta parte".

    Para K=2: N candidatos. Para K=3: C(N,2). Para K>3: C(N,k-1).

    Args:
        n_vars: Número total de variables en el subsistema.
        k     : Número de partes.

    Returns:
        Lista de tuplas de k listas; partes aisladas inician con centinela -1.
    """
    candidatos: list = []
    todos = list(range(n_vars))
    n_aislados = k - 1

    for aislados in itertools.combinations(todos, n_aislados):
        aislados_set = set(aislados)
        residual = [i for i in todos if i not in aislados_set]
        if not residual:
            continue
        # Centinela -1 al inicio de cada parte aislada → presente vacío (∅)
        partes: List[List[int]] = [[-1, a] for a in aislados] + [residual]
        candidatos.append(tuple(partes))

    return candidatos


def _refinar_particion_local(
    subsistema,
    indices_ncubos: np.ndarray,
    dims_ncubos: np.ndarray,
    particion_inicial: Tuple[List[int], ...],
    dist_original: np.ndarray,
    permitir_presente_vacio: bool,
    tiempo_maximo_segundos: Optional[float] = None,
) -> tuple[float, Tuple[List[int], ...]]:
    """
    Refinamiento local por vecindad 1-move sobre una k-partición.

    Explora movimientos de un nodo entre bloques y, si se permite, la variante
    con mecanismo vacío para bloques unitarios.
    """
    inicio = time.perf_counter()

    def agotado() -> bool:
        if tiempo_maximo_segundos is None:
            return False
        return (time.perf_counter() - inicio) >= tiempo_maximo_segundos

    def normalizar(particion: Tuple[List[int], ...]) -> list[list[int]]:
        partes: list[list[int]] = []
        for parte in particion:
            if parte and parte[0] == -1:
                partes.append(list(parte))
            else:
                partes.append(list(parte))
        return partes

    def evaluar(particion: Tuple[List[int], ...]) -> float:
        return evaluar_k_particion(
            subsistema,
            indices_ncubos,
            dims_ncubos,
            particion,
            dist_original,
        )

    mejor_particion = tuple(list(parte) for parte in particion_inicial)
    mejor_perdida = evaluar(mejor_particion)
    mejoro = True

    while mejoro and not agotado():
        mejoro = False
        partes_base = normalizar(mejor_particion)
        vecinos: list[Tuple[List[int], ...]] = []
        vistos: set[tuple] = set()

        for i, parte_origen in enumerate(partes_base):
            if parte_origen and parte_origen[0] == -1:
                parte_origen = parte_origen[1:]
            if len(parte_origen) <= 1:
                continue

            for nodo_idx, nodo in enumerate(parte_origen):
                for j in range(len(partes_base)):
                    if i == j:
                        continue
                    candidato = [list(parte) for parte in partes_base]
                    candidato[i].remove(nodo)
                    if not candidato[i]:
                        continue
                    candidato[j].append(nodo)
                    tupla_candidato = tuple(tuple(parte) for parte in candidato)
                    if tupla_candidato not in vistos:
                        vecinos.append(tupla_candidato)
                        vistos.add(tupla_candidato)

        if permitir_presente_vacio:
            for idx_parte, parte in enumerate(partes_base):
                parte_real = parte[1:] if parte and parte[0] == -1 else parte
                if len(parte_real) != 1:
                    continue
                candidato = [list(parte) for parte in partes_base]
                candidato[idx_parte] = [-1, parte_real[0]]
                tupla_candidato = tuple(tuple(parte) for parte in candidato)
                if tupla_candidato not in vistos:
                    vecinos.append(tupla_candidato)
                    vistos.add(tupla_candidato)

        if agotado() or not vecinos:
            break

        resultados = Parallel(n_jobs=min(len(vecinos), N_JOBS_INTERNOS), prefer="threads")(
            delayed(evaluar)(vecino) for vecino in vecinos
        )
        idx_mejor = int(np.argmin(resultados))
        perdida_mejor_vecino = float(resultados[idx_mejor])

        if perdida_mejor_vecino + 1e-12 < mejor_perdida:
            mejor_perdida = perdida_mejor_vecino
            mejor_particion = vecinos[idx_mejor]
            mejoro = True

    return mejor_perdida, mejor_particion


def evaluar_k_particion(
    subsistema,
    indices_ncubos: np.ndarray,
    dims_ncubos: np.ndarray,
    particion: Tuple[List[int], ...],
    dist_original: np.ndarray,
) -> float:
    """
    Compute the EMD loss for a k-partition using marginal L1 distance.

    Args:
        subsistema      : Conditioned subsystem with bipartir() support.
        indices_ncubos  : Global indices of the subsystem n-cubes.
        dims_ncubos     : Active mechanism dimensions of the subsystem.
        particion       : Tuple of k lists of local variable indices.
                          A list beginning with sentinel -1 means empty mechanism (∅).
        dist_original   : Marginal distribution vector of size N (P(node=1) per node).

    Returns:
        float: EMD loss (sum of marginal L1 differences).
    """
    n = len(dist_original)
    dist_reconstruida = np.empty(n, dtype=np.float64)

    for parte in particion:
        if not parte:
            continue
        # Sentinel -1 at start → empty mechanism (∅): future does not depend on present
        if parte[0] == -1:
            parte_real = parte[1:]
            presentes_parte = np.array([], dtype=np.int8)
        else:
            parte_real = parte
            futuros_tmp = indices_ncubos[np.array(parte_real, dtype=np.int8)]
            presentes_parte = np.intersect1d(futuros_tmp, dims_ncubos)
        if not parte_real:
            continue
        futuros_parte = indices_ncubos[np.array(parte_real, dtype=np.int8)]
        sistema_parte = subsistema.bipartir(futuros_parte, presentes_parte)
        dist_parte = sistema_parte.distribucion_marginal()
        for idx_pos in parte_real:
            dist_reconstruida[idx_pos] = dist_parte[idx_pos]

    # dist_original and dist_reconstruida are always marginal vectors of size N
    # (probability ON per node). For conditionally independent parts — which is
    # guaranteed by construction in any k-partition — EMD with Hamming cost on
    # the joint 2^N equals the marginal L1 sum exactly (marginal decomposition
    # theorem for the Hamming hypercube). Valid for all N with no size restriction.
    return float(np.sum(np.abs(dist_original - dist_reconstruida)))

def evaluar_corte_asimetrico(
    subsistema,
    future_1: "List[int]",
    present_1: "List[int]",
    future_2: "List[int]",
    present_2: "List[int]",
    dist_original: np.ndarray,
) -> float:
    """
    Evaluate an asymmetric bipartition where each block's future and present
    sets are specified independently (not necessarily symmetric).

    Unlike evaluar_k_particion, which derives present_i as the intersection of
    future_i with dims_ncubos, this function accepts arbitrary present sets so
    that the mechanism of each block can differ from its future coverage.

    Args:
        subsistema   : Conditioned subsystem with bipartir() support.
        future_1     : Real node indices (global) for block-1 future (alcance).
        present_1    : Real node indices (global) for block-1 present (mecanismo).
        future_2     : Real node indices (global) for block-2 future.
        present_2    : Real node indices (global) for block-2 present.
        dist_original: Marginal distribution vector of size N (P(node=1)).

    Returns:
        float: Marginal L1 EMD loss between dist_original and the reconstructed
               joint distribution obtained from the two independent blocks.
    """
    n = len(dist_original)
    dist_rec = np.empty(n, dtype=np.float32)

    for future_side, present_side in ((future_1, present_1), (future_2, present_2)):
        if not future_side:
            continue
        fa = np.array(future_side,  dtype=np.int8)
        pa = np.array(present_side, dtype=np.int8) if present_side else np.array([], dtype=np.int8)
        sub = subsistema.bipartir(fa, pa)
        dist_sub = sub.distribucion_marginal()
        for idx in future_side:
            pos = np.where(subsistema.indices_ncubos == idx)[0]
            if pos.size:
                dist_rec[pos[0]] = dist_sub[pos[0]]

    # Acumular en float64 (como BruteForceKMIP/QNodes) para Φ consistente.
    return float(np.sum(np.abs(dist_original - dist_rec), dtype=np.float64))


def _serializar_particion(
    particion: Tuple[List[int], ...],
    indices_reales: np.ndarray,
    dims_mecanismo: np.ndarray,
) -> dict:
    """
    Convierte una k-partición en un dict serializable para JSON.
    Cada parte queda como {"futuro": ["A","B"], "presente": ["a","b"]}.
    Solo aparecen en "presente" los nodos cuyo índice global esté en dims_mecanismo.
    """
    mec_set = set(dims_mecanismo.tolist())
    partes = []
    for parte in particion:
        # Centinela -1 → presente vacío (∅) explícito para esta parte
        if parte and parte[0] == -1:
            parte_real = parte[1:]
            presentes = []
        else:
            parte_real = parte
            presentes = [LOWER_ABECEDARY[indices_reales[i]] for i in parte_real if indices_reales[i] in mec_set]
        futuros = [ABECEDARY[indices_reales[i]] for i in parte_real]
        partes.append({
            "futuro":   futuros,
            "presente": presentes if presentes else [VOID_STR],
        })
    return {"partes": partes}

def _construir_matriz_afinidad_desde_tabla(
    tabla_T: np.ndarray,
    n_vars: int,
) -> np.ndarray:
    """
    Build the N×N affinity matrix from the full cost table tabla_T.

    Uses all 2^N rows of tabla_T so the column profiles of each variable
    capture its cost behaviour across the entire hypercube.

    Methodology:
    - C = tabla_T, shape (2^N, N).  Column x is variable x's cost profile.
    - Affinity A[i,j] = (1 + cosine_similarity(col_i, col_j)) / 2 ∈ [0, 1].
    - Variables that respond similarly to hypercube transitions cluster together.

    Args:
        tabla_T: Full cost table of shape (2**N, N) in float32.
        n_vars : Number of n-cubes (N); must equal tabla_T.shape[1].

    Returns:
        Affinity matrix of shape (n_vars, n_vars) in float64.
    """
    C = tabla_T.astype(np.float64)  # (2^N, N)

    norms = np.linalg.norm(C, axis=0, keepdims=True)  # (1, N)
    norms = np.where(norms < 1e-12, 1.0, norms)
    C_norm = C / norms  # (2^N, N)

    A = C_norm.T @ C_norm   # (N, N), values in [-1, 1]
    A = (A + 1.0) / 2.0     # normalise to [0, 1]
    np.fill_diagonal(A, 1.0)

    return A.astype(np.float64)


def _coste_simetrico_estados(tabla_T: np.ndarray, full_mask: int) -> np.ndarray:
    """
    Per-state symmetric cost of the whole hypercube in one vectorised pass.

    For every state j the cost is sum_x min(tabla_T[j, x], tabla_T[j^mask, x]):
    each output variable contributes its cheaper side (j or its bit-complement).
    Returns a (2^n_dims,) float64 vector aligned with tabla_T's row index.
    """
    indices = np.arange(len(tabla_T), dtype=np.int64)
    complemento = indices ^ full_mask
    return np.minimum(tabla_T, tabla_T[complemento]).sum(axis=1, dtype=np.float64)


def _generar_candidatos_hipercubo_completo(
    tabla_T: np.ndarray,
    dist: np.ndarray,
    idx_origen: int,
    full_mask: int,
    n: int,
    chunk_size: int = COST_TABLE_CHUNK_ROWS,
) -> "List[Tuple[List[int], List[int]]]":
    """
    Derive geometric bipartition candidates directly from the cost table.

    The whole-table symmetric cost is computed once (vectorised), then for each
    Hamming shell d = 1..n_dims//2+1 the single cheapest representative state is
    taken and split into a (present, effect) bipartition by comparing, variable
    by variable, its cost against the bit-complement.

    Selection is a plain per-shell minimum over the precomputed cost vector — no
    intermediate ordering of the shell is needed because only the minimiser is
    used. Duplicate bipartitions across shells are dropped.

    Args:
        tabla_T   : (2^n_dims, n) float32 cost table.
        dist      : (2^n_dims,) Hamming distances to the origin.
        idx_origen: Integer index of the origin state (kept for signature
                    compatibility; the symmetric cost already encodes it).
        full_mask : (1 << n_dims) - 1, addresses the bit complement.
        n         : Number of output variables (n-cubes).
        chunk_size: Unused; kept for backward-compatible call sites.

    Returns:
        List of (present_part, effect_part) tuples, each a bipartition of {0..n-1}.
    """
    n_dims = len(tabla_T).bit_length() - 1
    coste = _coste_simetrico_estados(tabla_T, full_mask)

    candidatos: list = []
    vistos: set = set()

    for d in range(1, n_dims // 2 + 2):
        shell = np.flatnonzero(dist == d)
        if shell.size == 0:
            continue

        # Representante del nivel: el estado de menor coste geométrico.
        j_star = int(shell[np.argmin(coste[shell])])

        fila = tabla_T[j_star]
        fila_complemento = tabla_T[j_star ^ full_mask]

        # Cada variable cae del lado "efecto" si su coste no es mayor en j_star
        # que en el complemento; las demás forman el lado "presente".
        lado_efecto = fila <= fila_complemento
        efectos = np.flatnonzero(lado_efecto[:n]).tolist()
        presentes = np.flatnonzero(~lado_efecto[:n]).tolist()

        if not efectos or not presentes:
            continue

        firma = (tuple(efectos), tuple(presentes))
        if firma in vistos:
            continue
        vistos.add(firma)
        candidatos.append((presentes, efectos))

    return candidatos


def _particion_grafo_hipercubo(
    matriz_afinidad: np.ndarray,
    k: int,
    n_candidatos: int = 6,
) -> "List[Tuple[List[int], ...]]":
    """
    Genera candidatos de k-partición usando Spectral Clustering y
    AgglomerativeClustering con múltiples configuraciones sobre la
    matriz de afinidad NxN del hipercubo.

    Se prueban varias semillas y etiquetados en SpectralClustering, y
    múltiples estrategias de linkage en AgglomerativeClustering para
    cubrir regiones distintas del espacio de particiones. Cada candidato
    se evalúa por EMD y solo se conserva el mejor tras el refinamiento.

    Args:
        matriz_afinidad: Matriz (N, N) en [0,1]; 1 = máxima similitud.
        k              : Número de clústeres deseados.
        n_candidatos   : Máximo de particiones candidatas por método.

    Returns:
        Lista de tuplas de k listas de índices (nunca vacía: siempre hay fallback).
    """
    n = matriz_afinidad.shape[0]

    if k >= n:
        return [tuple([i] for i in range(n))]

    candidatos: list = []
    A = np.clip(matriz_afinidad, 0.0, 1.0)

    def _agregar(particion):
        if all(len(p) > 0 for p in particion) and particion not in candidatos:
            candidatos.append(particion)

    if _SKLEARN_DISPONIBLE:
        semillas = [42, 0, 7, 13, 99, 2024][:n_candidatos]

        # SpectralClustering con kmeans
        for semilla in semillas:
            try:
                etiquetas = SpectralClustering(
                    n_clusters=k,
                    affinity="precomputed",
                    random_state=semilla,
                    assign_labels="kmeans",
                    n_init=30,
                ).fit_predict(A)
                _agregar(tuple([i for i in range(n) if etiquetas[i] == c] for c in range(k)))
            except Exception:
                pass

        # SpectralClustering con discretize (variante independiente)
        for semilla in semillas[:3]:
            try:
                etiquetas = SpectralClustering(
                    n_clusters=k,
                    affinity="precomputed",
                    random_state=semilla,
                    assign_labels="discretize",
                    n_init=30,
                ).fit_predict(A)
                _agregar(tuple([i for i in range(n) if etiquetas[i] == c] for c in range(k)))
            except Exception:
                pass

        # AgglomerativeClustering con varias estrategias de linkage
        D = np.clip(1.0 - A, 0.0, None)
        np.fill_diagonal(D, 0.0)
        for linkage in ("average", "complete", "single"):
            try:
                etiquetas = AgglomerativeClustering(
                    n_clusters=k,
                    metric="precomputed",
                    linkage=linkage,
                ).fit_predict(D)
                _agregar(tuple([i for i in range(n) if etiquetas[i] == c] for c in range(k)))
            except Exception:
                pass

    # Fallback deterministico (sin sklearn): divide por columna de mayor varianza
    if not candidatos:
        varianzas = np.var(matriz_afinidad, axis=0)
        orden = np.argsort(varianzas)[::-1]
        etiquetas_fb = np.zeros(n, dtype=int)
        for c, idx in enumerate(orden):
            etiquetas_fb[idx] = c % k
        particion_fb = tuple(
            [i for i in range(n) if etiquetas_fb[i] == c]
            for c in range(k)
        )
        if all(len(p) > 0 for p in particion_fb):
            candidatos.append(particion_fb)
        else:
            grupos = [[] for _ in range(k)]
            for i in range(n):
                grupos[i % k].append(i)
            candidatos.append(tuple(grupos))

    return candidatos


def fmt_k_particion(
    particion: Tuple[List[int], ...],
    indices_reales: np.ndarray,
    dims_mecanismo: np.ndarray,
) -> str:
    """
    Formatea una k-partición para mostrarla en consola.

    Fila superior (MAYÚSCULAS) = nodos futuros/alcance (t+1).
    Fila inferior (minúsculas) = solo los nodos presentes/mecanismo (t)
    cuyo índice global esté en dims_mecanismo; ∅ si ninguno.

    Args:
        particion     : Tupla de k listas de índices locales al subsistema.
        indices_reales: Mapeo local → índice global de la red completa.
        dims_mecanismo: Índices globales activos en el mecanismo (t).
    """
    mec_set = set(dims_mecanismo.tolist())
    partes_fmt = []
    for parte in particion:
        # Centinela -1 → presente vacío (∅) explícito para esta parte
        if parte and parte[0] == -1:
            parte_real = parte[1:]
            str_pres = VOID_STR
        else:
            parte_real = parte
            presentes = [LOWER_ABECEDARY[indices_reales[i]] for i in parte_real if indices_reales[i] in mec_set]
            str_pres = ",".join(presentes) if presentes else VOID_STR
        futuros  = [ABECEDARY[indices_reales[i]] for i in parte_real]
        str_fut  = ",".join(futuros) if futuros else VOID_STR
        ancho = max(len(str_fut), len(str_pres)) + 2
        partes_fmt.append((f"|{str_fut:^{ancho}}|", f"|{str_pres:^{ancho}}|"))
    linea_top = "".join(t for t, _ in partes_fmt)
    linea_bot = "".join(b for _, b in partes_fmt)
    return f"{linea_top}\n{linea_bot}"


def _perturbacion_aleatoria(
    particion: "Tuple[List[int], ...]",
    n_movimientos: int = 2,
    semilla: int = 42,
) -> "Tuple[List[int], ...]":
    """
    Perturba una k-partición moviendo aleatoriamente nodos entre bloques.

    Garantiza que ningún bloque quede vacío. Usado por la Búsqueda Local
    Iterada (ILS) para escapar de mínimos locales del refinamiento 1-move.

    Args:
        particion    : Tupla de k listas de índices (puede incluir centinela -1).
        n_movimientos: Número de nodos a reubicar en la perturbación.
        semilla      : Semilla para reproducibilidad.
    """
    rng = _random_module.Random(semilla)
    partes = [list(p) for p in particion]
    k = len(partes)

    for _ in range(n_movimientos):
        candidatos_origen = [
            i for i, p in enumerate(partes)
            if sum(1 for x in p if x != -1) > 1
        ]
        if not candidatos_origen:
            break
        i_origen = rng.choice(candidatos_origen)
        nodos_reales = [x for x in partes[i_origen] if x != -1]
        nodo = rng.choice(nodos_reales)
        candidatos_destino = [j for j in range(k) if j != i_origen]
        if not candidatos_destino:
            continue
        i_destino = rng.choice(candidatos_destino)
        partes[i_origen].remove(nodo)
        partes[i_destino].append(nodo)

    return tuple(partes)


def _evaluar_k_completo(
    k: int,
    subsistema,
    indices_ncubos: np.ndarray,
    dims_ncubos: np.ndarray,
    dists_marginales: np.ndarray,
    n_vars: int,
    n_jobs_internos: int,
    matriz_afinidad: Optional[np.ndarray] = None,
    permitir_presente_vacio: bool = PERMITIR_PRESENTE_VACIO_POR_DEFECTO,
    tiempo_maximo_segundos: Optional[float] = None,
    tabla_T: Optional[np.ndarray] = None,
    dist_array: Optional[np.ndarray] = None,
    idx_origen: int = 0,
    full_mask: int = 0,
    candidatos_asimetricos: Optional[List[Tuple[List[int], List[int]]]] = None,
) -> dict:
    """
    Evaluate a specific k value using geometric heuristics and ILS.

    Pipeline:
      1. SpectralClustering + AgglomerativeClustering (multiple seeds/linkage).
      2. For k=2: symmetric geometric candidates from the full tabla_T (if provided).
      3. Isolation candidates: k-1 isolated nodes + residual cluster.
      4. Variants with empty mechanism (∅) if enabled.
      5. Parallel EMD evaluation over all candidates.
      6. Local 1-move refinement on the best candidate.
      7. Iterated Local Search (ILS): perturbation + re-refinement × N_ILS.
      8. For k=2: asymmetric candidates from tabla_T (evaluar_corte_asimetrico).
         Checked after the full symmetric pipeline; update best if lower Phi found.

    Args:
        k                     : Number of parts to evaluate.
        subsistema            : Conditioned/substracted System (serialisable).
        indices_ncubos        : Indices of the subsystem n-cubes.
        dims_ncubos           : Dimensions of the subsystem n-cubes.
        dists_marginales      : Marginal distribution of the subsystem.
        n_vars                : Number of variables in the subsystem.
        n_jobs_internos       : Threads available for joblib inside this process.
        matriz_afinidad       : N×N geometric affinity matrix (optional).
        tabla_T               : Full (2^n_dims, n) cost table (optional).
        dist_array            : (2^n_dims,) Hamming distances to origin (with tabla_T).
        idx_origen            : Integer index of the origin state.
        full_mask             : (1 << n_dims) - 1 bit mask.
        candidatos_asimetricos: List of (future_real_indices, present_real_indices)
                                asymmetric cut candidates generated by
                                KGeoMIP._candidatos_desde_tabla_T(). Only used for k=2.

    Returns:
        dict with keys: k, perdida, particion, particion_grafica, error, asimetrico.
        The 'asimetrico' key is None unless an asymmetric cut beat the symmetric
        pipeline; when set it holds (future_1, present_1, future_2, present_2) with
        real node indices, allowing the caller to reconstruct Phi correctly.
    """
    N_ILS = 4  # reinicios de búsqueda local iterada

    try:
        # ── Phase 1: candidate generation ────────────────────────────────────
        candidatos_geo: list = []
        if matriz_afinidad is not None:
            candidatos_geo = _particion_grafo_hipercubo(
                matriz_afinidad, k, n_candidatos=6
            )

        # Geometric candidates from the full tabla_T (k=2 only)
        if k == 2 and tabla_T is not None and dist_array is not None:
            for c_geo in _generar_candidatos_hipercubo_completo(
                tabla_T, dist_array, idx_origen, full_mask, n_vars
            ):
                if c_geo not in candidatos_geo:
                    candidatos_geo.append(c_geo)

        for c_ais in _generar_candidatos_aislamiento(n_vars, k):
            if c_ais not in candidatos_geo:
                candidatos_geo.append(c_ais)

        if permitir_presente_vacio:
            for c_vac in _generar_candidatos_presente_vacio(n_vars, k):
                if c_vac not in candidatos_geo:
                    candidatos_geo.append(c_vac)

        if not candidatos_geo:
            # Fallback jerárquico bottom-up cuando no hay candidatos
            particiones: List[List[int]] = [[i] for i in range(n_vars)]

            def evaluar_fusion(i, j, particiones_actuales):
                nueva_parte = particiones_actuales[i] + particiones_actuales[j]
                particion_prueba = tuple(
                    [particiones_actuales[p] for p in range(len(particiones_actuales))
                     if p != i and p != j]
                    + [nueva_parte]
                )
                perdida = evaluar_k_particion(
                    subsistema, indices_ncubos, dims_ncubos,
                    particion_prueba, dists_marginales,
                )
                return (perdida, i, j, nueva_parte)

            while len(particiones) > k:
                n_partes = len(particiones)
                pares = [(i, j) for i in range(n_partes)
                         for j in range(i + 1, n_partes)]
                if not pares:
                    break
                resultados_fb = Parallel(n_jobs=n_jobs_internos, prefer="threads")(
                    delayed(evaluar_fusion)(i, j, particiones) for i, j in pares
                )
                _, i_idx, j_idx, mejor_union = min(resultados_fb, key=lambda x: x[0])
                nueva_lista = [
                    particiones[p] for p in range(len(particiones))
                    if p != i_idx and p != j_idx
                ]
                nueva_lista.append(mejor_union)
                particiones = nueva_lista

            mejor_particion = tuple(particiones)
            mejor_perdida = evaluar_k_particion(
                subsistema, indices_ncubos, dims_ncubos,
                mejor_particion, dists_marginales,
            )

        else:
            # ── Fase 2: evaluación EMD en paralelo ───────────────────────────
            perdidas_candidatos = Parallel(
                n_jobs=min(len(candidatos_geo), n_jobs_internos),
                prefer="threads",
            )(
                delayed(evaluar_k_particion)(
                    subsistema, indices_ncubos, dims_ncubos,
                    candidato, dists_marginales,
                )
                for candidato in candidatos_geo
            )
            mejor_idx = int(np.argmin(perdidas_candidatos))
            mejor_perdida = float(perdidas_candidatos[mejor_idx])
            mejor_particion = candidatos_geo[mejor_idx]

            # ── Fase 3: refinamiento local 1-move ────────────────────────────
            tiempo_refinar = (
                None if tiempo_maximo_segundos is None
                else max(0.0, tiempo_maximo_segundos * 0.20)
            )
            mejor_perdida, mejor_particion = _refinar_particion_local(
                subsistema,
                indices_ncubos,
                dims_ncubos,
                mejor_particion,
                dists_marginales,
                permitir_presente_vacio=permitir_presente_vacio,
                tiempo_maximo_segundos=tiempo_refinar,
            )

            # ── Fase 4: Búsqueda Local Iterada (ILS) ─────────────────────────
            # Perturba el óptimo local y vuelve a refinar; conserva el mejor.
            n_perturb = max(1, n_vars // 3)
            tiempo_ils_por_iter = (
                None if tiempo_maximo_segundos is None
                else max(0.0, tiempo_maximo_segundos * 0.15)
            )
            for iteracion_ils in range(N_ILS):
                particion_perturbada = _perturbacion_aleatoria(
                    mejor_particion,
                    n_movimientos=n_perturb,
                    semilla=42 + iteracion_ils * 17,
                )
                perdida_ils, particion_ils = _refinar_particion_local(
                    subsistema,
                    indices_ncubos,
                    dims_ncubos,
                    particion_perturbada,
                    dists_marginales,
                    permitir_presente_vacio=permitir_presente_vacio,
                    tiempo_maximo_segundos=tiempo_ils_por_iter,
                )
                if perdida_ils + 1e-12 < mejor_perdida:
                    mejor_perdida = perdida_ils
                    mejor_particion = particion_ils

        # ── Phase 5: asymmetric candidates from tabla_T (k=2 only) ──────────
        # Evaluated AFTER the full symmetric pipeline so that the symmetric
        # pipeline always provides a valid lower bound.  If any asymmetric cut
        # beats it, we record the cut and update mejor_perdida.  The result dict
        # carries the raw (f1, p1, f2, p2) real-index cut so the caller can
        # reconstruct Phi correctly (present sets differ from symmetric intersect).
        mejor_asimetrico: Optional[Tuple] = None
        if k == 2 and candidatos_asimetricos:
            for future_side, present_side in candidatos_asimetricos:
                future_other  = [int(x) for x in indices_ncubos if x not in set(future_side)]
                present_other = [int(x) for x in dims_ncubos   if x not in set(present_side)]
                if not future_side or not future_other:
                    continue
                perdida_asim = evaluar_corte_asimetrico(
                    subsistema,
                    future_side,  present_side,
                    future_other, present_other,
                    dists_marginales,
                )
                if perdida_asim + 1e-12 < mejor_perdida:
                    mejor_perdida = perdida_asim
                    mejor_asimetrico = (future_side, present_side,
                                        future_other, present_other)
                    # Convert future split to position format for display/refinement
                    pos_1 = sorted(
                        int(np.where(indices_ncubos == idx)[0][0])
                        for idx in future_side
                    )
                    pos_2 = sorted(
                        int(np.where(indices_ncubos == idx)[0][0])
                        for idx in future_other
                    )
                    mejor_particion = (pos_1, pos_2)

        fmt_pk = fmt_k_particion(mejor_particion, indices_ncubos, dims_ncubos)
        return {
            "k": k,
            "perdida": float(mejor_perdida),
            "particion": mejor_particion,
            "particion_grafica": fmt_pk,
            "error": None,
            "asimetrico": mejor_asimetrico,
        }

    except Exception as e:
        return {
            "k": k,
            "perdida": float("inf"),
            "particion": None,
            "particion_grafica": "",
            "error": str(e),
            "asimetrico": None,
        }


# ── Estrategia greedy top-down (enfoque 20263) ────────────────────────────


def _construir_cut_pool(
    tabla_T: np.ndarray,
    dist_array: np.ndarray,
    idx_origen: int,
    n_dims: int,
    indices_ncubos: np.ndarray,
    dims_ncubos: np.ndarray,
) -> "list[Block]":
    """
    Build the cut pool from the cost table.

    Three families of candidates:
    1. N symmetric isolation cuts ({i}, {present_i}) + complements.
    2. N empty-mechanism isolation cuts ({i}, ∅): isolates future node i with
       NO present mechanism, identical to 20263's single-bit-flip candidates.
       When applied to a block B, yields inside=({i}, ∅) and outside=(B−{i}, B.pre).
    3. Best geometric cut per Hamming level d=1..n_dims//2+1 + complements.

    All entries are (frozenset of global future indices,
                     frozenset of global present indices).
    """
    n = len(indices_ncubos)
    full_mask = (1 << n_dims) - 1
    pool: list = []
    seen: set = set()

    all_eff = frozenset(int(x) for x in indices_ncubos)
    all_pre = frozenset(int(x) for x in dims_ncubos)

    def _add(eff: frozenset, pre: frozenset) -> None:
        if not eff:
            return
        key = (eff, pre)
        if key not in seen:
            seen.add(key)
            pool.append((eff, pre))

    for i in range(n):
        eff = frozenset([int(indices_ncubos[i])])
        pre = frozenset([int(dims_ncubos[i])]) if i < len(dims_ncubos) else frozenset()
        _add(eff, pre)
        _add(all_eff - eff, all_pre - pre)
        # Corte de aislamiento vacío: nodo i sin ningún mecanismo presente.
        # Cuando se aplica a bloque B: inside=({i}, ∅), outside=(B−{i}, B.pre).
        # Equivalente al candidato single-bit de 20263.
        _add(eff, frozenset())

    all_states = np.arange(len(tabla_T), dtype=np.int32)
    for d in range(1, n_dims // 2 + 2):
        mask_d = dist_array == d
        estados_nivel = all_states[mask_d]
        if len(estados_nivel) == 0:
            continue

        best_cost = np.inf
        best_j = -1
        for start in range(0, len(estados_nivel), COST_TABLE_CHUNK_ROWS):
            chunk = estados_nivel[start : start + COST_TABLE_CHUNK_ROWS]
            costs = np.minimum(tabla_T[chunk], tabla_T[chunk ^ full_mask]).sum(axis=1)
            idx_local = int(np.argmin(costs))
            if float(costs[idx_local]) < best_cost:
                best_cost = float(costs[idx_local])
                best_j = int(chunk[idx_local])

        if best_j < 0:
            continue

        flipped = best_j ^ idx_origen
        present_pos = [b for b in range(n_dims) if not (flipped >> b) & 1]
        row_j = tabla_T[best_j]
        row_c = tabla_T[best_j ^ full_mask]
        effects_pos = [b for b in range(n) if row_j[b] <= row_c[b]]

        eff = frozenset(int(indices_ncubos[p]) for p in effects_pos)
        pre = frozenset(int(dims_ncubos[p]) for p in present_pos if p < len(dims_ncubos))
        _add(eff, pre)
        _add(all_eff - eff, all_pre - pre)

    return pool


def evaluar_bloques(
    subsistema,
    bloques: "list[Block]",
    dist_original: np.ndarray,
) -> float:
    """
    Marginal L1 EMD loss for an asymmetric k-partition given as Block list.

    Uses System.particionar() to process all cubes in a single pass, equivalent
    to 20263's k_partition_marginal_distribution. Each cube is marginalized to
    keep only the mechanism dims of its own block, so future and present sets
    are independent across blocks (no symmetric intersection forced).

    Args:
        subsistema   : Conditioned subsystem with particionar() support.
        bloques      : List of (frozenset global futures, frozenset global presents).
        dist_original: Marginal distribution vector of size N.

    Returns:
        float: Sum of marginal L1 differences.
    """
    particiones = [
        (
            np.array(sorted(future_set), dtype=np.int8),
            np.array(sorted(present_set), dtype=np.int8)
            if present_set
            else np.array([], dtype=np.int8),
        )
        for future_set, present_set in bloques
        if future_set
    ]
    if not particiones:
        return float(np.sum(np.abs(dist_original), dtype=np.float64))

    dist_rec = subsistema.particionar(particiones).distribucion_marginal()
    # Acumular la L1 en float64 (como BruteForceKMIP/QNodes). dist_original y
    # dist_rec son float32; sin dtype=float64 np.sum acumularía en float32 y el Φ
    # podría caer ~1e-8 POR DEBAJO del de la fuerza bruta (misma partición, suma
    # menos precisa) — lo que haría parecer que KGeoMIP "mejora" al óptimo exacto.
    return float(np.sum(np.abs(dist_original - dist_rec), dtype=np.float64))


def _mejor_split_bloques(
    subsistema,
    dist_original: np.ndarray,
    bloques: "list[Block]",
    cut_pool: "list[Block]",
    n_jobs: int = 1,
    permitir_presente_vacio: bool = True,
) -> "Optional[tuple[float, list]]":
    """
    Find the best single block split over all (block, cut) combinations.

    For each current block b and each candidate cut c computes:
        inside  = (b.future ∩ c.future, b.present ∩ c.present)
        outside = (b.future − c.future, b.present − c.present)
    and evaluates EMD for the resulting configuration.

    Returns None if no valid split exists.
    """
    configs: list = []
    for position, (eff_block, pre_block) in enumerate(bloques):
        if len(eff_block) + len(pre_block) < 2:
            continue
        for cut_eff, cut_pre in cut_pool:
            inside: Block = (eff_block & cut_eff, pre_block & cut_pre)
            outside: Block = (eff_block - cut_eff, pre_block - cut_pre)
            # Invariante: ningún bloque puede quedar con el futuro (alcance) vacío.
            # Un bloque sin futuro no representa nada y abre la puerta a partes (∅, ∅).
            if not inside[0] or not outside[0]:
                continue
            # Si el mecanismo vacío NO está permitido, ningún bloque puede quedar
            # con el presente vacío (cada parte usa el mecanismo de sus nodos).
            if not permitir_presente_vacio and (not inside[1] or not outside[1]):
                continue
            configs.append(bloques[:position] + [inside, outside] + bloques[position + 1:])

    if not configs:
        return None

    if n_jobs > 1 and len(configs) > 1:
        losses = Parallel(n_jobs=min(len(configs), n_jobs), prefer="threads")(
            delayed(evaluar_bloques)(subsistema, cfg, dist_original)
            for cfg in configs
        )
    else:
        losses = [evaluar_bloques(subsistema, cfg, dist_original) for cfg in configs]

    best_idx = int(np.argmin(losses))
    return float(losses[best_idx]), configs[best_idx]


def _greedy_k_particion(
    subsistema,
    dist_original: np.ndarray,
    cut_pool: "list[Block]",
    k: int,
    n_jobs: int = N_JOBS_INTERNOS,
    permitir_presente_vacio: bool = True,
) -> "tuple[float, list[Block]]":
    """
    Top-down greedy k-partition: start with one block covering the full
    subsystem, then perform k-1 best splits using the shared cut pool.

    At each step the split that minimises total EMD loss is chosen.
    The cut pool is fixed and shared across all k values (built once).

    Args:
        subsistema   : Conditioned subsystem.
        dist_original: Marginal distribution vector.
        cut_pool     : Pre-built list of (future_set, present_set) Block candidates.
        k            : Target number of blocks.
        n_jobs       : Threads for parallel candidate evaluation.

    Returns:
        (loss, list of Block) for the best k-partition found.
    """
    future_universe: frozenset = frozenset(int(x) for x in subsistema.indices_ncubos)
    present_universe: frozenset = frozenset(int(x) for x in subsistema.dims_ncubos)
    bloques: list = [(future_universe, present_universe)]

    while len(bloques) < k:
        result = _mejor_split_bloques(
            subsistema, dist_original, bloques, cut_pool, n_jobs,
            permitir_presente_vacio=permitir_presente_vacio,
        )
        if result is None:
            break
        _, bloques = result

    loss = evaluar_bloques(subsistema, bloques, dist_original)
    return loss, bloques


# ── Refinamiento 1-move sobre bloques asimétricos (AYDA) ─────────────────


def _refinar_bloques_1move(
    subsistema,
    dist_original: np.ndarray,
    bloques: "list[Block]",
    n_jobs: int = 1,
    permitir_presente_vacio: bool = True,
) -> "tuple[float, list[Block]]":
    """
    Refinamiento 1-move sobre listas de bloques asimétricos (fases 3/4 de AYDA).

    Explora dos tipos de movimientos, el segundo exclusivo de AYDA gracias a
    la representación asimétrica Block=(future_set, present_set):

      - Movimiento futuro: traslada un nodo futuro del bloque i al bloque j
        (equivalente conceptual al 1-move en particiones simétricas).
      - Movimiento presente (AYDA único): traslada un nodo del lado presente
        del bloque i al bloque j SIN mover su par futuro. Solo es posible
        porque en AYDA los lados futuro y presente se particionan de forma
        independiente, a diferencia de los cortes simétricos de 20263.

    Itera hasta convergencia local (ningún movimiento mejora la pérdida EMD).

    Args:
        subsistema   : Subsistema condicionado con soporte para particionar().
        dist_original: Vector de distribución marginal original.
        bloques      : Lista de Block = (frozenset futuros, frozenset presentes).
        n_jobs       : Hilos para evaluación paralela con joblib.

    Returns:
        (perdida, bloques_refinados): Mejor pérdida y configuración encontrada.
    """
    mejor_perdida = evaluar_bloques(subsistema, bloques, dist_original)
    mejor_bloques: "list[Block]" = list(bloques)
    mejoro = True

    while mejoro:
        mejoro = False
        k = len(mejor_bloques)
        vecinos: "list[list[Block]]" = []

        # ── Movimientos futuros ────────────────────────────────────────────
        for i, (eff_i, pre_i) in enumerate(mejor_bloques):
            if len(eff_i) <= 1:
                continue  # no vaciar el conjunto futuro del bloque
            for node in eff_i:
                for j in range(k):
                    if i == j:
                        continue
                    eff_j, pre_j = mejor_bloques[j]
                    cfg = list(mejor_bloques)
                    cfg[i] = (eff_i - {node}, pre_i)
                    cfg[j] = (eff_j | {node}, pre_j)
                    vecinos.append(cfg)

        # ── Movimientos presentes (exclusivo AYDA) ─────────────────────────
        # Mueve un nodo del lado presente de i al lado presente de j,
        # sin afectar la asignación futura de ese nodo. Explora el espacio
        # asimétrico imposible en representaciones simétricas como 20263.
        for i, (eff_i, pre_i) in enumerate(mejor_bloques):
            # Si ∅ no está permitido, no vaciar el presente de un bloque.
            if not permitir_presente_vacio and len(pre_i) <= 1:
                continue
            for node in pre_i:
                for j in range(k):
                    if i == j:
                        continue
                    eff_j, pre_j = mejor_bloques[j]
                    cfg = list(mejor_bloques)
                    cfg[i] = (eff_i, pre_i - {node})
                    cfg[j] = (eff_j, pre_j | {node})
                    vecinos.append(cfg)

        if not vecinos:
            break

        if n_jobs > 1 and len(vecinos) > 1:
            perdidas = Parallel(
                n_jobs=min(len(vecinos), n_jobs), prefer="threads"
            )(delayed(evaluar_bloques)(subsistema, cfg, dist_original) for cfg in vecinos)
        else:
            perdidas = [evaluar_bloques(subsistema, cfg, dist_original) for cfg in vecinos]

        idx_mejor = int(np.argmin(perdidas))
        if float(perdidas[idx_mejor]) < mejor_perdida - 1e-9:
            mejor_perdida = float(perdidas[idx_mejor])
            mejor_bloques = vecinos[idx_mejor]
            mejoro = True

    return mejor_perdida, mejor_bloques


def _refinar_bloques_2move(
    subsistema,
    dist_original: np.ndarray,
    bloques: "list[Block]",
    mejor_perdida: float,
    n_jobs: int = 1,
    permitir_presente_vacio: bool = True,
) -> "tuple[float, list[Block], bool]":
    """
    2-move sistemático post-convergencia 1-move (VNS).

    Genera todos los movimientos 1-move válidos sobre los bloques actuales
    y evalúa cada par de movimientos INDEPENDIENTES aplicados simultáneamente.
    Si algún par reduce la pérdida, aplica el mejor y devuelve mejoro=True.

    Un par es inválido si:
      • el segundo movimiento afecta un nodo que el primero ya reubicó
        (detección implícita: el nodo ya no está en el bloque fuente).
      • algún bloque queda con el futuro vacío.

    Returns:
        (perdida, bloques, mejoro) — mejoro=True si se encontró un par que mejora.
    """
    k = len(bloques)

    # Generar todos los movimientos 1-move válidos
    movimientos: list = []
    for i, (eff_i, pre_i) in enumerate(bloques):
        if len(eff_i) > 1:
            for nodo in sorted(eff_i):
                for j in range(k):
                    if j != i:
                        movimientos.append(("f", i, j, nodo))
        for nodo in sorted(pre_i):
            if not permitir_presente_vacio and len(pre_i) <= 1:
                break
            for j in range(k):
                if j != i:
                    movimientos.append(("p", i, j, nodo))

    configs: list = []
    M = len(movimientos)

    for a in range(M):
        tipo_a, i_a, j_a, nodo_a = movimientos[a]
        # Aplicar mov_a desde el estado original
        bl_a = list(bloques)
        eff_ia, pre_ia = bl_a[i_a]
        eff_ja, pre_ja = bl_a[j_a]
        if tipo_a == "f":
            if nodo_a not in eff_ia:
                continue
            bl_a[i_a] = (eff_ia - {nodo_a}, pre_ia)
            bl_a[j_a] = (eff_ja | {nodo_a}, pre_ja)
        else:
            if nodo_a not in pre_ia:
                continue
            bl_a[i_a] = (eff_ia, pre_ia - {nodo_a})
            bl_a[j_a] = (eff_ja, pre_ja | {nodo_a})

        for b in range(a + 1, M):
            tipo_b, i_b, j_b, nodo_b = movimientos[b]
            bl_b = list(bl_a)
            eff_ib, pre_ib = bl_b[i_b]
            eff_jb, pre_jb = bl_b[j_b]
            if tipo_b == "f":
                if nodo_b not in eff_ib or len(eff_ib) <= 1:
                    continue
                bl_b[i_b] = (eff_ib - {nodo_b}, pre_ib)
                bl_b[j_b] = (eff_jb | {nodo_b}, pre_jb)
            else:
                if nodo_b not in pre_ib:
                    continue
                if not permitir_presente_vacio and len(pre_ib) <= 1:
                    continue
                bl_b[i_b] = (eff_ib, pre_ib - {nodo_b})
                bl_b[j_b] = (eff_jb, pre_jb | {nodo_b})

            if any(not eff for eff, _ in bl_b):
                continue
            configs.append(bl_b)

    if not configs:
        return mejor_perdida, bloques, False

    if n_jobs > 1 and len(configs) > 1:
        perdidas = Parallel(n_jobs=min(len(configs), n_jobs), prefer="threads")(
            delayed(evaluar_bloques)(subsistema, cfg, dist_original)
            for cfg in configs
        )
    else:
        perdidas = [evaluar_bloques(subsistema, cfg, dist_original) for cfg in configs]

    idx_mejor = int(np.argmin(perdidas))
    if float(perdidas[idx_mejor]) < mejor_perdida - 1e-9:
        return float(perdidas[idx_mejor]), configs[idx_mejor], True

    return mejor_perdida, bloques, False


def _perturbacion_bloques(
    bloques: "list[Block]",
    n_movimientos: int = 2,
    semilla: int = 42,
    permitir_presente_vacio: bool = True,
) -> "list[Block]":
    """
    Perturba una k-partición de bloques asimétricos moviendo aleatoriamente
    nodos futuros y presentes entre bloques.

    Garantiza que ningún bloque quede con el futuro vacío (invariante crítico).
    Los movimientos futuros y presentes son independientes (asimetría).
    """
    rng = _random_module.Random(semilla)
    fut_listas = [list(eff) for eff, _ in bloques]
    pre_listas = [list(pre) for _, pre in bloques]
    k = len(bloques)

    for _ in range(n_movimientos):
        # Mover un nodo futuro entre bloques
        candidatos_orig = [i for i, f in enumerate(fut_listas) if len(f) > 1]
        if candidatos_orig:
            i = rng.choice(candidatos_orig)
            nodo = rng.choice(fut_listas[i])
            j = rng.choice([x for x in range(k) if x != i])
            fut_listas[i].remove(nodo)
            fut_listas[j].append(nodo)

        # Mover un nodo presente entre bloques (independiente del futuro)
        if permitir_presente_vacio:
            candidatos_pre = [i for i, p in enumerate(pre_listas) if len(p) > 0]
        else:
            candidatos_pre = [i for i, p in enumerate(pre_listas) if len(p) > 1]
        if candidatos_pre:
            i = rng.choice(candidatos_pre)
            if pre_listas[i]:
                nodo = rng.choice(pre_listas[i])
                j = rng.choice([x for x in range(k) if x != i])
                pre_listas[i].remove(nodo)
                pre_listas[j].append(nodo)

    return [(frozenset(f), frozenset(p)) for f, p in zip(fut_listas, pre_listas)]


def fmt_bloques(
    bloques: "list[Block]",
    indices_ncubos: np.ndarray,
    dims_ncubos: np.ndarray,
) -> str:
    """
    Format a Block list as a two-line partition string.

    Upper line (UPPERCASE) = future nodes (t+1) per block.
    Lower line (lowercase) = present nodes (t) per block; ∅ if empty.
    """
    mec_set = set(int(x) for x in dims_ncubos)
    partes_fmt = []
    for future_set, present_set in bloques:
        futuros = sorted(future_set)
        str_fut = ",".join(ABECEDARY[i] for i in futuros) if futuros else VOID_STR
        presentes = sorted(p for p in present_set if p in mec_set)
        str_pres = ",".join(LOWER_ABECEDARY[i] for i in presentes) if presentes else VOID_STR
        ancho = max(len(str_fut), len(str_pres)) + 2
        partes_fmt.append((f"|{str_fut:^{ancho}}|", f"|{str_pres:^{ancho}}|"))
    linea_top = "".join(t for t, _ in partes_fmt)
    linea_bot = "".join(b for _, b in partes_fmt)
    return f"{linea_top}\n{linea_bot}"


def warmup_motor() -> None:
    """
    Precalienta los dos componentes con coste de inicio relevante en GeoMIP:

      1. Kernel Numba _kernel_tabla_costos (si Numba está disponible):
         La primera llamada carga el binario @njit(cache=True) guardado en disco
         (~0.1–0.5 s) o lo compila desde cero si no existe aún (~1–3 s). Las
         llamadas posteriores dentro del mismo proceso no tienen coste adicional.
         Se dispara aquí con arrays mínimos (n_dims=2, n=2) sin efectos secundarios.

      2. Pool de hilos joblib:
         El primer Parallel(..., prefer="threads") lanza los N_JOBS_INTERNOS workers.
         Una corrida trivial aquí evita que la primera prueba del lote pague ese
         coste de creación de hilos (~50–200 ms según número de núcleos).

    Uso recomendado: llamar una vez antes del bucle de pruebas en modo bloque.
    En modo manual el coste se absorbe en la única ejecución sin impacto apreciable.
    """
    if _NUMBA_DISPONIBLE:
        _n, _n_dims = 2, 2
        _total = 1 << _n_dims
        _prob_T      = np.zeros((_total, _n), dtype=np.float32)
        _prob_origen = np.zeros(_n, dtype=np.float32)
        _xor_origen  = np.zeros(_total, dtype=np.uint32)
        _orden       = np.arange(_total, dtype=np.int32)
        _tabla_T     = np.zeros((_total, _n), dtype=np.float32)
        _kernel_tabla_costos(
            _prob_T, _prob_origen, _xor_origen, _orden, _n_dims, _n, _tabla_T
        )

    Parallel(n_jobs=N_JOBS_INTERNOS, prefer="threads")(
        delayed(lambda x: x)(i) for i in range(N_JOBS_INTERNOS * 4)
    )


# ── Solver exacto (fuerza bruta) para N pequeño ─────────────────────────────

# Para N ≤ _KGEOMIP_N_EXACTO se resuelve la k-MIP por enumeración exhaustiva del
# mismo espacio asimétrico que BruteForceKMIP/QNodes, usando evaluar_bloques: así
# KGeoMIP devuelve el óptimo global y coincide con la fuerza bruta y con QNodes en
# CSVs pequeños (deterministas o estocásticos). _CAP acota configuraciones por k.
_KGEOMIP_N_EXACTO: int = 6
_KGEOMIP_CAP_EXACTO: int = 300_000


def _stirling2(n: int, k: int) -> int:
    """Número de Stirling de segunda especie S(n, k) (DP iterativo)."""
    if k < 0 or k > n:
        return 0
    if k == 0:
        return 1 if n == 0 else 0
    fila = [0] * (k + 1)
    fila[0] = 1
    for _ in range(1, n + 1):
        nueva = [0] * (k + 1)
        for j in range(1, k + 1):
            nueva[j] = j * fila[j] + fila[j - 1]
        fila = nueva
    return fila[k]


def _particiones_en_k(items: list, k: int):
    """
    Genera todas las particiones de `items` en EXACTAMENTE `k` bloques no vacíos
    (S(len(items), k) particiones). Esquema recursivo clásico: cada elemento entra
    en un bloque existente o abre uno nuevo, exigiendo k bloques al final.
    """
    n = len(items)
    if k < 1 or k > n:
        return

    def _rec(idx: int, bloques: "list[list]"):
        if idx == n:
            if len(bloques) == k:
                yield [b[:] for b in bloques]
            return
        restantes = n - idx
        elemento = items[idx]
        if len(bloques) + (restantes - 1) >= k:
            for i in range(len(bloques)):
                bloques[i].append(elemento)
                yield from _rec(idx + 1, bloques)
                bloques[i].pop()
        if len(bloques) < k:
            bloques.append([elemento])
            yield from _rec(idx + 1, bloques)
            bloques.pop()

    yield from _rec(0, [])


def _resolver_exacto_geomip(
    subsistema,
    dist_original: np.ndarray,
    k: Optional[int],
    permitir_vacio: bool,
) -> "Optional[tuple[float, list, int]]":
    """
    Enumera exhaustivamente las k-particiones asimétricas válidas y devuelve
    (Φ mínimo, bloques globales, k) — réplica del espacio de BruteForceKMIP usando
    el mismo evaluar_bloques de KGeoMIP, así el óptimo es idéntico a la fuerza bruta.

    Cada bloque se construye con índices GLOBALES (valores de indices_ncubos /
    dims_ncubos), como espera evaluar_bloques. Si el espacio de algún k requerido
    supera _KGEOMIP_CAP_EXACTO devuelve None (el llamador usa la heurística).
    """
    idx = subsistema.indices_ncubos
    dims = subsistema.dims_ncubos
    n_vars = len(idx)
    n_dims = len(dims)

    ks = [k] if k is not None else list(range(2, n_vars + 1))
    for kk in ks:
        if _stirling2(n_vars, kk) * (kk ** n_dims) > _KGEOMIP_CAP_EXACTO:
            return None

    futuros = list(range(n_vars))
    mejor_por_k: dict[int, tuple[float, list]] = {}

    for kk in ks:
        mejor_phi = float("inf")
        mejor_bloques: Optional[list] = None
        for part_fut in _particiones_en_k(futuros, kk):
            bloques_fut = [frozenset(int(idx[p]) for p in b) for b in part_fut]
            if n_dims == 0:
                bloques = [(bloques_fut[i], frozenset()) for i in range(kk)]
                phi = evaluar_bloques(subsistema, bloques, dist_original)
                if phi < mejor_phi:
                    mejor_phi, mejor_bloques = phi, bloques
                continue
            for asignacion in itertools.product(range(kk), repeat=n_dims):
                pre_sets: list[list[int]] = [[] for _ in range(kk)]
                for pos_pre, b_idx in enumerate(asignacion):
                    pre_sets[b_idx].append(int(dims[pos_pre]))
                if not permitir_vacio and any(len(s) == 0 for s in pre_sets):
                    continue
                bloques = [
                    (bloques_fut[i], frozenset(pre_sets[i])) for i in range(kk)
                ]
                phi = evaluar_bloques(subsistema, bloques, dist_original)
                if phi < mejor_phi:
                    mejor_phi, mejor_bloques = phi, bloques
        if mejor_bloques is not None:
            mejor_por_k[kk] = (mejor_phi, mejor_bloques)

    if not mejor_por_k:
        return None
    if k is not None:
        r = mejor_por_k.get(k)
        return (r[0], r[1], k) if r is not None else None
    # k libre: preferir k ≥ 3 si existe (coherente con la rama heurística).
    candidatos = {kk: v for kk, v in mejor_por_k.items() if kk >= 3} or mejor_por_k
    mejor_k = min(candidatos, key=lambda kk: candidatos[kk][0])
    return (candidatos[mejor_k][0], candidatos[mejor_k][1], mejor_k)


# ── Clase principal ────────────────────────────────────────────────────────

class KGeoMIP(SIA):
    """
    Extensión de GeoMIP para k-particiones (k ≥ 2).

    Para cualquier k ≥ 2 usa un pipeline heurístico completo:
    SpectralClustering + AgglomerativeClustering (múltiples semillas y
    linkage) + candidatos de aislamiento + refinamiento 1-move +
    Búsqueda Local Iterada (ILS) con perturbaciones aleatorias.

    Args:
        gestor (Manager): Gestor con el estado inicial y ruta de la TPM.
        k      (int)    : Número de partes de la partición (default=2).

    Attributes:
        k                   : Número de partes solicitado.
        tabla_transiciones  : Tabla de costos de transición para optimizar
                              el costo EMD evaluado.
        memoria_particiones : Diccionario con las pérdidas evaluadas.
    """

    def __init__(self, gestor: Manager, k: int = 2) -> None:
        super().__init__(gestor)
        if k < 2:
            raise ValueError(f"k debe ser ≥ 2, recibido k={k}.")
        self.k = k
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}_k{k}"
        )
        self.logger = SafeLogger(KGEOMIP_STRATEGY_TAG)
        self.tabla_transiciones: dict = {}
        self.memoria_particiones: Dict[tuple, Tuple[float, np.ndarray]] = {}
        self.historico_particiones: list = []

    # ── Método principal ───────────────────────────────────────────────────

    @profile(context={TYPE_TAG: KGEOMIP_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        k: Optional[int] = None,
        permitir_presente_vacio: bool = PERMITIR_PRESENTE_VACIO_POR_DEFECTO,
        tiempo_maximo_segundos: Optional[float] = None,
    ) -> Solution:
        """
        Encuentra la Participación de Mínima Información ÓPTIMA GLOBAL (Optimal K-MIP).
        Itera sobre todos los k (desde 2 hasta N) y elige la partición
        con la mínima pérdida EMD para cumplir con el requisito formal del proyecto.

        Para N > 20, utiliza automáticamente cálculo de EMD optimizado con muestreo
        para evitar colapso de memoria.
        """
        self.memoria_particiones.clear()
        self.logger.critic("Iniciando KGeoMIP con bucle de k PARALELIZADO.")
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        n_vars = len(self.sia_subsistema.indices_ncubos)

        # ── Validar k si fue especificado por el usuario ───────────────────
        if k is not None:
            if k < 2 or k > n_vars:
                raise ValueError(
                    f"Error: K={k} está fuera del rango permitido [2, {n_vars}]"
                )
            self.logger.critic(f"K especificado por el usuario: K={k}")

        if n_vars > 20:
            self.logger.critic(
                f"⚠️  MODO OPTIMIZADO ACTIVADO: N={n_vars} variables.\n"
                f"   EMD con muestreo para N > 20."
            )

        # ── Caso trivial ───────────────────────────────────────────────────
        if n_vars <= 1:
            future_universe: frozenset = frozenset(
                int(x) for x in self.sia_subsistema.indices_ncubos
            )
            present_universe: frozenset = frozenset(
                int(x) for x in self.sia_subsistema.dims_ncubos
            )
            bloques_optimos: list = [(future_universe, present_universe)]
            k_optimo = 1
            mejor_perdida = 0.0
            dist_reconstruida = self.sia_dists_marginales.copy()

        elif n_vars <= _KGEOMIP_N_EXACTO and (
            _exacto := _resolver_exacto_geomip(
                self.sia_subsistema,
                self.sia_dists_marginales,
                k,
                permitir_presente_vacio,
            )
        ) is not None:
            # Ruta EXACTA: óptimo global por enumeración (coincide con fuerza bruta).
            mejor_perdida, bloques_optimos, k_optimo = _exacto
            self.logger.critic(
                f"Ruta EXACTA (N={n_vars} ≤ {_KGEOMIP_N_EXACTO}): "
                f"k óptimo={k_optimo}, pérdida={mejor_perdida:.6f}"
            )
            self.historico_particiones = [{
                "k": k_optimo,
                "perdida": mejor_perdida,
                "particion_grafica": fmt_bloques(
                    bloques_optimos,
                    self.sia_subsistema.indices_ncubos,
                    self.sia_subsistema.dims_ncubos,
                ),
            }]
            indices = self.sia_subsistema.indices_ncubos
            dist_reconstruida = np.empty(len(self.sia_dists_marginales), dtype=np.float32)
            for future_set, present_set in bloques_optimos:
                if not future_set:
                    continue
                futuros = np.array(sorted(future_set), dtype=np.int8)
                presentes = (
                    np.array(sorted(present_set), dtype=np.int8)
                    if present_set
                    else np.array([], dtype=np.int8)
                )
                dist_parte = (
                    self.sia_subsistema.bipartir(futuros, presentes)
                    .distribucion_marginal()
                )
                for idx in future_set:
                    pos = np.where(indices == idx)[0]
                    if pos.size:
                        dist_reconstruida[int(pos[0])] = dist_parte[int(pos[0])]

        else:
            self._construir_tabla_costos()
            cut_pool = _construir_cut_pool(
                self.tabla_T,
                self._dist_array,
                self._idx_origen,
                self._n_dims,
                self.sia_subsistema.indices_ncubos,
                self.sia_subsistema.dims_ncubos,
            )
            self.logger.critic(f"Cut pool: {len(cut_pool)} candidatos.")

            valores_k = (
                [k] if k is not None
                else list(range(2, min(6, n_vars + 1)))
            )
            self.logger.critic(
                f"Evaluando K={valores_k} con estrategia greedy top-down."
            )

            resultados_k: list = []
            for test_k in valores_k:
                try:
                    perdida_k, bloques_k = _greedy_k_particion(
                        self.sia_subsistema,
                        self.sia_dists_marginales,
                        cut_pool,
                        test_k,
                        permitir_presente_vacio=permitir_presente_vacio,
                    )
                    self.logger.critic(
                        f"  Greedy K={test_k} → pérdida={perdida_k:.6f}"
                    )

                    # ── Fase 3: refinamiento 1-move (documentado AYDA, ausente en 20263) ──
                    # Incluye movimientos presentes asimétricos: mueve un nodo del
                    # lado presente sin tocar su par futuro, imposible en 20263.
                    perdida_k, bloques_k = _refinar_bloques_1move(
                        self.sia_subsistema,
                        self.sia_dists_marginales,
                        bloques_k,
                        n_jobs=N_JOBS_INTERNOS,
                        permitir_presente_vacio=permitir_presente_vacio,
                    )
                    self.logger.critic(
                        f"  +1-move K={test_k} perdida={perdida_k:.6f}"
                    )

                    # ── VNS: ciclos 2-move + 1-move hasta convergencia ────────
                    # Escapa de mínimos locales 1-move evaluando pares de
                    # movimientos simultáneos; máx 3 ciclos para acotar el tiempo.
                    N_VNS_MAX = 0
                    for _vns_i in range(N_VNS_MAX):
                        perdida_2m, bloques_2m, mejoro_2m = _refinar_bloques_2move(
                            self.sia_subsistema,
                            self.sia_dists_marginales,
                            bloques_k,
                            perdida_k,
                            n_jobs=N_JOBS_INTERNOS,
                            permitir_presente_vacio=permitir_presente_vacio,
                        )
                        if not mejoro_2m:
                            break
                        perdida_k, bloques_k = _refinar_bloques_1move(
                            self.sia_subsistema,
                            self.sia_dists_marginales,
                            bloques_2m,
                            n_jobs=N_JOBS_INTERNOS,
                            permitir_presente_vacio=permitir_presente_vacio,
                        )
                        self.logger.critic(
                            f"  VNS[{_vns_i}] 2-move+1-move → {perdida_k:.6f}"
                        )

                    # ── ILS ligero: 2 reinicios con perturbación aleatoria ────
                    # Escapa de mínimos locales del greedy top-down sin sacrificar
                    # demasiado tiempo; 2 iteraciones con semillas distintas.
                    N_ILS_LIGHT = 0
                    n_perturb = max(1, n_vars // 3)
                    for _ils_i in range(N_ILS_LIGHT):
                        bloques_pert = _perturbacion_bloques(
                            bloques_k,
                            n_movimientos=n_perturb,
                            semilla=37 + _ils_i * 19,
                            permitir_presente_vacio=permitir_presente_vacio,
                        )
                        perdida_pert, bloques_pert = _refinar_bloques_1move(
                            self.sia_subsistema,
                            self.sia_dists_marginales,
                            bloques_pert,
                            n_jobs=N_JOBS_INTERNOS,
                            permitir_presente_vacio=permitir_presente_vacio,
                        )
                        if perdida_pert < perdida_k - 1e-9:
                            perdida_k = perdida_pert
                            bloques_k = bloques_pert
                            self.logger.critic(
                                f"  ILS[{_ils_i}] mejoró → {perdida_k:.6f}"
                            )

                    fmt_pk = fmt_bloques(
                        bloques_k,
                        self.sia_subsistema.indices_ncubos,
                        self.sia_subsistema.dims_ncubos,
                    )
                    self.logger.critic(f"  ✓ K={test_k} → pérdida final={perdida_k:.6f}")
                    resultados_k.append({
                        "k": test_k,
                        "perdida": perdida_k,
                        "bloques": bloques_k,
                        "particion_grafica": fmt_pk,
                        "error": None,
                    })
                except Exception as exc:
                    self.logger.critic(f"  ✗ K={test_k} excepción: {exc}")
                    resultados_k.append({
                        "k": test_k,
                        "perdida": float("inf"),
                        "bloques": None,
                        "particion_grafica": "",
                        "error": str(exc),
                    })

            self.historico_particiones = []
            mejor_perdida = float("inf")
            bloques_optimos = None
            k_optimo = None

            for res in sorted(resultados_k, key=lambda r: r["k"]):
                self.historico_particiones.append({
                    "k": res["k"],
                    "perdida": res["perdida"],
                    "particion_grafica": res["particion_grafica"],
                })
                if res["perdida"] < mejor_perdida and res["bloques"] is not None:
                    mejor_perdida = res["perdida"]
                    bloques_optimos = res["bloques"]
                    k_optimo = res["k"]

            if bloques_optimos is None:
                raise RuntimeError(
                    "Todos los valores de k fallaron. Revisa los logs."
                )

            # Reconstruct marginal distribution from winning partition
            indices = self.sia_subsistema.indices_ncubos
            dist_reconstruida = np.empty(len(self.sia_dists_marginales), dtype=np.float32)
            for future_set, present_set in bloques_optimos:
                if not future_set:
                    continue
                futuros = np.array(sorted(future_set), dtype=np.int8)
                presentes = (
                    np.array(sorted(present_set), dtype=np.int8)
                    if present_set
                    else np.array([], dtype=np.int8)
                )
                dist_parte = (
                    self.sia_subsistema.bipartir(futuros, presentes)
                    .distribucion_marginal()
                )
                for idx in future_set:
                    pos = np.where(indices == idx)[0]
                    if pos.size:
                        dist_reconstruida[int(pos[0])] = dist_parte[int(pos[0])]

        fmt = fmt_bloques(
            bloques_optimos,
            self.sia_subsistema.indices_ncubos,
            self.sia_subsistema.dims_ncubos,
        )

        self.logger.critic(
            f"ÓPTIMA k-MIP (k={k_optimo}) pérdida={mejor_perdida:.6f}:\n{fmt}"
        )

        tiempo_busqueda = time.time() - self.sia_tiempo_inicio
        tiempo_prep = getattr(self, "sia_tiempo_preparacion", 0.0)

        return Solution(
            estrategia=f"{KGEOMIP_LABEL}(Greedy k={k_optimo})",
            perdida=mejor_perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_reconstruida,
            particion=fmt,
            tiempo_total=tiempo_busqueda + tiempo_prep,
            tiempo_preparacion=tiempo_prep,
        )
    # ── Construcción de tabla de costos ─────────


    def _construir_tabla_costos(self) -> None:
        """
        Populate self.tabla_T, the (2**n_dims, n) float32 transition-cost table.

        n      = len(indices_ncubos) = number of future/output n-cubes.
        n_dims = len(dims_ncubos)    = dimensionality of the joint state space
                                       (present + future active dimensions).

        Each n-cube column holds 2^n_dims probabilities indexed by the joint
        present state, so the table spans 2^n_dims rows (not 2^n).

        Recurrence (filled by increasing Hamming shell d around the origin):
            local[j, x]    = |P[x][j] - P[x][origin]|
            vecinal[j, x]  = Σ_{b : bit b of (j⊕origin) is set} tabla_T[j ^ 2^b, x]
            tabla_T[j, x]  = (local[j, x] + vecinal[j, x]) · 2^(-d)
        Only the bits that differ from the origin contribute to vecinal, and
        each contribution reuses a row at the previous shell (already filled).

        Persists: self.tabla_T, self._idx_origen, self._full_mask,
        self._dist_array, self._n_dims, self._estado_inicial_tabla.

        Complexity: O(n_dims · 2^n_dims) time, O(2^n_dims · n) space.
        """
        subsistema = self.sia_subsistema
        n = len(subsistema.indices_ncubos)       # output variables (n-cubes)
        n_dims = len(subsistema.dims_ncubos)     # joint state-space dimensions
        total_states = 1 << n_dims

        if n_dims > 20:
            self.logger.critic(
                f"WARNING: n_dims={n_dims} > 20. "
                f"tabla_T will require ~{total_states * n * 4 / 1e9:.2f} GB float32."
            )

        # Per-node probability rows P[x][s] = P(variable x = 1 | joint state s).
        prob = np.array(
            [ncubo.data.ravel() for ncubo in subsistema.ncubos],
            dtype=np.float32,
        )

        # Origin = little-endian packing of the initial state over the mechanism dims.
        estado_ini = subsistema.estado_inicial[subsistema.dims_ncubos]
        idx_origen = 0
        for pos, bit in enumerate(estado_ini):
            if bit:
                idx_origen |= 1 << pos
        prob_origen = prob[:, idx_origen]        # (n,)

        estados = np.arange(total_states, dtype=np.int32)
        xor_origen = (estados ^ idx_origen).astype(np.uint32)
        dist = _popcount_vec(xor_origen)

        self.tabla_T = np.zeros((total_states, n), dtype=np.float32)

        if _NUMBA_DISPONIBLE:
            # Ruta JIT: un kernel njit recorre los estados en orden de popcount
            # ascendente y aplica la misma recurrencia (resultado equivalente a la
            # ruta numpy). prob_T contiguo por filas para acceso cache-friendly.
            prob_T = np.ascontiguousarray(prob.T)                 # (total_states, n)
            prob_origen_c = np.ascontiguousarray(prob_origen)     # (n,)
            orden = np.argsort(dist, kind="stable").astype(np.int32)
            _kernel_tabla_costos(
                prob_T, prob_origen_c, xor_origen, orden,
                int(n_dims), int(n), self.tabla_T,
            )
        else:
            bit_pos = range(n_dims)

            # Walk the Hamming shells outward; shell d only depends on shell d-1.
            for d in range(1, n_dims + 1):
                shell = np.flatnonzero(dist == d)
                if shell.size == 0:
                    continue

                gamma = np.float32(1.0 / (1 << d))

                for ini in range(0, shell.size, COST_TABLE_CHUNK_ROWS):
                    bloque = shell[ini : ini + COST_TABLE_CHUNK_ROWS]

                    local = np.abs(prob[:, bloque].T - prob_origen)   # (m, n)

                    if d == 1:
                        # Sole differing bit has its neighbor at the (zero) origin.
                        self.tabla_T[bloque] = local * gamma
                        continue

                    vecinal = np.zeros_like(local)
                    xor_bloque = xor_origen[bloque]
                    for b in bit_pos:                  # ascending bit order (fixed)
                        difiere = ((xor_bloque >> np.uint32(b)) & np.uint32(1)).astype(bool)
                        if not difiere.any():
                            continue
                        vecino = (bloque ^ np.int32(1 << b))[difiere]
                        vecinal[difiere] += self.tabla_T[vecino]

                    self.tabla_T[bloque] = (local + vecinal) * gamma

        assert self.tabla_T.shape == (total_states, n)

        self._idx_origen: int = idx_origen
        self._full_mask: int = total_states - 1
        self._dist_array: np.ndarray = dist
        self._n_dims: int = n_dims
        self._estado_inicial_tabla: list = estado_ini.tolist()

        # Empty dict retained for backward compatibility with external references.
        self.tabla_transiciones = {}

        self._construir_matriz_afinidad()


    def _calcular_costo(
        self,
        estado_ini: list,
        estado_fin: list,
        idx_ncubos: list,
    ) -> None:
        """
        Calcula t_X(i,j) para TODOS los n-cubos X y lo almacena en
        self.tabla_transiciones[(tuple(i), tuple(j))].

        Fórmula:
            t_X(i,j) = γ · ( |X[i] - X[j]| + Σ_{k ∈ N(i,j)} t_X(i,k) )

        donde:
            γ = 2^(-dH(i,j))           ← factor de decrecimiento exponencial
            N(i,j) = vecinos de j en los caminos óptimos desde i hacia j
        """
        clave = (tuple(estado_ini), tuple(estado_fin))
        if clave in self.tabla_transiciones:
            return   # ya calculado, memoización

        ini = np.array(estado_ini)
        fin = np.array(estado_fin)

        diff = ini != fin
        dH = int(diff.sum())
        gamma = 2.0 ** (-dH)

        def estado_a_indice(estado):
            return int(sum(b * (2 ** k) for k, b in enumerate(estado)))

        idx_ini = estado_a_indice(estado_ini)
        idx_fin = estado_a_indice(estado_fin)

        costos = [
            abs(float(self._flat_data[c][idx_ini]) - float(self._flat_data[c][idx_fin]))
            for c in idx_ncubos
        ]

        posiciones_diferentes = np.where(diff)[0]
        for pos in posiciones_diferentes:
            vecino = fin.copy()
            vecino[pos] = ini[pos]      # "deshacer" el flip en la posición pos
            clave_vecino = (tuple(estado_ini), tuple(vecino))

            if clave_vecino in self.tabla_transiciones:
                costos_vecino = self.tabla_transiciones[clave_vecino]
                for c in idx_ncubos:
                    costos[c] += costos_vecino[c]

        costos_finales = [gamma * costo for costo in costos]
        self.tabla_transiciones[clave] = costos_finales

    # ── Matriz de afinidad geométrica ─────────────────────────────────────

    def _construir_matriz_afinidad(self) -> None:
        """
        Build the N×N geometric affinity matrix from self.tabla_T.

        Delegates to _construir_matriz_afinidad_desde_tabla with the full
        (2^N, N) cost table so every variable's column profile spans the
        entire hypercube rather than just the N-state BFS path.

        Result stored in self._matriz_afinidad for _evaluar_k_completo.
        """
        n = len(self.sia_subsistema.indices_ncubos)
        self._matriz_afinidad: np.ndarray = _construir_matriz_afinidad_desde_tabla(
            self.tabla_T,
            n,
        )
        self.logger.critic(
            f"Affinity matrix {n}×{n} built from full tabla_T "
            f"(sklearn={'available' if _SKLEARN_DISPONIBLE else 'not available'})."
        )

    # ── Candidatos asimétricos desde tabla_T ─────────────────────────────────

    def _candidatos_desde_tabla_T(
        self,
    ) -> "List[Tuple[List[int], List[int]]]":
        """
        Generate asymmetric bipartition candidates from the full cost table.

        Unlike _generar_candidatos_hipercubo_completo (which produces symmetric
        cuts where future and present are derived from the same bit-flip set),
        this method separates the two axes:

          - present_pos: positions (in dims_ncubos) corresponding to bits that
            did NOT change when moving from idx_origen to j* — these represent
            the nodes that remain on the "present/mechanism" side.
          - effects_pos: positions (in indices_ncubos) where j*'s cost profile is
            cheaper than its complement — these represent the future/alcance side.

        The two sets are derived from independent criteria and can be different,
        producing cuts where each block has its own (future, present) specification.

        Algorithm:
          For each Hamming level d = 1..(n_dims//2 + 1):
            j* = argmin_j  min(tabla_T[j], tabla_T[j ^ full_mask]).sum()
            flipped = j* XOR idx_origen
            present_pos = bits that did NOT change: {b : not (flipped >> b) & 1}
            effects_pos = output vars cheaper at j*: {b : tabla_T[j*][b] <= tabla_T[comp][b]}
            Convert positions to real node indices using indices_ncubos / dims_ncubos.
            Emit both the primary side and its complement as separate candidates.

        Returns:
            List of (future_real_indices, present_real_indices) tuples.
            Indices are real global node labels (values of indices_ncubos / dims_ncubos),
            NOT local position integers.
        """
        n      = len(self.sia_subsistema.indices_ncubos)   # number of future n-cubes
        n_dims = self._n_dims                               # joint state-space dimensionality
        indices = self.sia_subsistema.indices_ncubos        # real future node indices
        dims    = self.sia_subsistema.dims_ncubos           # real present node indices

        idx_origen = self._idx_origen
        full_mask  = self._full_mask
        dist       = self._dist_array
        total_states = len(self.tabla_T)
        all_states   = np.arange(total_states, dtype=np.int32)

        candidates: list = []
        seen: set        = set()

        for d in range(1, n_dims // 2 + 2):
            mask_d = dist == d
            estados_nivel = all_states[mask_d]
            if len(estados_nivel) == 0:
                continue

            # Find j* = argmin symmetric cost over all states at this Hamming level.
            best_cost = np.inf
            best_j    = -1
            for start in range(0, len(estados_nivel), COST_TABLE_CHUNK_ROWS):
                chunk      = estados_nivel[start : start + COST_TABLE_CHUNK_ROWS]
                current    = self.tabla_T[chunk]
                complement = self.tabla_T[chunk ^ full_mask]
                costs      = np.minimum(current, complement).sum(axis=1)
                idx_local  = int(np.argmin(costs))
                if float(costs[idx_local]) < best_cost:
                    best_cost = float(costs[idx_local])
                    best_j    = int(chunk[idx_local])

            if best_j < 0:
                continue

            flipped = best_j ^ idx_origen

            # present_pos: positions (in dims) whose corresponding state bit did NOT
            # change moving from origin to j*.  Iterates over n_dims since flipped
            # is an n_dims-bit integer.
            present_pos = [b for b in range(n_dims) if not (flipped >> b) & 1]

            # effects_pos: output variable positions (in indices_ncubos) where
            # the cost at j* is cheaper than the cost at j*'s bit complement.
            row_j = self.tabla_T[best_j]
            row_c = self.tabla_T[best_j ^ full_mask]
            effects_pos = [b for b in range(n) if row_j[b] <= row_c[b]]

            # Convert positions to real global node indices.
            future_side  = [int(indices[p]) for p in effects_pos]
            present_side = [int(dims[p])    for p in present_pos if p < len(dims)]

            # Complement: nodes NOT on each side.
            future_set  = set(future_side)
            present_set = set(present_side)
            future_other  = [int(x) for x in indices if x not in future_set]
            present_other = [int(x) for x in dims    if x not in present_set]

            # Emit both the primary side and its complement as separate candidates.
            for fs, ps in [(future_side, present_side), (future_other, present_other)]:
                if not fs:   # future side must be non-empty for a valid block
                    continue
                key = (tuple(sorted(fs)), tuple(sorted(ps)))
                if key not in seen:
                    seen.add(key)
                    candidates.append((list(fs), list(ps)))

        return candidates

    # ── Estrategia: heurística jerárquica bottom-up (fallback de clase) ──────

    def _agrupamiento_jerarquico(
        self, k: int
    ) -> Tuple[List[int], ...]:
        """
        Algoritmo de Agrupamiento Jerárquico basado en Costos de Discrepancia (Bottom-Up).

        Empieza con n particiones (cada variable sola). Iterativamente fusiona los dos
        subsistemas cuya unión genere la MENOR pérdida de información EMD respecto
        al sistema original, hasta alcanzar el número k de particiones deseado.

        Args:
            k: Número de partes objetivo.

        Returns:
            Tupla de k listas de índices.
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)

        if k > n_vars:
            raise ValueError(f"k={k} no puede ser mayor que n_vars={n_vars}")
        if k < 1:
            raise ValueError(f"k={k} debe ser al menos 1")

        particiones: List[List[int]] = [[i] for i in range(n_vars)]

        def evaluar_fusion(i, j, particiones_actuales):
            nueva_parte = particiones_actuales[i] + particiones_actuales[j]
            particion_prueba = tuple(
                [particiones_actuales[p] for p in range(len(particiones_actuales)) if p != i and p != j]
                + [nueva_parte]
            )

            perdida = evaluar_k_particion(
                self.sia_subsistema,
                self.sia_subsistema.indices_ncubos,
                self.sia_subsistema.dims_ncubos,
                particion_prueba,
                self.sia_dists_marginales,
            )
            return (perdida, i, j, nueva_parte)

        iteracion_num = 0
        while len(particiones) > k:
            iteracion_num += 1
            n_partes = len(particiones)
            pares_a_evaluar = [(i, j) for i in range(n_partes) for j in range(i + 1, n_partes)]

            if not pares_a_evaluar:
                self.logger.critic(f"  ADVERTENCIA: No hay pares a evaluar pero len(particiones)={n_partes} > k={k}")
                break

            resultados = Parallel(n_jobs=N_JOBS_INTERNOS, prefer="threads")(
                delayed(evaluar_fusion)(i, j, particiones) for i, j in pares_a_evaluar
            )

            mejor_perdida, i_idx, j_idx, mejor_union = min(resultados, key=lambda x: x[0])

            nueva_lista_particiones = [
                particiones[p] for p in range(n_partes) if p != i_idx and p != j_idx
            ]
            nueva_lista_particiones.append(mejor_union)
            particiones = nueva_lista_particiones

            self.logger.critic(
                f"Jerárquico (Bottom-Up): Fusionadas partes {(i_idx, j_idx)} -> {mejor_union}. "
                f"Particiones restantes: {len(particiones)}, Pérdida: {mejor_perdida:.6f}"
            )

        self.logger.critic(f"  → Agrupamiento jerárquico para k={k} TERMINADO (iteraciones: {iteracion_num})")
        resultado = tuple(particiones)
        return resultado
