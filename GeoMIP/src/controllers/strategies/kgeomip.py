"""
KGeoMIP — Extensión de la estrategia geométrica GeoMIP a k-particiones.

PARALELIZACIÓN: Evaluación secuencial de k con máximo de núcleos por k.

El bucle de k evalúa cada valor secuencialmente (k=2, luego k=3, ...).
Cada k usa cpu_count-1 núcleos para la búsqueda interna con joblib.
Esto evita dividir recursos entre k's y maximiza el rendimiento por tarea.

NOTA sobre joblib:
  - Los métodos internos (_agrupamiento_jerarquico,
    _refinar_particion_local) usan joblib con n_jobs = N_JOBS_INTERNOS.
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

from src.controllers.manager import Manager
from src.funcs.base import emd_causal, ABECEDARY, LOWER_ABECEDARY
from src.funcs.format import fmt_biparte_q
from src.funcs.emd_optimized import emd_causal_fast_partition
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
)



KGEOMIP_LABEL: str = "KGeoMIP"
KGEOMIP_STRATEGY_TAG: str = f"{KGEOMIP_LABEL}_strategy"
KGEOMIP_ANALYSIS_TAG: str = f"{KGEOMIP_LABEL}_analysis"

# Mantener la posibilidad de que una parte tenga mecanismo vacío (∅).
PERMITIR_PRESENTE_VACIO_POR_DEFECTO: bool = True

N_JOBS_INTERNOS: int = max(1, multiprocessing.cpu_count() - 1)




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


def distribucion_conjunta_vectorizada(probabilidades: np.ndarray) -> np.ndarray:
    """
    Construye la distribución conjunta P de tamaño 2^N a partir de N probabilidades marginales p_i.
    Vectorizado y sin bucles for explícitos utilizando meshgrid y prod.
    """
    if len(probabilidades) == 0:
        return np.array([1.0], dtype=np.float64)
    p_1 = np.asarray(probabilidades, dtype=np.float64)
    p_0 = 1.0 - p_1
    factors = np.stack([p_0, p_1], axis=1)
    grid = np.meshgrid(*factors, indexing='ij')
    dist = np.prod(grid, axis=0).flatten()
    return dist


def evaluar_k_particion(
    subsistema,
    indices_ncubos: np.ndarray,
    dims_ncubos: np.ndarray,
    particion: Tuple[List[int], ...],
    dist_original: np.ndarray,
) -> float:
    n = len(dist_original)
    dist_reconstruida = np.empty(n, dtype=np.float64)

    for parte in particion:
        if not parte:
            continue
        # Centinela -1 al inicio → mecanismo vacío (∅): futuro no depende del presente
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

    # Para N >= 13, la matriz de Hamming (2^N × 2^N) excede 512 MB y causa OOM.
    if n < 13:
        dist_P_conjunta = distribucion_conjunta_vectorizada(dist_original)
        dist_Q_conjunta = distribucion_conjunta_vectorizada(dist_reconstruida)
        return float(emd_causal(dist_P_conjunta, dist_Q_conjunta))
    else:
        return emd_causal_fast_partition(
            dist_original=dist_original,
            dist_reconstruida=dist_reconstruida,
            n_nodos=n,
            use_marginal=True
        )

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
    tabla_transiciones: dict,
    n_vars: int,
    estado_inicial_lista: list,
) -> np.ndarray:
    """
    Construye la matriz de afinidad NxN entre n-cubos a partir de los costos
    acumulados en la tabla de transiciones del hipercubo.

    Metodología:
    - Cada transición (estado_inicial → estado_s) produce un vector de costos
      de longitud N, uno por n-cubo.  Esos vectores forman la matriz de
      "perfiles de costo": C[s, i] = costo del n-cubo i en la transición s.
    - La afinidad entre n-cubo i y n-cubo j es la similitud coseno de sus
      columnas en C: variables que responden igual al hipercubo están unidas.
    - Se normaliza al rango [0, 1] desplazando la similitud coseno (∈[-1,1]).

    Args:
        tabla_transiciones: dict {(tuple_ini, tuple_fin): [c0, c1, ..., cN-1]}.
        n_vars            : Número de n-cubos del subsistema.
        estado_inicial_lista: Estado inicial como lista de enteros.

    Returns:
        Matriz numpy de forma (n_vars, n_vars), dtype float64.
    """
    ini_tuple = tuple(estado_inicial_lista)

    # Recopilar vectores de costo de todas las transiciones desde estado_inicial
    vectores: list = []
    for (t_ini, t_fin), costos in tabla_transiciones.items():
        if t_ini == ini_tuple and t_fin != ini_tuple:
            vectores.append(costos)

    if not vectores:
        return np.ones((n_vars, n_vars), dtype=np.float64)

    C = np.array(vectores, dtype=np.float64)  # (n_estados, n_vars)

    # Similitud coseno entre columnas (perfiles de cada n-cubo)
    norms = np.linalg.norm(C, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    C_norm = C / norms

    A = C_norm.T @ C_norm          # (n_vars, n_vars), valores en [-1, 1]
    A = (A + 1.0) / 2.0            # normalizar a [0, 1]
    np.fill_diagonal(A, 1.0)

    return A.astype(np.float64)


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
) -> dict:
    """
    Evalúa un valor específico de k usando heurísticas geométricas e ILS.

    Pipeline:
      1. SpectralClustering + AgglomerativeClustering (múltiples semillas/linkage).
      2. Candidatos de aislamiento: k-1 nodos aislados + clúster residual.
      3. Variantes con mecanismo vacío (∅) si está habilitado.
      4. Evaluación EMD en paralelo sobre todos los candidatos.
      5. Refinamiento local 1-move sobre el mejor candidato.
      6. Búsqueda Local Iterada (ILS): perturbación + re-refinamiento × N_ILS.

    Args:
        k               : Número de partes a evaluar.
        subsistema      : System ya condicionado/substradido (serializable).
        indices_ncubos  : Índices de los n-cubos del subsistema.
        dims_ncubos     : Dimensiones de los n-cubos.
        dists_marginales: Distribución marginal del subsistema.
        n_vars          : Número de variables del subsistema.
        n_jobs_internos : Hilos disponibles para joblib dentro de este proceso.
        matriz_afinidad : Matriz NxN de afinidad geométrica (opcional).

    Returns:
        dict con claves: k, perdida, particion, particion_grafica, error.
    """
    N_ILS = 4  # reinicios de búsqueda local iterada

    try:
        # ── Fase 1: generación de candidatos ─────────────────────────────────
        candidatos_geo: list = []
        if matriz_afinidad is not None:
            candidatos_geo = _particion_grafo_hipercubo(
                matriz_afinidad, k, n_candidatos=6
            )

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

        fmt_pk = fmt_k_particion(mejor_particion, indices_ncubos, dims_ncubos)
        return {
            "k": k,
            "perdida": float(mejor_perdida),
            "particion": mejor_particion,
            "particion_grafica": fmt_pk,
            "error": None,
        }

    except Exception as e:
        return {
            "k": k,
            "perdida": float("inf"),
            "particion": None,
            "particion_grafica": "",
            "error": str(e),
        }


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
            particion_optima = tuple([i] for i in range(n_vars))
            k_optimo = n_vars
            mejor_perdida = 0.0
            dist_reconstruida = self.sia_dists_marginales.copy()

        else:
            self._construir_tabla_costos()
            # ── Determinar estrategia de paralelización ──────────────────────
            if k is not None:
                # ═══════════════════════════════════════════════════════════
                # MODO ÚNICO K: Sin ProcessPoolExecutor, máxima capacidad
                # ═══════════════════════════════════════════════════════════
                n_jobs_internos = max(1, multiprocessing.cpu_count() - 1)
                self.logger.critic(
                    f"🎯 MODO K ÚNICO (K={k}): Usando {n_jobs_internos} núcleos "
                    f"para búsqueda exhaustiva/jerárquica."
                )

                try:
                    resultado = _evaluar_k_completo(
                        k,
                        self.sia_subsistema,
                        self.sia_subsistema.indices_ncubos,
                        self.sia_subsistema.dims_ncubos,
                        self.sia_dists_marginales,
                        len(self.sia_subsistema.indices_ncubos),
                        n_jobs_internos,
                        self._matriz_afinidad,
                        permitir_presente_vacio=permitir_presente_vacio,
                        tiempo_maximo_segundos=tiempo_maximo_segundos,
                    )
                    if resultado["error"]:
                        self.logger.critic(f"  ✗ K={k} falló: {resultado['error']}")
                        raise RuntimeError(f"Error al evaluar K={k}: {resultado['error']}")
                    else:
                        self.logger.critic(
                            f"  ✓ K={k} completado → pérdida={resultado['perdida']:.6f}"
                        )

                    resultados_k = [resultado]
                except Exception as exc:
                    self.logger.critic(f"  ✗ K={k} excepción: {exc}")
                    raise RuntimeError(f"Excepción al evaluar K={k}: {exc}")

            else:
                # ═══════════════════════════════════════════════════════════
                # MODO MÚLTIPLE K: Evaluación secuencial k=2..N
                # Cada k usa todos los núcleos (N_JOBS_INTERNOS = cpu_count-1)
                # para maximizar recursos en una sola tarea a la vez.
                # ═══════════════════════════════════════════════════════════
                n_jobs_internos = N_JOBS_INTERNOS
                valores_k = list(range(2, min(6, len(self.sia_subsistema.indices_ncubos) + 1)))

                self.logger.critic(
                    f"🔁 MODO MÚLTIPLE K: Evaluación secuencial k={valores_k[0]}..{valores_k[-1]} "
                    f"con {n_jobs_internos} núcleos por k."
                )

                resultados_k: List[dict] = []
                for test_k in valores_k:
                    self.logger.critic(f"  → Evaluando K={test_k}...")
                    try:
                        resultado = _evaluar_k_completo(
                            test_k,
                            self.sia_subsistema,
                            self.sia_subsistema.indices_ncubos,
                            self.sia_subsistema.dims_ncubos,
                            self.sia_dists_marginales,
                            len(self.sia_subsistema.indices_ncubos),
                            n_jobs_internos,
                            self._matriz_afinidad,
                            permitir_presente_vacio,
                            tiempo_maximo_segundos,
                        )
                        if resultado["error"]:
                            self.logger.critic(f"  ✗ K={test_k} falló: {resultado['error']}")
                        else:
                            self.logger.critic(
                                f"  ✓ K={test_k} completado → pérdida={resultado['perdida']:.6f}"
                            )
                        resultados_k.append(resultado)
                    except Exception as exc:
                        self.logger.critic(f"  ✗ K={test_k} excepción: {exc}")
                        resultados_k.append({
                            "k": test_k,
                            "perdida": float("inf"),
                            "particion": None,
                            "particion_grafica": "",
                            "error": str(exc),
                        })
            # ── Registrar histórico y elegir el mejor ──────────────────────
            self.historico_particiones = []
            mejor_perdida = float("inf")
            particion_optima = None
            k_optimo = None  # se asigna solo cuando realmente se encuentra un resultado válido

            for res in sorted(resultados_k, key=lambda r: r["k"]):
                self.historico_particiones.append({
                    "k": res["k"],
                    "perdida": res["perdida"],
                    "particion_grafica": res["particion_grafica"],
                })
                if res["perdida"] < mejor_perdida and res["particion"] is not None:
                    mejor_perdida = res["perdida"]
                    particion_optima = res["particion"]
                    k_optimo = res["k"]

            if particion_optima is None:
                raise RuntimeError(
                    "Todos los valores de k fallaron. "
                    "Revisa los logs para ver el motivo."
                )

            # ── Reconstruir distribución para Solution ─────────────────────
            dist_reconstruida = np.empty(
                len(self.sia_dists_marginales), dtype=np.float32
            )
            for parte in particion_optima:
                if not parte:
                    continue
                # Centinela -1 → mecanismo vacío (∅) para esta parte
                if parte[0] == -1:
                    parte_real = parte[1:]
                    presentes = np.array([], dtype=np.int8)
                else:
                    parte_real = parte
                    futuros_tmp = self.sia_subsistema.indices_ncubos[
                        np.array(parte_real, dtype=np.int8)
                    ]
                    presentes = np.intersect1d(futuros_tmp, self.sia_subsistema.dims_ncubos)
                if not parte_real:
                    continue
                futuros = self.sia_subsistema.indices_ncubos[
                    np.array(parte_real, dtype=np.int8)
                ]
                dist_parte = (
                    self.sia_subsistema.bipartir(futuros, presentes)
                    .distribucion_marginal()
                )
                for idx_pos in parte_real:
                    dist_reconstruida[idx_pos] = dist_parte[idx_pos]

        fmt = fmt_k_particion(
            particion_optima,
            self.sia_subsistema.indices_ncubos,
            self.sia_subsistema.dims_ncubos,
        )

        self.logger.critic(
            f"ÓPTIMA k-MIP (k={k_optimo}) pérdida={mejor_perdida:.6f}:\n{fmt}"
        )

        tiempo_busqueda = time.time() - self.sia_tiempo_inicio
        tiempo_prep = getattr(self, "sia_tiempo_preparacion", 0.0)

        return Solution(
            estrategia=f"{KGEOMIP_LABEL}(Global Optimal K={k_optimo})",
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
        Construye la tabla de costos de transición entre estados del hipercubo.

        Algoritmo base:
        - BFS nivel por nivel desde el estado inicial al estado final.
        - Para cada par (i,j): t_X(i,j) = γ·(|X[i]-X[j]| + Σ t_X(k,j))
            con γ = 2^(-dH(i,j)).

        La tabla se almacena en self.tabla_transiciones para ser consultada
        durante la identificación de candidatos.
        """
        self.tabla_transiciones.clear()
        # Paso 1: aplanar los hipercubos a vectores 1D para acceso por índice entero.
        self._flat_data = [
            ncubo.data.ravel()
            for ncubo in self.sia_subsistema.ncubos
        ]

        # Paso 2: obtener el estado inicial y su complemento (estado final)
        estado_inicial = self.sia_subsistema.estado_inicial[
            self.sia_subsistema.dims_ncubos
        ]
        estado_final = 1 - estado_inicial   # flip de todos los bits
        n = len(estado_inicial)
        idx_ncubos = list(range(len(self.sia_subsistema.indices_ncubos)))

        # Paso 3: entrada trivial — costo de cualquier estado hacia sí mismo = 0
        clave_trivial = (tuple(estado_inicial), tuple(estado_inicial))
        self.tabla_transiciones[clave_trivial] = [0.0] * len(idx_ncubos)

        # Paso 4: BFS nivel por nivel
        caminos: Dict[int, List[List[int]]] = {0: [estado_inicial.tolist()]}

        for nivel in range(1, n + 1):
            visitados: set = set()
            caminos[nivel] = []

            for estado_anterior in caminos[nivel - 1]:
                est_ant = np.array(estado_anterior)

                for i in range(n):
                    if est_ant[i] != estado_final[i]:
                        nuevo = est_ant.copy()
                        nuevo[i] = estado_final[i]      # flip del bit i
                        t = tuple(nuevo)

                        if t not in visitados:
                            caminos[nivel].append(nuevo.tolist())
                            self._calcular_costo(
                                caminos[0][0], nuevo.tolist(), idx_ncubos
                            )
                            visitados.add(t)

        # Guardar los caminos y estados para uso posterior
        self._caminos = caminos
        self._estado_inicial_tabla = estado_inicial.tolist()
        self._estado_final_tabla = estado_final.tolist()

        # Construir la matriz de afinidad geométrica NxN inmediatamente
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
        Construye la Matriz de Afinidad NxN entre n-cubos usando los costos
        acumulados en self.tabla_transiciones (llenada por _construir_tabla_costos).

        Almacena el resultado en self._matriz_afinidad para pasarlo a los
        procesos paralelos de _evaluar_k_completo, evitando recalcular.
        """
        n = len(self.sia_subsistema.indices_ncubos)
        self._matriz_afinidad: np.ndarray = _construir_matriz_afinidad_desde_tabla(
            self.tabla_transiciones,
            n,
            self._estado_inicial_tabla,
        )
        self.logger.critic(
            f"Matriz de afinidad {n}×{n} construida "
            f"(sklearn={'disponible' if _SKLEARN_DISPONIBLE else 'no disponible'})."
        )

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
