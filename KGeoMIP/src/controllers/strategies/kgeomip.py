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
from typing import List, Tuple, Optional
import multiprocessing

import numpy as np
from joblib import Parallel, delayed

try:
    from numba import njit
    _NUMBA_DISPONIBLE = True
except ImportError:
    _NUMBA_DISPONIBLE = False

from src.controllers.manager import Manager
from src.funcs.base import ABECEDARY, LOWER_ABECEDARY
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.base import (
    NET_LABEL,
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


# Presupuesto de candidatos de aislamiento a evaluar por k. Acota el número de
# k-particiones de aislamiento que se evalúan, INDEPENDIENTE de N: cuando hay más
# átomos de los que caben, se rankean por costo de aislamiento individual y solo se
# combinan los mejores T (el mayor T con C(T, k-1) ≤ cap). Así el coste extra queda
# acotado a ~cap evaluaciones EMD baratas para cualquier N y k, mientras se conservan
# las semillas que cierran la brecha contra KQNodes (validado: k=5/N10, top-12=495
# candidatos cierra los 5 casos que greedy+1-move perdía).
CAP_AISLAMIENTO_BLOQUES: int = 500


def _candidatos_aislamiento_bloques(
    subsistema,
    dist_original: np.ndarray,
    k: int,
    permitir_presente_vacio: bool,
    cap: int = CAP_AISLAMIENTO_BLOQUES,
) -> "list[list[Block]]":
    """
    Genera k-particiones de aislamiento (k-1 átomos + residual) en formato Block.

    Sembrar la búsqueda local desde cortes de aislamiento evita que el greedy
    top-down quede atrapado en la cuenca del primer corte miope, alcanzando óptimos
    locales que ni el 1-move ni el 2-move recuperan desde el resultado greedy (es el
    mecanismo que cierra la brecha contra KQNodes; VNS/ILS no aportan sobre esto).

    Dos familias de átomos:
      · Futuro aislado:   ({f}, ∅) si se permite mecanismo vacío; ({f}, {f})
        (simétrico, conserva su propio presente) si NO se permite.
      · Presente aislado: (∅, {p}) — bloque sin futuro; siempre válido.

    ACOTAMIENTO INDEPENDIENTE DE N: el número total de átomos es ~(futuros+presentes),
    así que C(átomos, k-1) explota en N grande. Para evitarlo se elige el mayor T tal
    que C(T, k-1) ≤ cap y, si hay más átomos que T, se rankean por su costo de
    aislamiento individual (Φ de (átomo, complemento), ~2N evaluaciones baratas) y se
    combinan solo los mejores T. Resultado: ≤ cap candidatos para CUALQUIER N y k. Si
    todos los átomos caben (C ≤ cap) se usan todos (idéntico a la enumeración total).

    La cobertura de futuros está garantizada: cada futuro o bien queda aislado en un
    átomo futuro, o bien permanece en el residual (los átomos presentes NO retiran
    futuros del residual). Se descartan bloques (∅, ∅) y, si no se permite presente
    vacío, los bloques con futuro pero sin mecanismo.

    Args:
        subsistema              : Subsistema condicionado (para rankear átomos por EMD).
        dist_original           : Distribución marginal original (size N).
        k                       : Número de bloques objetivo (≥ 2).
        permitir_presente_vacio : Si False, los átomos futuros conservan su presente.
        cap                     : Tope de candidatos a generar.

    Returns:
        Lista de candidatos; cada candidato es una lista de k Blocks (≤ cap entradas).
    """
    fut = [int(x) for x in subsistema.indices_ncubos]
    pre = [int(x) for x in subsistema.dims_ncubos]
    pre_set = set(pre)

    if permitir_presente_vacio:
        atomos: "list[Block]" = [(frozenset({f}), frozenset()) for f in fut]
    else:
        atomos = [
            (frozenset({f}), frozenset({f}) if f in pre_set else frozenset())
            for f in fut
        ]
    atomos += [(frozenset(), frozenset({p})) for p in pre]

    fut_all = frozenset(fut)
    pre_all = frozenset(pre)
    n_atomos = len(atomos)
    if k - 1 > n_atomos:
        return []

    # Mayor T con C(T, k-1) ≤ cap (acota los candidatos sin importar N).
    T = k - 1
    while T < n_atomos and comb(T + 1, k - 1) <= cap:
        T += 1

    if T >= n_atomos:
        elegidos = atomos
    else:
        # Rankear átomos por costo de aislamiento individual (Φ de (átomo, complemento)).
        costos: "list[float]" = []
        for fa, pa in atomos:
            comp_f = fut_all - fa
            comp_p = pre_all - pa
            invalido = (
                (not fa and not pa)
                or (not comp_f and not comp_p)
                or (not permitir_presente_vacio and ((fa and not pa) or (comp_f and not comp_p)))
            )
            if invalido:
                costos.append(float("inf"))
            else:
                costos.append(
                    evaluar_bloques(subsistema, [(fa, pa), (comp_f, comp_p)], dist_original)
                )
        orden = np.argsort(costos)[:T]
        elegidos = [atomos[i] for i in orden]

    candidatos: "list[list[Block]]" = []
    for combo in itertools.combinations(range(len(elegidos)), k - 1):
        fu: set = set()
        pu: set = set()
        bloques: "list[Block]" = []
        for a in combo:
            fa, pa = elegidos[a]
            bloques.append(elegidos[a])
            fu |= fa
            pu |= pa
        res_f = fut_all - fu
        res_p = pre_all - pu
        if not res_f and not res_p:
            continue
        bloques.append((res_f, res_p))
        if any(not f and not p for f, p in bloques):
            continue  # bloque (∅, ∅) inválido
        if not permitir_presente_vacio and any(f and not p for f, p in bloques):
            continue  # bloque con futuro sin mecanismo, no permitido
        candidatos.append(bloques)

    return candidatos


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
        if not eff and not pre:
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
        # Corte de aislamiento con mecanismo vacío: nodo i sin presente.
        # Cuando se aplica a bloque B: inside=({i}, ∅), outside=(B−{i}, B.pre).
        _add(eff, frozenset())
        # Corte de aislamiento con futuro vacío: presente j aislado sin futuro.
        # Cuando se aplica a bloque B: inside=(∅, {j}), outside=(B.fut, B.pre−{j}).
        if i < len(dims_ncubos):
            _add(frozenset(), frozenset([int(dims_ncubos[i])]))

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
            # Invariante: ningún bloque puede quedar completamente vacío (∅, ∅).
            # Un bloque puede tener futuro ∅ (si conserva presente) o presente ∅ (si
            # conserva futuro), pero no ambos vacíos a la vez.
            if (not inside[0] and not inside[1]) or (not outside[0] and not outside[1]):
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
            for node in eff_i:
                new_eff_i = eff_i - {node}
                if not new_eff_i and not pre_i:
                    continue  # resultaría en bloque (∅, ∅)
                for j in range(k):
                    if i == j:
                        continue
                    eff_j, pre_j = mejor_bloques[j]
                    cfg = list(mejor_bloques)
                    cfg[i] = (new_eff_i, pre_i)
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
            # Nunca vaciar el único presente de un bloque con futuro ∅ → resultaría (∅,∅).
            if not eff_i and len(pre_i) <= 1:
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
      • algún bloque queda con futuro y presente vacíos a la vez (∅, ∅).

    Returns:
        (perdida, bloques, mejoro) — mejoro=True si se encontró un par que mejora.
    """
    k = len(bloques)

    # Generar todos los movimientos 1-move válidos
    movimientos: list = []
    for i, (eff_i, pre_i) in enumerate(bloques):
        for nodo in sorted(eff_i):
            if not (eff_i - {nodo}) and not pre_i:
                continue  # resultaría en (∅, ∅)
            for j in range(k):
                if j != i:
                    movimientos.append(("f", i, j, nodo))
        if not permitir_presente_vacio and len(pre_i) <= 1:
            continue
        if not eff_i and len(pre_i) <= 1:
            continue  # evitar (∅, ∅): bloque con futuro ∅ y único presente
        for nodo in sorted(pre_i):
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
                if nodo_b not in eff_ib:
                    continue
                if not (eff_ib - {nodo_b}) and not pre_ib:
                    continue  # resultaría en (∅, ∅)
                bl_b[i_b] = (eff_ib - {nodo_b}, pre_ib)
                bl_b[j_b] = (eff_jb | {nodo_b}, pre_jb)
            else:
                if nodo_b not in pre_ib:
                    continue
                if not permitir_presente_vacio and len(pre_ib) <= 1:
                    continue
                bl_b[i_b] = (eff_ib, pre_ib - {nodo_b})
                bl_b[j_b] = (eff_jb, pre_jb | {nodo_b})

            if any(not eff and not pre for eff, pre in bl_b):
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

    Garantiza que ningún bloque quede completamente vacío (∅, ∅).
    Los movimientos futuros y presentes son independientes (asimetría).
    """
    rng = _random_module.Random(semilla)
    fut_listas = [list(eff) for eff, _ in bloques]
    pre_listas = [list(pre) for _, pre in bloques]
    k = len(bloques)

    for _ in range(n_movimientos):
        # Mover un nodo futuro entre bloques
        candidatos_orig = [
            i for i, (f, p) in enumerate(zip(fut_listas, pre_listas))
            if len(f) >= 1 and (len(f) > 1 or len(p) > 0)
        ]
        if candidatos_orig:
            i = rng.choice(candidatos_orig)
            nodo = rng.choice(fut_listas[i])
            j = rng.choice([x for x in range(k) if x != i])
            fut_listas[i].remove(nodo)
            fut_listas[j].append(nodo)

        # Mover un nodo presente entre bloques (independiente del futuro)
        if permitir_presente_vacio:
            # Excluir bloques con futuro ∅ y un solo presente: quedarían (∅, ∅).
            candidatos_pre = [
                i for i, (f, p) in enumerate(zip(fut_listas, pre_listas))
                if len(p) > 0 and (len(f) > 0 or len(p) > 1)
            ]
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

        # j = número de bloques con futuro no vacío (j=kk: caso original).
        # j < kk: bloques j..kk-1 tienen futuro ∅ (deben conservar ≥1 presente).
        for j in range(1, kk + 1):
            iters_j = _stirling2(n_vars, j) * (kk ** n_dims)
            if iters_j > _KGEOMIP_CAP_EXACTO or _stirling2(n_vars, j) == 0:
                continue  # demasiado costoso o imposible para este j

            for part_fut in _particiones_en_k(futuros, j):
                bloques_fut = [frozenset(int(idx[p]) for p in b) for b in part_fut]
                # Rellenar bloques j..kk-1 con futuro vacío
                bloques_fut += [frozenset()] * (kk - j)

                if n_dims == 0:
                    if j < kk:
                        continue  # bloques sin futuro necesitan presente; n_dims=0 imposible
                    bloques = [(bloques_fut[i], frozenset()) for i in range(kk)]
                    phi = evaluar_bloques(subsistema, bloques, dist_original)
                    if phi < mejor_phi:
                        mejor_phi, mejor_bloques = phi, bloques
                    continue

                for asignacion in itertools.product(range(kk), repeat=n_dims):
                    pre_sets: list[list[int]] = [[] for _ in range(kk)]
                    for pos_pre, b_idx in enumerate(asignacion):
                        pre_sets[b_idx].append(int(dims[pos_pre]))
                    # Descartar bloques (∅, ∅)
                    if any(not bloques_fut[i] and not pre_sets[i] for i in range(kk)):
                        continue
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

    Para N ≤ _KGEOMIP_N_EXACTO resuelve por enumeración exhaustiva. Para N mayor
    usa el pipeline heurístico de producción: greedy top-down sobre un cut pool
    geométrico + candidatos de aislamiento + refinamiento 1-move.

    Args:
        gestor (Manager): Gestor con el estado inicial y ruta de la TPM.
        k      (int)    : Número de partes de la partición (default=2).

    Attributes:
        k : Número de partes solicitado.
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

                    # ── Siembra por candidatos de aislamiento (determinista) ──
                    # El greedy top-down es miope en el primer corte y cae en
                    # cuencas peores que ni el 1-move ni el 2-move recuperan. Se
                    # generan k-particiones de aislamiento (k-1 átomos + residual),
                    # se evalúan en paralelo (EMD barato) y se refina solo la mejor;
                    # se conserva si bate al greedy+1-move. Es el mecanismo que cierra
                    # la brecha contra KQNodes en el régimen permitir_presente_vacio
                    # (caso N10A k=3: 0.017578 → 0.011719). Acotado por
                    # CAP_AISLAMIENTO_BLOQUES para que el tiempo no se dispare en k alto.
                    cands_ais = _candidatos_aislamiento_bloques(
                        self.sia_subsistema,
                        self.sia_dists_marginales,
                        test_k,
                        permitir_presente_vacio,
                    )
                    if cands_ais:
                        perdidas_ais = Parallel(
                            n_jobs=min(len(cands_ais), N_JOBS_INTERNOS),
                            prefer="threads",
                        )(
                            delayed(evaluar_bloques)(
                                self.sia_subsistema, c, self.sia_dists_marginales
                            )
                            for c in cands_ais
                        )
                        idx_ais = int(np.argmin(perdidas_ais))
                        perdida_ref, bloques_ref = _refinar_bloques_1move(
                            self.sia_subsistema,
                            self.sia_dists_marginales,
                            cands_ais[idx_ais],
                            n_jobs=N_JOBS_INTERNOS,
                            permitir_presente_vacio=permitir_presente_vacio,
                        )
                        if perdida_ref < perdida_k - 1e-9:
                            perdida_k, bloques_k = perdida_ref, bloques_ref
                            self.logger.critic(
                                f"  +aislamiento K={test_k} mejoró → {perdida_k:.6f} "
                                f"({len(cands_ais)} cand.)"
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

        # Origin = little-endian packing of the initial state over the mechanism dims.
        estado_ini = subsistema.estado_inicial[subsistema.dims_ncubos]
        idx_origen = 0
        for pos, bit in enumerate(estado_ini):
            if bit:
                idx_origen |= 1 << pos

        estados = np.arange(total_states, dtype=np.int32)
        xor_origen = (estados ^ idx_origen).astype(np.uint32)
        dist = _popcount_vec(xor_origen)

        self.tabla_T = np.zeros((total_states, n), dtype=np.float32)

        if _NUMBA_DISPONIBLE:
            # Ruta JIT: un kernel njit recorre los estados en orden de popcount
            # ascendente y aplica la recurrencia. prob_T (filas = estados, contiguo
            # por filas para acceso cache-friendly) se construye DIRECTAMENTE columna
            # a columna desde los n-cubos, SIN materializar antes prob (n×2^d) ni su
            # transpuesta: una sola copia de 2^d×n en vez de dos (ahorra ~3.35 GB en
            # N25). Mismos valores y mismo Φ que la versión anterior.
            prob_T = np.empty((total_states, n), dtype=np.float32)
            for i, ncubo in enumerate(subsistema.ncubos):
                prob_T[:, i] = ncubo.data.ravel()
            prob_origen_c = np.ascontiguousarray(prob_T[idx_origen])   # (n,)
            orden = np.argsort(dist, kind="stable").astype(np.int32)
            _kernel_tabla_costos(
                prob_T, prob_origen_c, xor_origen, orden,
                int(n_dims), int(n), self.tabla_T,
            )
        else:
            # Ruta numpy (sin Numba): opera por bloques de columnas prob[:, bloque],
            # así que aquí sí se necesita prob (n×2^d). Solo se construye en esta rama.
            prob = np.array(
                [ncubo.data.ravel() for ncubo in subsistema.ncubos],
                dtype=np.float32,
            )
            prob_origen = prob[:, idx_origen]        # (n,)
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

        # El camino de producción (greedy top-down + cut_pool + 1-move) no usa
        # matriz de afinidad ni clustering espectral.


    # ── Candidatos asimétricos desde tabla_T ─────────────────────────────────

    def _candidatos_desde_tabla_T(
        self,
    ) -> "List[Tuple[List[int], List[int]]]":
        """
        Generate asymmetric bipartition candidates from the full cost table.

        Unlike symmetric cuts (where future and present are derived from the same
        bit-flip set), this method separates the two axes:

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
                if not fs and not ps:  # descartar cortes (∅, ∅)
                    continue
                key = (tuple(sorted(fs)), tuple(sorted(ps)))
                if key not in seen:
                    seen.add(key)
                    candidates.append((list(fs), list(ps)))

        return candidates
