"""
QNodes — k-MIP heurístico mediante cortes ASIMÉTRICOS y greedy top-down.

El problema k-MIP (Partición de Mínima Información) requiere dividir un sistema
de N nodos en k subconjuntos minimizando la pérdida de información integrada
Phi (Φ). La búsqueda exhaustiva sobre B(N) particiones es intratable para
N ≥ 15, por lo que QNodes usa una estrategia greedy con refinamiento local.

Representación ASIMÉTRICA de bloques (clave para coherencia entre k)
───────────────────────────────────────────────────────────────────
Cada bloque es un par (futuros, presentes) de posiciones locales que se
particionan de forma INDEPENDIENTE:

  • futuros  (t+1): el alcance/efecto que el bloque produce.
  • presentes (t):  el mecanismo/causa que el bloque conserva como condicionante.

A diferencia de un corte SIMÉTRICO (donde el presente de un bloque es siempre
futuros ∩ dims, es decir cada grupo solo condiciona sobre sus propios nodos),
el corte asimétrico permite que un nodo aislado en su futuro siga actuando como
condicionante causal de otro bloque. Esto replica exactamente la lógica de
GeoMIP y evita el "sobre-corte" que infla Φ.

Motivación (medido en N10A, estado=1000…0):
  - Corte simétrico:  k=2 → 0.4746, k=3 → 2.5059  (salto incoherente)
  - Corte asimétrico: k=2 → 0.4746, k=3 → 0.9590  (monótono, coherente)

El caso k=2 asimétrico — aislar Xi con mecanismo ∅ y dejar al residual con TODOS
los presentes — ya reproducía el Φ de GeoMIP; aquí se generaliza a todo k.

Algoritmo
─────────
1. Queyranne exacto con 2N átomos ASIMÉTRICOS para todo N:
       ({i},∅) para futuros  +  (∅,{j}) para presentes.
       Queyranne explora TODO el espacio de biparticiones (fut,pre) de AYDA.
       Garantía: bipartición k=2 globalmente óptima en la métrica exacta.
       NCube.marginal_valor hace cada evaluación O(2^(N-|pre|)) en vez de O(2^N),
       viable para todo N hasta ~25. Los pares colgantes (2N-2 candidatos)
       forman el pool de cortes para k≥3.
2. Pool de cortes O(N): por cada nodo i se generan
     ({i}, {pre_i})          aislamiento simétrico,
     (resto, resto_pre)      su complemento,
     ({i}, ∅)                aislamiento con mecanismo vacío (corte GeoMIP).
   Se combina con el pool de Queyranne (pares colgantes tienen prioridad).
3. Greedy top-down: se parte del bloque único (TODOS los futuros, TODOS los
   presentes) y se aplican k-1 mejores divisiones usando el pool combinado.
   Un solo descenso registra Φ para CADA k (jerarquía nido → coherencia).
4. Refinamiento local best-improvement sobre bloques:
     - movimiento futuro:  reubica un nodo futuro entre bloques,
     - movimiento presente (asimétrico): reubica el mecanismo de un nodo sin
       mover su futuro — imposible en representaciones simétricas.
5. ILS — Búsqueda Local Iterada: perturba y re-refina el k ganador.

Métrica EMD (suma L1 marginal — EXACTA, no aproximada):
   La reconstrucción de toda k-partición es un producto de marginales por nodo
   (independencia condicional garantizada por construcción), y la distribución
   original se modela igual. Para DOS distribuciones producto con métrica base
   de Hamming, la Wasserstein-1 (EMD real) coincide EXACTAMENTE con la suma de
   diferencias L1 marginales, porque la distancia de Hamming es separable por
   coordenada y el acoplamiento óptimo factoriza. Verificado numéricamente:
   |emd_causal − L1| < 1e-14 para N=2..12. Por tanto se usa L1 directo, que es
   O(N) en lugar de O(4^N) del pyemd, dando el MISMO Φ (p.ej. k=2 N10A = 0.4746,
   idéntico a GeoMIP) sin restricción de tamaño en N.

Memoización:
  • _cache_bloque[(futuros, presentes)]: distribución marginal del bloque.
  Los cachés se limpian entre ejecuciones independientes.
"""
import random as _random_module
import time
from itertools import product
from typing import Generator, Optional

import numpy as np

from src.funcs.force import particiones_en_k
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import gestor_perfilado, profile
from src.funcs.format import fmt_k_bloques
from src.funcs.iit import HAMMING_EMD_MAX_N
from src.models.base.sia import SIA
from src.models.core.ncube import NCube
from src.models.core.solution import Solution
from src.constants.models import (
    QNODES_ANALYSIS_TAG,
    QNODES_LABEL,
    QNODES_STRAREGY_TAG,
)
from src.constants.base import COLS_IDX, NET_LABEL, TYPE_TAG
from src.models.base.application import aplicacion


# Default alineado con KGeoMIP y run_suite_2026 (convención del proyecto): la
# k-MIP "real" permite mecanismo ∅. Antes era False, lo que dejaba a QNodes
# explorando un espacio distinto al de KGeoMIP por defecto → Φ divergentes.
PERMITIR_PRESENTE_VACIO_POR_DEFECTO: bool = True

# Iteraciones base de Búsqueda Local Iterada (escala según N en _refinar_con_ils).
_N_ILS: int = 4

# Iteraciones de refinamiento por nivel en k=None (el ILS final refina el ganador).
_MAX_ITER_NIVEL: int = 5

# Para N ≤ _N_EXACTO se resuelve la k-MIP por enumeración exhaustiva (mismo espacio
# que BruteForceKMIP): garantía de óptimo global exacto, así QNodes/KGeoMIP/fuerza
# bruta coinciden en CSVs pequeños. _CAP_EXACTO acota configuraciones por k para no
# colgar (si se excede, ese tamaño cae a la heurística).
_N_EXACTO: int = 6
_CAP_EXACTO: int = 300_000



# Un bloque asimétrico: (frozenset futuros_pos, frozenset presentes_pos).
Bloque = "tuple[frozenset, frozenset]"


def _bits_activos(mascara: int) -> Generator[int, None, None]:
    """Genera los índices de bits activos de `mascara` en orden ascendente."""
    m = mascara
    while m:
        bit = m & (-m)
        yield bit.bit_length() - 1
        m ^= bit


class QNodes(SIA):
    """
    Estrategia QNodes para k-MIP mediante cortes asimétricos y greedy top-down.

    Cada bloque mantiene futuros y presentes independientes, lo que permite
    cortes asimétricos coherentes para todo k (ver docstring del módulo). El
    greedy top-down construye una jerarquía nido (un Φ por cada k en un solo
    descenso) que luego se refina con búsqueda local best-improvement (1-move
    futuro + 1-move presente) e ILS.

    Attrs:
        _cache_bloque : (futuros, presentes) → distribución marginal del bloque.
    """

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self._N: int = 0
        self._n_dims: int = 0
        self.logger = SafeLogger(QNODES_STRAREGY_TAG)
        self._permitir_presente_vacio: bool = False
        self._cache_bloque: dict[tuple, np.ndarray] = {}
        self._idx: np.ndarray = np.array([], dtype=np.int8)
        self._dims: np.ndarray = np.array([], dtype=np.int8)
        self._ncubos_idx: dict[int, NCube] = {}
        # Última k-partición ganadora (expuesta para validación/comparación externa).
        self.mejor_bloques: list = []

    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: Optional[int] = None,
        permitir_presente_vacio: bool = PERMITIR_PRESENTE_VACIO_POR_DEFECTO,
    ) -> Solution:
        self._permitir_presente_vacio = permitir_presente_vacio
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        self._idx = self.sia_subsistema.indices_ncubos
        self._dims = self.sia_subsistema.dims_ncubos
        self._N = len(self._idx)
        self._n_dims = len(self._dims)

        # Índice O(1) NCube por índice global (para marginal_valor en _dist_bloque).
        self._ncubos_idx = {int(nc.indice): nc for nc in self.sia_subsistema.ncubos}

        # Limpiar caché entre ejecuciones independientes
        self._cache_bloque.clear()

        if self._N < 2:
            dist_trivial = self.sia_dists_marginales.copy()
            return Solution(
                estrategia=QNODES_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=dist_trivial,
                tiempo_total=time.time() - self.sia_tiempo_inicio,
                tiempo_preparacion=self.sia_tiempo_preparacion,
                particion="Sistema trivial (N<2)\n",
            )

        if k is not None and (k < 2 or k > self._N):
            raise ValueError(f"k={k} fuera del rango permitido [2, {self._N}]")

        # ── Ruta EXACTA para N pequeño ──────────────────────────────────────
        # Enumera todas las k-particiones asimétricas válidas (mismo espacio que
        # BruteForceKMIP) y devuelve el óptimo global. Garantiza que QNodes coincide
        # con la fuerza bruta y con KGeoMIP en CSVs pequeños, sin depender de que la
        # heurística alcance el óptimo. Devuelve None si el espacio supera el cap.
        exacto = None
        if self._N <= _N_EXACTO:
            exacto = self._resolver_exacto(k, permitir_presente_vacio)

        if exacto is not None:
            mejor_phi, mejor_bloques = exacto
        else:
            mejor_phi, mejor_bloques = self._resolver_heuristico(
                k, permitir_presente_vacio
            )

        # Expone la k-partición ganadora para comparación externa (validación).
        self.mejor_bloques = mejor_bloques

        # Reconstrucción de la distribución de la partición óptima.
        dist_reconstruida = np.empty(self._N, dtype=np.float32)
        for fut_pos, pre_pos in mejor_bloques:
            if not fut_pos:
                continue
            dist_bloque = self._dist_bloque(fut_pos, pre_pos)
            for p in fut_pos:
                dist_reconstruida[p] = float(dist_bloque[p])

        fmt_mip = fmt_k_bloques(mejor_bloques, self._idx, self._dims)

        return Solution(
            estrategia=QNODES_LABEL,
            perdida=mejor_phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_reconstruida,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            tiempo_preparacion=self.sia_tiempo_preparacion,
            particion=fmt_mip,
        )

    # ── Solver exacto (fuerza bruta) para N pequeño ────────────────────────

    @staticmethod
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

    def _resolver_exacto(
        self, k: Optional[int], permitir_vacio: bool
    ) -> "Optional[tuple[float, list]]":
        """
        Enumera exhaustivamente las k-particiones asimétricas válidas y devuelve
        (Φ mínimo, bloques) — réplica del espacio de BruteForceKMIP usando el
        mismo _emd_bloques de QNodes, así el óptimo es idéntico bit a bit.

        Para cada k: particiones de los N futuros en k bloques no vacíos
        (Stirling2(N,k)) × asignaciones de los n_dims presentes a los k bloques
        (k^n_dims). Con permitir_vacio=False descarta asignaciones que dejen algún
        bloque sin presente. Si el espacio de algún k requerido supera _CAP_EXACTO
        devuelve None (el llamador usa la heurística).

        Returns:
            (phi, bloques) del mejor k (o del k pedido), o None si no es tratable.
        """
        ks = [k] if k is not None else list(range(2, self._N + 1))
        for kk in ks:
            if self._stirling2(self._N, kk) * (kk ** self._n_dims) > _CAP_EXACTO:
                return None

        futuros = list(range(self._N))
        mejor_por_k: dict[int, tuple[float, list]] = {}

        for kk in ks:
            mejor_phi = float("inf")
            mejor_bloques: Optional[list] = None

            # j = número de bloques con futuro no vacío (j=kk: caso original).
            # j < kk: bloques j..kk-1 tienen futuro ∅ (deben conservar ≥1 presente).
            for j in range(1, kk + 1):
                s_n_j = self._stirling2(self._N, j)
                iters_j = s_n_j * (kk ** self._n_dims)
                if iters_j > _CAP_EXACTO or s_n_j == 0:
                    continue  # demasiado costoso o imposible para este j

                for part_fut in particiones_en_k(futuros, j):
                    bloques_fut = [frozenset(b) for b in part_fut]
                    # Rellenar bloques j..kk-1 con futuro vacío
                    bloques_fut += [frozenset()] * (kk - j)

                    if self._n_dims == 0:
                        if j < kk:
                            continue  # bloques sin futuro necesitan presente; n_dims=0 imposible
                        bloques = [(bloques_fut[i], frozenset()) for i in range(kk)]
                        phi = self._emd_bloques(bloques)
                        if phi < mejor_phi:
                            mejor_phi, mejor_bloques = phi, bloques
                        continue

                    for asignacion in product(range(kk), repeat=self._n_dims):
                        pre_sets: list[list[int]] = [[] for _ in range(kk)]
                        for pos_pre, b_idx in enumerate(asignacion):
                            pre_sets[b_idx].append(pos_pre)
                        # Descartar bloques (∅, ∅)
                        if any(not bloques_fut[i] and not pre_sets[i] for i in range(kk)):
                            continue
                        if not permitir_vacio and any(len(s) == 0 for s in pre_sets):
                            continue
                        bloques = [
                            (bloques_fut[i], frozenset(pre_sets[i])) for i in range(kk)
                        ]
                        phi = self._emd_bloques(bloques)
                        if phi < mejor_phi:
                            mejor_phi, mejor_bloques = phi, bloques

            if mejor_bloques is not None:
                mejor_por_k[kk] = (mejor_phi, mejor_bloques)

        if not mejor_por_k:
            return None
        if k is not None:
            return mejor_por_k.get(k)
        # k libre: preferir k ≥ 3 si existe (coherente con la rama heurística).
        candidatos = {kk: v for kk, v in mejor_por_k.items() if kk >= 3} or mejor_por_k
        mejor_k = min(candidatos, key=lambda kk: candidatos[kk][0])
        return candidatos[mejor_k]

    # ── Solver heurístico (greedy + refinamiento + ILS) ────────────────────

    def _resolver_heuristico(
        self, k: Optional[int], permitir_presente_vacio: bool
    ) -> "tuple[float, list]":
        """Pipeline heurístico original: Queyranne k=2 + greedy top-down + ILS."""
        # Universo de átomos según el flag: asimétrico (2N, permite ∅) cuando se
        # permite mecanismo vacío; simétrico (N, ({i},{i})) cuando NO se permite,
        # para que ningún corte de Queyranne deje un bloque con futuro sin presente.
        # marginal_valor hace cada evaluación O(2^(N-|pre|)) en vez de O(2^N).
        atomos = (
            self._atomos_asimetricos()
            if permitir_presente_vacio
            else self._atomos_simetricos()
        )
        phi_q2, bloques_q2, pool_pendant = self._queyranne(atomos)

        # Pool combinado: pares colgantes de Queyranne (prioridad) + geométrico O(N).
        pool_geo = self._construir_pool_cortes()
        _vistos: set = set()
        pool: list = []
        for _corte in pool_pendant + pool_geo:
            _clave = (frozenset(_corte[0]), frozenset(_corte[1]))
            if _clave not in _vistos:
                _vistos.add(_clave)
                pool.append(_clave)

        if k is not None:
            if k == 2:
                # phi_q2 es el óptimo global exacto (2N átomos asimétricos).
                mejor_phi, mejor_bloques = phi_q2, bloques_q2
            else:
                mejor_phi, mejor_bloques = self._greedy_bloques(pool, k)
            mejor_phi, mejor_bloques = self._refinar_bloques(mejor_bloques, mejor_phi)
            mejor_phi, mejor_bloques = self._refinar_con_ils(mejor_bloques, mejor_phi)
        else:
            # k libre: descenso con pool enriquecido + refinamiento por nivel.
            historico = self._greedy_descenso(pool)

            historico_refinado: dict[int, tuple[float, list]] = {}
            for k_nivel, (phi_nivel, bloques_nivel) in historico.items():
                if k_nivel < 2:
                    continue
                phi_r, bloques_r = self._refinar_bloques(
                    bloques_nivel, phi_nivel, max_iter=_MAX_ITER_NIVEL
                )
                historico_refinado[k_nivel] = (phi_r, bloques_r)

            # Inyectar k=2 de Queyranne si mejora el histórico del descenso greedy.
            phi_q2_r, bloques_q2_r = self._refinar_bloques(
                bloques_q2, phi_q2, max_iter=_MAX_ITER_NIVEL
            )
            if 2 not in historico_refinado or phi_q2_r < historico_refinado[2][0]:
                historico_refinado[2] = (phi_q2_r, bloques_q2_r)

            candidatos_k3 = {kk: ph for kk, (ph, _) in historico_refinado.items() if kk >= 3}
            if not candidatos_k3:
                candidatos_k3 = {kk: ph for kk, (ph, _) in historico_refinado.items()}
            mejor_k = min(candidatos_k3, key=candidatos_k3.get)
            mejor_phi, mejor_bloques = historico_refinado[mejor_k]
            mejor_phi, mejor_bloques = self._refinar_con_ils(mejor_bloques, mejor_phi)

        return mejor_phi, mejor_bloques

    # ── Pool de cortes asimétricos ─────────────────────────────────────────

    def _construir_pool_cortes(self) -> "list[tuple[frozenset, frozenset]]":
        """
        Construye el pool de cortes candidatos a partir de la estructura por nodo.

        Tres familias por cada nodo i (posición local):
          1. ({i}, {pre_i})            aislamiento simétrico (futuro+su mecanismo).
          2. (resto, resto_pre)        su complemento.
          3. ({i}, ∅)                  aislamiento con mecanismo vacío (corte GeoMIP):
             al aplicarse a un bloque B produce inside=({i}, ∅) y outside con el
             mecanismo de i intacto — i sigue condicionando causalmente al resto.

        Para K=2 una de estas familias reproduce el corte MIP exacto en la
        práctica totalidad de los casos; para K≥3 el greedy las encadena y el
        refinamiento las pule.
        """
        all_fut = frozenset(range(self._N))
        all_pre = frozenset(range(self._n_dims))
        pool: list = []
        vistos: set = set()

        def _add(eff: frozenset, pre: frozenset) -> None:
            if not eff and not pre:
                return
            clave = (eff, pre)
            if clave not in vistos:
                vistos.add(clave)
                pool.append(clave)

        for i in range(self._N):
            eff = frozenset((i,))
            pre = frozenset((i,)) if i < self._n_dims else frozenset()
            _add(eff, pre)
            _add(all_fut - eff, all_pre - pre)
            if self._permitir_presente_vacio:
                _add(eff, frozenset())  # corte de mecanismo vacío (solo si se permite)
            # Corte de futuro vacío: presente j aislado sin futuro.
            # Cuando se aplica a bloque B: inside=(∅, {j}), outside=(B.fut, B.pre−{j}).
            if i < self._n_dims:
                _add(frozenset(), frozenset((i,)))

        return pool

    # ── Generadores de átomos para Queyranne ──────────────────────────────

    def _atomos_asimetricos(self) -> "list[tuple[frozenset, frozenset]]":
        """
        2N átomos para el universo asimétrico completo (N ≤ _QUEYRANNE_N_MAX).

        N átomos futuros  ({i}, ∅)  para i = 0..N-1
        N átomos presentes (∅, {j}) para j = 0..n_dims-1

        Queyranne puede descubrir cualquier corte (S_fut, S_pre) donde futuros
        y presentes se distribuyen de forma completamente independiente.
        Precalienta _cache_bloque para los singletons futuros (se reutilizan
        en la evaluación _f({k},∅) de cada delta en el bucle interno).
        """
        atomos: list[tuple[frozenset, frozenset]] = (
            [(frozenset({i}), frozenset()) for i in range(self._N)]
            + [(frozenset(), frozenset({j})) for j in range(self._n_dims)]
        )
        for fut, pre in atomos:
            if fut:
                self._dist_bloque(fut, pre)
        return atomos

    def _atomos_simetricos(self) -> "list[tuple[frozenset, frozenset]]":
        """
        N átomos SIMÉTRICOS ({i}, {i}) — universo para permitir_presente_vacio=False.

        Cada átomo acopla el futuro i con su propio presente i (∅ solo si i no tiene
        mecanismo, i ≥ n_dims). Así TODO corte de Queyranne lleva el presente junto
        a su futuro: ningún bloque con futuro queda sin mecanismo cuando el sistema
        tiene mecanismo para esos nodos. Es el universo correcto cuando NO se permite
        mecanismo ∅, a diferencia de los 2N átomos asimétricos (que sí generan ∅).
        """
        atomos: list[tuple[frozenset, frozenset]] = [
            (frozenset({i}), frozenset({i}) if i < self._n_dims else frozenset())
            for i in range(self._N)
        ]
        for fut, pre in atomos:
            if fut:
                self._dist_bloque(fut, pre)
        return atomos

    # ── Algoritmo de Queyranne (núcleo genérico) ───────────────────────────

    def _queyranne(
        self,
        atomos: "list[tuple[frozenset, frozenset]]",
    ) -> "tuple[float, list[tuple[frozenset, frozenset]], list[tuple[frozenset, frozenset]]]":
        """
        Algoritmo de Queyranne sobre un conjunto de átomos arbitrario.

        Minimiza f(S) = _emd_bloques(S | V-S) en O(|atomos|²) evaluaciones de
        bipartición, usando ordenamiento por adyacencia máxima (maximum adjacency
        ordering) y contracción del par colgante de cada fase.

        f es simétrica (f(S)=f(V-S)) y submodular: hereda ambas propiedades de
        la suma L1 marginal sobre distribuciones producto con métrica de Hamming
        separable. Con estas dos propiedades la garantía de Queyranne aplica: el
        par colgante de ALGUNA fase es la bipartición k=2 óptima en el espacio
        de los átomos dados.

        Soporta dos universos de átomos (ver _atomos_asimetricos / _atomos_simetricos):
          • Asimétrico (2N átomos): átomos presentes-only (fut=∅) se registran
            por su complemento al encontrarlos como par colgante.
          • Simétrico (N átomos): todos los átomos tienen fut≠∅; la diagonal
            crece coherentemente en cada contracción.

        _cache_bloque es compartido: las evaluaciones de Queyranne precalientan
        el caché para el greedy y el refinamiento que corren después.
        omega_fut/pre se mantienen incrementales — O(1) por iteración interna.

        Retorna:
            phi_k2:       Φ de la bipartición k=2 óptima hallada.
            bloques_k2:   Los dos bloques asimétricos de esa bipartición.
            pool_pendant: Todos los pares colgantes (un candidato por fase).
        """
        all_fut = frozenset(range(self._N))
        all_pre = frozenset(range(self._n_dims))

        def _f(s_fut: frozenset, s_pre: frozenset) -> float:
            comp_fut = all_fut - s_fut
            comp_pre = all_pre - s_pre
            partes: list = []
            if s_fut or s_pre:  # bloque válido: no (∅, ∅)
                partes.append((s_fut, s_pre))
            if comp_fut or comp_pre:  # complemento válido: no (∅, ∅)
                partes.append((comp_fut, comp_pre))
            if len(partes) < 2:
                return float("inf")
            # Con presente vacío deshabilitado, un bloque con futuro pero sin
            # mecanismo es inválido: descártalo para que Queyranne no lo elija.
            if not self._permitir_presente_vacio and any(
                fut and not pre for fut, pre in partes
            ):
                return float("inf")
            return self._emd_bloques(partes)

        fase_atomos: list[tuple[frozenset, frozenset]] = list(atomos)
        candidatos: dict[tuple[frozenset, frozenset], float] = {}

        while len(fase_atomos) > 2:
            omegas: list[tuple[frozenset, frozenset]] = [fase_atomos[0]]
            deltas: list[tuple[frozenset, frozenset]] = list(fase_atomos[1:])
            omega_fut: frozenset = fase_atomos[0][0]
            omega_pre: frozenset = fase_atomos[0][1]

            for _ in range(len(deltas) - 1):
                min_ganancia = float("inf")
                mejor_k = 0
                for k_idx, (v_fut, v_pre) in enumerate(deltas):
                    ganancia = _f(omega_fut | v_fut, omega_pre | v_pre) - _f(v_fut, v_pre)
                    if ganancia < min_ganancia:
                        min_ganancia = ganancia
                        mejor_k = k_idx

                elegido_fut, elegido_pre = deltas[mejor_k]
                omega_fut = omega_fut | elegido_fut
                omega_pre = omega_pre | elegido_pre
                omegas.append(deltas[mejor_k])
                deltas.pop(mejor_k)

            pendant_fut, pendant_pre = deltas[0]
            # Registrar el par colgante directamente; _f ya soporta futuro ∅.
            candidatos[(pendant_fut, pendant_pre)] = _f(pendant_fut, pendant_pre)

            omegas[-1] = (omegas[-1][0] | pendant_fut, omegas[-1][1] | pendant_pre)
            fase_atomos = omegas

        for at_fut, at_pre in fase_atomos:
            if at_fut or at_pre:  # no registrar (∅, ∅)
                candidatos[(at_fut, at_pre)] = _f(at_fut, at_pre)

        if not candidatos:
            phi = self._emd_bloques([(all_fut, all_pre)])
            return phi, [(all_fut, all_pre)], []

        mejor_corte = min(candidatos, key=candidatos.get)
        mejor_phi = candidatos[mejor_corte]
        mejor_fut, mejor_pre = mejor_corte

        bloques_k2: list[tuple[frozenset, frozenset]] = []
        comp_fut = all_fut - mejor_fut
        comp_pre = all_pre - mejor_pre
        # Registrar AMBOS lados del corte si son válidos (no (∅,∅)). Incluye el
        # caso en que un lado tiene futuro ∅ pero conserva presente — p.ej. el
        # corte aísla un mecanismo (∅,{j}): ese bloque es válido y debe figurar,
        # de lo contrario bloques_k2 colapsa a un solo bloque (la "no-partición").
        if mejor_fut or mejor_pre:
            bloques_k2.append((mejor_fut, mejor_pre))
        if comp_fut or comp_pre:
            bloques_k2.append((comp_fut, comp_pre))

        pool_pendant: list[tuple[frozenset, frozenset]] = [
            (frozenset(f), frozenset(p)) for f, p in candidatos if f
        ]

        return mejor_phi, bloques_k2, pool_pendant

    # ── Greedy top-down sobre bloques asimétricos ──────────────────────────

    def _mejor_split_bloques(
        self,
        bloques: "list[tuple[frozenset, frozenset]]",
        pool: "list[tuple[frozenset, frozenset]]",
    ) -> "Optional[tuple[float, list]]":
        """
        Halla la mejor división de un único bloque sobre todas las (bloque, corte).

        Para cada bloque b y cada corte c calcula:
            inside  = (b.fut ∩ c.fut, b.pre ∩ c.pre)
            outside = (b.fut − c.fut, b.pre − c.pre)
        Se exige que ningún bloque quede completamente vacío (∅, ∅); un bloque
        puede tener futuro ∅ (si conserva presente) o presente ∅ (si conserva futuro).
        """
        mejor: list | None = None
        mejor_phi = float("inf")

        for pos, (eff_b, pre_b) in enumerate(bloques):
            if len(eff_b) + len(pre_b) <= 1:
                continue  # bloque demasiado pequeño para dividir en dos partes válidas
            for cut_eff, cut_pre in pool:
                in_eff = eff_b & cut_eff
                out_eff = eff_b - cut_eff
                in_pre = pre_b & cut_pre
                out_pre = pre_b - cut_pre
                if (not in_eff and not in_pre) or (not out_eff and not out_pre):
                    continue  # algún lado sería (∅, ∅)
                inside = (in_eff, in_pre)
                outside = (out_eff, out_pre)
                # Sin presente vacío: ningún lado del corte puede quedar sin mecanismo.
                if not self._permitir_presente_vacio and (
                    not inside[1] or not outside[1]
                ):
                    continue
                cfg = bloques[:pos] + [inside, outside] + bloques[pos + 1:]
                phi = self._emd_bloques(cfg)
                if phi < mejor_phi:
                    mejor_phi = phi
                    mejor = cfg

        if mejor is None:
            return None
        return mejor_phi, mejor

    def _greedy_bloques(
        self,
        pool: "list[tuple[frozenset, frozenset]]",
        k: int,
    ) -> "tuple[float, list]":
        """Greedy top-down deteniéndose al alcanzar k bloques."""
        bloques: list = [(frozenset(range(self._N)), frozenset(range(self._n_dims)))]
        phi = self._emd_bloques(bloques)
        while len(bloques) < k:
            resultado = self._mejor_split_bloques(bloques, pool)
            if resultado is None:
                break
            phi, bloques = resultado
        return phi, bloques

    def _greedy_descenso(
        self,
        pool: "list[tuple[frozenset, frozenset]]",
    ) -> "dict[int, tuple[float, list]]":
        """
        Descenso greedy completo de k=1 a k=N registrando Φ en cada nivel.

        Un único descenso produce una jerarquía NIDO: cada k surge de dividir
        un bloque del nivel anterior, lo que garantiza coherencia (Φ monótono
        no decreciente) entre k consecutivos.
        """
        bloques: list = [(frozenset(range(self._N)), frozenset(range(self._n_dims)))]
        historico: dict[int, tuple[float, list]] = {1: (self._emd_bloques(bloques), list(bloques))}
        while len(bloques) < self._N:
            resultado = self._mejor_split_bloques(bloques, pool)
            if resultado is None:
                break
            phi, bloques = resultado
            historico[len(bloques)] = (phi, list(bloques))
        return historico

    # ── Refinamiento local best-improvement (1-move futuro + presente) ─────

    def _refinar_bloques(
        self,
        bloques: "list[tuple[frozenset, frozenset]]",
        phi: float,
        max_iter: int = 20,
    ) -> "tuple[float, list]":
        """
        Refinamiento best-improvement sobre bloques asimétricos.

        En cada ronda evalúa TODOS los vecinos y aplica el globalmente mejor:
          - movimiento futuro:  traslada un nodo futuro del bloque i al j.
          - movimiento presente: traslada el mecanismo de un nodo del bloque i
            al j SIN mover su futuro (exclusivo del esquema asimétrico).
        Repite hasta convergencia o max_iter rondas.
        """
        bloques = [(frozenset(e), frozenset(p)) for e, p in bloques]

        for _ in range(max_iter):
            mejor: list | None = None
            mejor_phi = phi
            k = len(bloques)

            # Movimientos futuros
            for i, (eff_i, pre_i) in enumerate(bloques):
                for nodo in eff_i:
                    new_eff_i = eff_i - {nodo}
                    if not new_eff_i and not pre_i:
                        continue  # resultaría en (∅, ∅)
                    for j in range(k):
                        if i == j:
                            continue
                        eff_j, pre_j = bloques[j]
                        cfg = list(bloques)
                        cfg[i] = (new_eff_i, pre_i)
                        cfg[j] = (eff_j | {nodo}, pre_j)
                        phi_cand = self._emd_bloques(cfg)
                        if phi_cand < mejor_phi - 1e-10:
                            mejor_phi = phi_cand
                            mejor = cfg

            # Movimientos presentes (asimétrico)
            for i, (eff_i, pre_i) in enumerate(bloques):
                # Sin presente vacío: no sacar el único mecanismo de un bloque.
                if not self._permitir_presente_vacio and len(pre_i) <= 1:
                    continue
                # Nunca vaciar el único presente de un bloque con futuro ∅ → resultaría (∅,∅).
                if not eff_i and len(pre_i) <= 1:
                    continue
                for nodo in pre_i:
                    for j in range(k):
                        if i == j:
                            continue
                        eff_j, pre_j = bloques[j]
                        cfg = list(bloques)
                        cfg[i] = (eff_i, pre_i - {nodo})
                        cfg[j] = (eff_j, pre_j | {nodo})
                        phi_cand = self._emd_bloques(cfg)
                        if phi_cand < mejor_phi - 1e-10:
                            mejor_phi = phi_cand
                            mejor = cfg

            if mejor is None:
                break
            bloques = mejor
            phi = mejor_phi

        return phi, bloques

    # ── 2-move sistemático (VNS) ───────────────────────────────────────────

    def _refinar_bloques_2move(
        self,
        bloques: "list[tuple[frozenset, frozenset]]",
        phi: float,
    ) -> "tuple[float, list, bool]":
        """
        2-move sistemático post-convergencia 1-move (VNS).

        Evalúa todos los pares de movimientos 1-move independientes aplicados
        simultáneamente y devuelve el mejor par que mejore phi.

        Un par (mov_a, mov_b) es descartado si el nodo que mueve mov_b ya fue
        reubicado por mov_a (detección implícita por ausencia en el bloque fuente).

        Returns:
            (phi, bloques, mejoro) — mejoro=True si se encontró mejora.
        """
        k = len(bloques)

        movimientos: list = []
        for i, (eff_i, pre_i) in enumerate(bloques):
            for nodo in sorted(eff_i):
                if not (eff_i - {nodo}) and not pre_i:
                    continue  # resultaría en (∅, ∅)
                for j in range(k):
                    if j != i:
                        movimientos.append(("f", i, j, nodo))
            # Sin presente vacío: no generar movimientos que saquen el único mecanismo.
            if not self._permitir_presente_vacio and len(pre_i) <= 1:
                continue
            # Nunca vaciar el único presente de un bloque con futuro ∅ → resultaría (∅,∅).
            if not eff_i and len(pre_i) <= 1:
                continue
            for nodo in sorted(pre_i):
                for j in range(k):
                    if j != i:
                        movimientos.append(("p", i, j, nodo))

        mejor_phi = phi
        mejor_cfg: list | None = None
        M = len(movimientos)

        for a in range(M):
            tipo_a, i_a, j_a, nodo_a = movimientos[a]
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
                    bl_b[i_b] = (eff_ib, pre_ib - {nodo_b})
                    bl_b[j_b] = (eff_jb, pre_jb | {nodo_b})

                if any(not eff and not pre for eff, pre in bl_b):
                    continue
                # Sin presente vacío: el par combinado no puede dejar bloques sin mecanismo.
                if not self._permitir_presente_vacio and any(
                    not pre for _, pre in bl_b
                ):
                    continue

                phi_cand = self._emd_bloques(bl_b)
                if phi_cand < mejor_phi - 1e-10:
                    mejor_phi = phi_cand
                    mejor_cfg = bl_b

        if mejor_cfg is not None:
            return mejor_phi, mejor_cfg, True
        return phi, bloques, False

    # ── Perturbación + ILS ─────────────────────────────────────────────────

    def _perturbar_bloques(
        self,
        bloques: "list[tuple[frozenset, frozenset]]",
        n_movimientos: int = 2,
        semilla: int = 42,
    ) -> "list[tuple[frozenset, frozenset]]":
        """
        Perturba una configuración de bloques alternando movimientos futuros y
        presentes aleatorios, garantizando que ningún bloque quede vacío (∅, ∅).
        """
        rng = _random_module.Random(semilla)
        result = [(frozenset(e), frozenset(p)) for e, p in bloques]
        k = len(result)
        if k < 2:
            return result

        for _ in range(n_movimientos):
            tipo = rng.randint(0, 1)
            if tipo == 0:  # movimiento futuro
                candidatos = [
                    i for i, (e, p) in enumerate(result)
                    if len(e) >= 1 and (len(e) > 1 or len(p) > 0)
                ]
                if not candidatos:
                    continue
                i = rng.choice(candidatos)
                eff_i, pre_i = result[i]
                nodo = rng.choice(sorted(eff_i))
                j = rng.choice([x for x in range(k) if x != i])
                eff_j, pre_j = result[j]
                result[i] = (eff_i - {nodo}, pre_i)
                result[j] = (eff_j | {nodo}, pre_j)
            else:  # movimiento presente (asimétrico)
                # Sin presente vacío: solo bloques con ≥2 mecanismos pueden ceder uno.
                # Excluir además bloques con futuro ∅ y un solo presente: quedarían (∅, ∅).
                if self._permitir_presente_vacio:
                    candidatos = [
                        i for i, (e, p) in enumerate(result)
                        if len(p) > 0 and (len(e) > 0 or len(p) > 1)
                    ]
                else:
                    candidatos = [i for i, (_, p) in enumerate(result) if len(p) > 1]
                if not candidatos:
                    continue
                i = rng.choice(candidatos)
                eff_i, pre_i = result[i]
                nodo = rng.choice(sorted(pre_i))
                j = rng.choice([x for x in range(k) if x != i])
                eff_j, pre_j = result[j]
                result[i] = (eff_i, pre_i - {nodo})
                result[j] = (eff_j, pre_j | {nodo})

        return result

    def _refinar_con_ils(
        self,
        bloques: "list[tuple[frozenset, frozenset]]",
        phi: float,
    ) -> "tuple[float, list]":
        """
        ILS: refinamiento best-improvement + ciclos N-adaptativos de perturbación.

        n_ils y max_iter decrecen con N porque cada evaluación EMD es O(2^N) para
        N ≤ HAMMING_EMD_MAX_N: pocos ciclos de calidad superan muchos mediocres.
          max_iter = max(5, 20 - max(0, N - HAMMING_EMD_MAX_N))
          n_ils    = max(1, _N_ILS - max(0, (N - 16) // 2))
        """
        max_it = max(5, 20 - max(0, self._N - HAMMING_EMD_MAX_N))
        n_ils = max(1, _N_ILS - max(0, (self._N - 16) // 2))

        mejor_phi, mejor_bloques = self._refinar_bloques(bloques, phi, max_iter=max_it)

        # VNS: ciclos 2-move + 1-move hasta que no haya mejora (máx 3 ciclos).
        N_VNS_MAX = 3
        for _ in range(N_VNS_MAX):
            phi_2m, bloques_2m, mejoro_2m = self._refinar_bloques_2move(
                mejor_bloques, mejor_phi
            )
            if not mejoro_2m:
                break
            phi_ref, bloques_ref = self._refinar_bloques(bloques_2m, phi_2m, max_iter=max_it)
            if phi_ref < mejor_phi - 1e-9:
                mejor_phi = phi_ref
                mejor_bloques = bloques_ref

        n_mov = max(1, self._N // 4)
        for iter_ils in range(n_ils):
            perturbado = self._perturbar_bloques(
                mejor_bloques, n_movimientos=n_mov, semilla=42 + iter_ils * 17
            )
            phi_p = self._emd_bloques(perturbado)
            phi_r, bloques_r = self._refinar_bloques(perturbado, phi_p, max_iter=max_it)
            if phi_r < mejor_phi - 1e-9:
                mejor_phi = phi_r
                mejor_bloques = bloques_r

        return mejor_phi, mejor_bloques

    # ── Evaluación EMD de una partición de bloques ─────────────────────────

    def _emd_bloques(self, bloques: "list[tuple[frozenset, frozenset]]") -> float:
        """
        EMD total de una k-partición asimétrica de bloques.

        Reconstruye la distribución marginal a partir de las distribuciones de
        cada bloque (futuro condicionado por su propio presente) y la compara
        con la original mediante suma L1 marginal. Esta L1 es la Wasserstein-1
        EXACTA con métrica de Hamming, porque ambas distribuciones son productos
        de marginales (ver docstring del módulo): O(N) en vez de O(4^N).
        """
        dist_rec = np.empty(self._N, dtype=np.float64)
        for fut_pos, pre_pos in bloques:
            if not fut_pos:
                continue
            dist_bloque = self._dist_bloque(fut_pos, pre_pos)
            for p in fut_pos:
                dist_rec[p] = float(dist_bloque[p])

        return float(np.sum(np.abs(dist_rec - self.sia_dists_marginales)))

    # ── Memoización de distribuciones de bloque ────────────────────────────

    def _dist_bloque(self, fut_pos: frozenset, pre_pos: frozenset) -> np.ndarray:
        """
        Distribución marginal del bloque (futuros, presentes), calculada una vez.

        Usa NCube.marginal_valor: fija los dims del mecanismo al estado inicial y
        promedia solo sobre los dims restantes → O(2^(N-|pre|)) por cubo en vez de
        O(2^N) de bipartir→marginalizar. Para |pre|≈N/2 esto es un speedup de 2^(N/2)
        (×1000 en N=20). El resultado es idéntico por linealidad del valor esperado:
        E_{V-pre}[data[pre=s0, •]] == marginalizar(V-pre)[s0].

        Resultado en float64: igual que `System.distribucion_marginal` (float64) y que
        la fuerza bruta. El vector es de tamaño N (no la tabla de costos), así que
        float64 no impacta memoria y hace que QNodes, KGeoMIP y la fuerza bruta
        coincidan a ~1e-15 en vez de diferir ~1e-8 por redondeo en float32.
        """
        clave = (fut_pos, pre_pos)
        cache = self._cache_bloque.get(clave)
        if cache is None:
            pre_global: frozenset = frozenset(
                int(self._dims[q]) for q in pre_pos if q < self._n_dims
            )
            estado = self.sia_subsistema.estado_inicial
            result = np.zeros(self._N, dtype=np.float64)
            for p in fut_pos:
                ncubo = self._ncubos_idx[int(self._idx[p])]
                # Ejes a promediar = dims del cubo NO en el mecanismo.
                ejes = np.array(
                    [d for d in ncubo.dims if int(d) not in pre_global],
                    dtype=np.int8,
                )
                result[p] = ncubo.marginal_valor(ejes, estado)
            cache = result
            self._cache_bloque[clave] = cache
        return cache


# Alias de compatibilidad con exec.py (que importa DynamicPartition)
DynamicPartition = QNodes
