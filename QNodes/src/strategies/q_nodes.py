"""
QNodes — k-MIP heurístico mediante agrupamiento jerárquico aglomerativo.

El problema k-MIP (Partición de Mínima Información) requiere dividir un sistema
de N nodos en k subconjuntos disjuntos minimizando la pérdida de información
integrada Phi (Φ). La búsqueda exhaustiva sobre B(N) particiones es intractable
para N ≥ 15 (números de Bell), por lo que QNodes usa una estrategia greedy O(N³).

Algoritmo principal — tres fases para todo N
─────────────────────────────────────────────
1. Agrupamiento aglomerativo greedy (siempre O(N³)):
   - Inicializar N singletons {nodo_i} para i ∈ [0, N).
   - Repetir hasta tener 2 grupos:
       a. Evaluar todos los C(k,2) pares candidatos con _emd_particion.
       b. Fusionar el par con menor Phi resultante.
       c. Registrar historico[k] = (phi, grupos) para cada nivel k.
   - Retornar historial completo {k: (phi, grupos)} para k ∈ [2, N].

2. Refinamiento local 1-move (siempre):
   - Para cada nodo en cada grupo, evaluar moverlo a otro grupo con _emd_particion.
   - Aceptar si mejora Phi. Repetir hasta convergencia (máx. 20 pasadas).

3. Candidatos de aislamiento (siempre, para cada k evaluado):
   - Generar C(N, k-1) candidatos con k-1 nodos individualmente aislados.
   - Si alguno supera la solución greedy+refinamiento, adoptarlo y refinar de nuevo.

Para k libre (k=None): las tres fases se aplican a CADA nivel k del historial
y se elige el k ∈ [3, N] con menor Phi global.

Métrica EMD:
   _emd_particion usa internamente la métrica más precisa que el tamaño permite:
   N ≤ HAMMING_EMD_MAX_N → Wasserstein-1 con d_Hamming sobre el espacio 2^N (exacta).
   N > HAMMING_EMD_MAX_N → suma L1 marginal (aproximación rápida y tratable).
   Este detalle de implementación es transparente para la estrategia.

Memoización:
  • _cache_dist[mascara]:  distribución marginal de la parte 'mascara'.
  • _cache_costo[mascara]: costo L1 de 'mascara' (para decisión de mecanismo vacío).
  Ambos cachés se limpian entre ejecuciones independientes.
"""
import time
from typing import Generator, Optional

import numpy as np

from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import gestor_perfilado, profile
from src.funcs.format import fmt_k_particion_dp
from src.funcs.iit import distribucion_conjunta_vectorizada, emd_causal, HAMMING_EMD_MAX_N
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import (
    QNODES_ANALYSIS_TAG,
    QNODES_LABEL,
    QNODES_STRAREGY_TAG,
)
from src.constants.base import COLS_IDX, NET_LABEL, TYPE_TAG
from src.models.base.application import aplicacion


PERMITIR_PRESENTE_VACIO_POR_DEFECTO: bool = False


def _bits_activos(mascara: int) -> Generator[int, None, None]:
    """Genera los índices de bits activos de `mascara` en orden ascendente."""
    m = mascara
    while m:
        bit = m & (-m)
        yield bit.bit_length() - 1
        m ^= bit


class QNodes(SIA):
    """
    Estrategia QNodes para k-MIP mediante agrupamiento jerárquico aglomerativo.

    Aplica tres fases para todo N: agrupamiento greedy O(N³), refinamiento local
    1-move y candidatos de aislamiento C(N, k-1). Para k libre, las tres fases se
    ejecutan sobre cada nivel k del historial y se elige el k con menor Phi global.

    La métrica usada en _emd_particion es Hamming EMD (N ≤ HAMMING_EMD_MAX_N) o
    L1 marginal (N > HAMMING_EMD_MAX_N), de forma transparente para la estrategia.

    Attrs:
        _cache_dist      : mascara → distribución marginal normal.
        _cache_dist_vacio: mascara → distribución marginal con presentes = ∅.
        _cache_costo     : mascara → costo L1 de esa parte (para decisión de ∅).
        _usar_vacio      : mascara → True si la variante vacía dio menor costo.
    """

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self._cache_dist: dict[int, np.ndarray] = {}
        self._cache_costo: dict[int, float] = {}
        self._N: int = 0
        self.logger = SafeLogger(QNODES_STRAREGY_TAG)
        self._permitir_presente_vacio: bool = False
        self._cache_dist_vacio: dict[int, np.ndarray] = {}
        self._usar_vacio: dict[int, bool] = {}

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

        self._N = len(self.sia_subsistema.indices_ncubos)

        # Limpiar cachés entre ejecuciones
        self._cache_dist.clear()
        self._cache_costo.clear()
        self._cache_dist_vacio.clear()
        self._usar_vacio.clear()

        if self._N < 2:
            dist_trivial = self.sia_dists_marginales.copy()
            return Solution(
                estrategia=QNODES_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=dist_trivial,
                tiempo_total=time.time() - self.sia_tiempo_inicio,
                particion="Sistema trivial (N<2)\n",
            )

        if k is not None and (k < 2 or k > self._N):
            raise ValueError(f"k={k} fuera del rango permitido [2, {self._N}]")

        # Fase 1: agrupamiento aglomerativo — retorna la jerarquía completa {k: (phi, grupos)}
        historico = self._aglomerar()

        if k is not None:
            # k especificado: tomar ese nivel, refinar y evaluar candidatos de aislamiento
            if k not in historico:
                raise ValueError(f"k={k} no alcanzable con N={self._N}")
            mejor_phi, mejor_grupos = historico[k]

            # Fase 2: refinamiento local 1-move
            mejor_phi, mejor_grupos = self._refinamiento_local(mejor_grupos, mejor_phi)

            # Fase 3: candidatos de aislamiento para el k dado
            for candidato_grupos in self._candidatos_aislamiento(k):
                phi_cand = self._emd_particion(candidato_grupos)
                if phi_cand < mejor_phi - 1e-10:
                    mejor_phi = phi_cand
                    mejor_grupos = candidato_grupos
            mejor_phi, mejor_grupos = self._refinamiento_local(mejor_grupos, mejor_phi)

        else:
            # k libre: las tres fases se aplican a CADA nivel k del historial.
            # Se elige el k ∈ [3, N] con menor Phi entre todos los niveles refinados.
            historico_refinado: dict[int, tuple[float, list[int]]] = {}
            for k_nivel, (phi_nivel, grupos_nivel) in historico.items():
                phi_r, grupos_r = self._refinamiento_local(grupos_nivel, phi_nivel)
                # candidatos de aislamiento para este nivel (k=N son triviales, se omiten)
                if k_nivel < self._N:
                    for candidato_grupos in self._candidatos_aislamiento(k_nivel):
                        phi_cand = self._emd_particion(candidato_grupos)
                        if phi_cand < phi_r - 1e-10:
                            phi_r = phi_cand
                            grupos_r = candidato_grupos
                    phi_r, grupos_r = self._refinamiento_local(grupos_r, phi_r)
                historico_refinado[k_nivel] = (phi_r, grupos_r)

            candidatos_k3 = {
                kk: ph for kk, (ph, _) in historico_refinado.items() if kk >= 3
            }
            if not candidatos_k3:
                candidatos_k3 = {kk: ph for kk, (ph, _) in historico_refinado.items()}
            mejor_k = min(candidatos_k3, key=candidatos_k3.get)
            mejor_phi, mejor_grupos = historico_refinado[mejor_k]

        # Reconstruir distribución de la partición óptima
        dist_reconstruida = np.empty(self._N, dtype=np.float32)
        for mascara in mejor_grupos:
            dist_parte = self._dist_parte_efectiva(mascara)
            for i in _bits_activos(mascara):
                dist_reconstruida[i] = float(dist_parte[i])

        mascaras_vacio = {m for m in mejor_grupos if self._usar_vacio.get(m, False)}
        fmt_mip = fmt_k_particion_dp(
            mejor_grupos,
            self.sia_subsistema.indices_ncubos,
            self.sia_subsistema.dims_ncubos,
            mascaras_vacio,
        )

        return Solution(
            estrategia=QNODES_LABEL,
            perdida=mejor_phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_reconstruida,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    # ── Agrupamiento aglomerativo greedy ───────────────────────────────────

    def _aglomerar(self) -> dict[int, tuple[float, list[int]]]:
        """
        Construye la jerarquía completa de k-particiones fusionando de N hasta 2.

        Siempre evalúa cada fusión candidata con _emd_particion (que internamente
        usa Hamming EMD o L1 según N). Retorna el historial completo:
        {k: (phi, grupos)} para k ∈ [2, N].
        """
        grupos: list[int] = [1 << i for i in range(self._N)]

        # Pre-poblar caché de distribuciones para todos los singletons
        for g in grupos:
            self._dist_parte(g)
        phi_total = self._emd_particion(grupos)

        historico: dict[int, tuple[float, list[int]]] = {
            self._N: (phi_total, list(grupos))
        }

        while len(grupos) > 2:
            mejor_phi_merge = float("inf")
            mejor_i = mejor_j = -1

            n_grupos = len(grupos)
            for i in range(n_grupos):
                for j in range(i + 1, n_grupos):
                    union = grupos[i] | grupos[j]
                    candidato = [
                        g for idx, g in enumerate(grupos)
                        if idx != i and idx != j
                    ] + [union]
                    phi_cand = self._emd_particion(candidato)
                    if phi_cand < mejor_phi_merge:
                        mejor_phi_merge = phi_cand
                        mejor_i, mejor_j = i, j

            gi, gj = grupos[mejor_i], grupos[mejor_j]
            union = gi | gj
            grupos = [
                g for idx, g in enumerate(grupos)
                if idx != mejor_i and idx != mejor_j
            ]
            grupos.append(union)
            phi_total = mejor_phi_merge
            historico[len(grupos)] = (phi_total, list(grupos))

        return historico

    # ── Refinamiento local 1-move ───────────────────────────────────────────

    def _refinamiento_local(
        self,
        grupos: list[int],
        phi_total: float,
        max_iter: int = 20,
    ) -> tuple[float, list[int]]:
        """
        Refinamiento post-agrupamiento: prueba mover un nodo a otro grupo.

        Evalúa cada movimiento candidato con _emd_particion. Acepta si mejora
        Phi. Repite hasta convergencia (máximo max_iter pasadas).
        Garantiza un óptimo local 1-move: ningún traslado individual mejora Phi.
        """
        grupos = list(grupos)

        for _ in range(max_iter):
            mejorado = False

            for idx_origen in range(len(grupos)):
                g_origen = grupos[idx_origen]

                if bin(g_origen).count("1") <= 1:
                    continue

                for bit_nodo in list(_bits_activos(g_origen)):
                    mascara_nodo = 1 << bit_nodo
                    g_sin_nodo = g_origen ^ mascara_nodo

                    for idx_dest in range(len(grupos)):
                        if idx_dest == idx_origen:
                            continue

                        g_dest = grupos[idx_dest]
                        g_con_nodo = g_dest | mascara_nodo

                        candidato = list(grupos)
                        candidato[idx_origen] = g_sin_nodo
                        candidato[idx_dest] = g_con_nodo

                        phi_cand = self._emd_particion(candidato)

                        if phi_cand < phi_total - 1e-10:
                            grupos = candidato
                            phi_total = phi_cand
                            g_origen = g_sin_nodo
                            mejorado = True
                            break

                    if mejorado:
                        break
                if mejorado:
                    break

            if not mejorado:
                break

        return phi_total, grupos

    # ── Candidatos de aislamiento ──────────────────────────────────────────

    def _candidatos_aislamiento(self, k: int):
        """
        Genera candidatos donde k-1 nodos están aislados y el resto forma un grupo.

        Para k=3, N=10: C(10,2)=45 candidatos.
        Para k libre con todos los niveles: Σ C(N,k-1) = 2^N − 2 candidatos en total.
        Idéntico a GeoMIP's _generar_candidatos_aislamiento.
        """
        import itertools
        todos = list(range(self._N))
        n_aislados = k - 1
        for aislados in itertools.combinations(todos, n_aislados):
            aislados_set = set(aislados)
            residual = [i for i in todos if i not in aislados_set]
            if not residual:
                continue
            mascara_residual = sum(1 << i for i in residual)
            mascaras = [1 << a for a in aislados] + [mascara_residual]
            yield mascaras

    # ── EMD de partición completa ───────────────────────────────────────────

    def _emd_particion(self, grupos: list[int]) -> float:
        """
        EMD total de una k-partición completa.

        Métrica adaptada al tamaño del sistema (transparente para la estrategia):
          N ≤ HAMMING_EMD_MAX_N → Wasserstein-1 con d_Hamming sobre 2^N estados.
          N > HAMMING_EMD_MAX_N → suma L1 marginal sobre N nodos (aproximación rápida).

        La memoización de _cache_dist hace que llamadas repetidas para la misma
        máscara sean O(1) sin recomputar la distribución marginal.
        """
        if self._permitir_presente_vacio:
            for mascara in grupos:
                if mascara not in self._usar_vacio:
                    self._costo_parte(mascara)

        dist_rec = np.empty(self._N, dtype=np.float64)
        for mascara in grupos:
            dist_parte = self._dist_parte_efectiva(mascara)
            for i in _bits_activos(mascara):
                dist_rec[i] = float(dist_parte[i])

        if self._N <= HAMMING_EMD_MAX_N:
            P = distribucion_conjunta_vectorizada(self.sia_dists_marginales.astype(np.float64))
            Q = distribucion_conjunta_vectorizada(dist_rec)
            return float(emd_causal(P, Q))
        else:
            return float(np.sum(np.abs(dist_rec - self.sia_dists_marginales)))

    # ── Memoización de distribuciones y costos ─────────────────────────────

    def _dist_parte(self, mascara: int) -> np.ndarray:
        """Distribución marginal de la parte `mascara` (calculada una sola vez)."""
        if mascara not in self._cache_dist:
            idx_arr = np.fromiter(_bits_activos(mascara), dtype=np.int8)
            futuros = self.sia_subsistema.indices_ncubos[idx_arr]
            presentes = np.intersect1d(futuros, self.sia_subsistema.dims_ncubos)
            dist = (
                self.sia_subsistema
                .bipartir(futuros, presentes)
                .distribucion_marginal()
            )
            self._cache_dist[mascara] = dist
        return self._cache_dist[mascara]

    def _dist_parte_vacio(self, mascara: int) -> np.ndarray:
        """Distribución marginal de la parte `mascara` con mecanismo vacío (∅)."""
        if mascara not in self._cache_dist_vacio:
            idx_arr = np.fromiter(_bits_activos(mascara), dtype=np.int8)
            futuros = self.sia_subsistema.indices_ncubos[idx_arr]
            dist = (
                self.sia_subsistema
                .bipartir(futuros, np.array([], dtype=np.int8))
                .distribucion_marginal()
            )
            self._cache_dist_vacio[mascara] = dist
        return self._cache_dist_vacio[mascara]

    def _dist_parte_efectiva(self, mascara: int) -> np.ndarray:
        """Retorna la distribución de la parte según si usa variante vacía."""
        if self._usar_vacio.get(mascara, False):
            return self._dist_parte_vacio(mascara)
        return self._dist_parte(mascara)

    def _costo_parte(self, mascara: int) -> float:
        """
        Costo L1 de la parte `mascara` para la decisión de mecanismo vacío.

        Solo se usa cuando `_permitir_presente_vacio=True`: compara el costo L1
        normal vs el costo L1 con mecanismo ∅ y registra cuál es menor en
        `_usar_vacio`. La selección de distribución resultante luego la usa
        `_emd_particion` al llamar a `_dist_parte_efectiva`.
        """
        if mascara not in self._cache_costo:
            dist = self._dist_parte(mascara)
            idx_arr = np.fromiter(_bits_activos(mascara), dtype=np.int8)
            costo_normal = float(
                np.sum(np.abs(dist[idx_arr] - self.sia_dists_marginales[idx_arr]))
            )

            if self._permitir_presente_vacio:
                dist_v = self._dist_parte_vacio(mascara)
                costo_vacio = float(
                    np.sum(np.abs(dist_v[idx_arr] - self.sia_dists_marginales[idx_arr]))
                )
                if costo_vacio < costo_normal:
                    self._usar_vacio[mascara] = True
                    self._cache_costo[mascara] = costo_vacio
                else:
                    self._usar_vacio[mascara] = False
                    self._cache_costo[mascara] = costo_normal
            else:
                self._cache_costo[mascara] = costo_normal

        return self._cache_costo[mascara]


# Alias de compatibilidad con exec.py (que importa DynamicPartition)
DynamicPartition = QNodes
