"""
DynamicPartition — k-MIP exhaustivo mediante Programación Dinámica + Branch & Bound.

Tres optimizaciones sobre el problema de k-partición óptima (k=2..N):

1. Poda Branch & Bound: mantiene `_mejor_phi` como cota superior dinámica.
   Abandona cualquier rama cuyo costo acumulado >= _mejor_phi sin completarla.

2. Memoización Perezosa (Top-Down): `_dist_parte(mascara)` y `_costo_parte(mascara)`
   solo se calculan cuando una rama activa los instancia, nunca de forma anticipada.
   Esto evita la inundación de RAM propia del bottom-up cuando N es alto.

3. Operaciones Bitwise estrictas: subconjuntos representados como máscaras enteras.
   Unión, intersección e iteración se realizan con operadores bit a bit en lugar de
   búsquedas en arrays, tuplas o strings, reduciendo lookups costosos a ciclos de CPU.

Adicionalmente, la búsqueda usa ordenación canónica (la próxima parte siempre contiene
el nodo de índice mínimo restante) para eliminar permutaciones equivalentes de la misma
partición y así explorar cada partición única exactamente una vez.
"""
import time
from typing import Generator, Optional

import numpy as np

from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import gestor_perfilado, profile
from src.funcs.format import fmt_k_particion_dp
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import (
    QNODES_ANALYSIS_TAG,
    QNODES_LABEL,
    QNODES_STRAREGY_TAG,
)
from src.constants.base import COLS_IDX, NET_LABEL, TYPE_TAG
from src.models.base.application import aplicacion


# Mantener la posibilidad de que una parte tenga mecanismo vacío (∅).
PERMITIR_PRESENTE_VACIO_POR_DEFECTO: bool = False


def _indices_activos(mascara: int) -> Generator[int, None, None]:
    """Genera los índices de bits activos de `mascara` en orden ascendente."""
    m = mascara
    while m:
        bit = m & (-m)               # bit menos significativo activo
        yield bit.bit_length() - 1   # posición local del bit
        m ^= bit                     # apagar ese bit y seguir


class DynamicPartition(SIA):
    """
    Estrategia de Partición Dinámica con Memoización para k-MIP exhaustivo.

    Encuentra la k-partición óptima (k ∈ [2, N]) que minimiza la pérdida
    de información integrada (Phi) mediante Programación Dinámica con:

    - Memoización Top-Down perezosa: calcula dist_parte[mascara] solo cuando
      esa sub-parte es instanciada por una rama activa del árbol de búsqueda.
    - Branch & Bound: abandona ramas cuyo costo acumulado >= mejor_phi conocido.
    - Representación bitwise: subconjuntos como enteros para operaciones O(1).
    - Ordenación canónica: garantiza que cada partición única se explore una sola vez.

    A diferencia de la implementación anterior (Stoer-Wagner, k=2 únicamente),
    esta estrategia garantiza el óptimo global evaluando todos los valores de k.

    Attrs:
        _cache_dist      : mascara (int) → distribución marginal normal.
        _cache_dist_vacio: mascara (int) → distribución marginal con presentes=∅.
        _cache_costo     : mascara (int) → contribución Phi de esa parte.
        _usar_vacio      : mascara (int) → True si la variante vacía dio menor costo.
        _mejor_phi       : menor Phi encontrado hasta el momento (cota superior B&B).
        _mejor_particion : lista de máscaras que define la partición óptima actual.
    """

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self._cache_dist: dict[int, np.ndarray] = {}
        self._cache_costo: dict[int, float] = {}
        self._mejor_phi: float = float("inf")
        self._mejor_particion: list[int] = []
        self._N: int = 0
        self.logger = SafeLogger(QNODES_STRAREGY_TAG)
        # Soporte para mecanismo vacío (∅)
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
        mascara_total = (1 << self._N) - 1

        # Reiniciar estado entre ejecuciones
        self._cache_dist.clear()
        self._cache_costo.clear()
        self._cache_dist_vacio.clear()
        self._usar_vacio.clear()
        self._mejor_phi = float("inf")
        self._mejor_particion = []

        if self._N < 2:
            # Sistema trivial: no existe partición válida k>=2
            dist_trivial = self.sia_dists_marginales.copy()
            return Solution(
                estrategia=QNODES_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=dist_trivial,
                tiempo_total=time.time() - self.sia_tiempo_inicio,
                particion="Sistema trivial (N<2)\n",
            )

        # Validar k si fue especificado por el usuario
        if k is not None and (k < 2 or k > self._N):
            raise ValueError(
                f"k={k} fuera del rango permitido [2, {self._N}]"
            )

        # Warm-start: cota superior inicial para acelerar la poda B&B
        if k is None or k == 2:
            # N biparticiones de singleton {nodo_i | resto}
            for i in range(self._N):
                bit_i = 1 << i
                bit_resto = mascara_total ^ bit_i
                costo_warmup = self._costo_parte(bit_i) + self._costo_parte(bit_resto)
                if costo_warmup < self._mejor_phi:
                    self._mejor_phi = costo_warmup
                    self._mejor_particion = [bit_i, bit_resto]
        else:
            # Warm-start para k > 2: distribución round-robin
            grupos_rr = [0] * k
            for i in range(self._N):
                grupos_rr[i % k] |= (1 << i)
            if all(g > 0 for g in grupos_rr):
                costo_ws = sum(self._costo_parte(g) for g in grupos_rr)
                if costo_ws < self._mejor_phi:
                    self._mejor_phi = costo_ws
                    self._mejor_particion = list(grupos_rr)

            # Variante k-1 singletons consecutivos + residual
            for primer_nodo in range(self._N - k + 1):
                partes_ws: list[int] = []
                bits_usados = 0
                for idx_s in range(primer_nodo, primer_nodo + k - 1):
                    bit = 1 << idx_s
                    partes_ws.append(bit)
                    bits_usados |= bit
                bit_residual = mascara_total ^ bits_usados
                if bit_residual and len(partes_ws) == k - 1:
                    partes_ws.append(bit_residual)
                    costo_ws = sum(self._costo_parte(p) for p in partes_ws)
                    if costo_ws < self._mejor_phi:
                        self._mejor_phi = costo_ws
                        self._mejor_particion = list(partes_ws)

        # Búsqueda DP exhaustiva con Branch & Bound sobre todas las k-particiones
        self._dp_buscar(mascara_total, 0.0, [], k)

        # Reconstruir distribución de la partición óptima
        dist_reconstruida = np.empty(self._N, dtype=np.float32)
        for mascara in self._mejor_particion:
            dist_parte = self._dist_parte_efectiva(mascara)
            for i in _indices_activos(mascara):
                dist_reconstruida[i] = float(dist_parte[i])

        mascaras_vacio = {m for m in self._mejor_particion if self._usar_vacio.get(m, False)}
        fmt_mip = fmt_k_particion_dp(
            self._mejor_particion,
            self.sia_subsistema.indices_ncubos,
            self.sia_subsistema.dims_ncubos,
            mascaras_vacio,
        )

        return Solution(
            estrategia=QNODES_LABEL,
            perdida=self._mejor_phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_reconstruida,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    # ── Memoización perezosa (Top-Down) ────────────────────────────────────

    def _dist_parte(self, mascara: int) -> np.ndarray:
        """
        Calcula y cachea la distribución marginal de la parte definida por `mascara`.

        La parte identificada por `mascara` contiene todos los nodos cuyo bit local
        esté activo en la máscara. Los nodos del sistema aparecen a la vez como
        futuros (t+1) y presentes (t), donde:

          futuros(mascara)  = indices_ncubos[bits(mascara)]
          presentes(mascara) = intersect(futuros, dims_ncubos)

        Solo se computa cuando la máscara es instanciada por una rama activa
        del árbol de PD — nunca se pre-calcula de forma anticipada.
        """
        if mascara not in self._cache_dist:
            idx_arr = np.fromiter(_indices_activos(mascara), dtype=np.int8)
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
        """Distribución marginal de la parte usando mecanismo vacío (∅)."""
        if mascara not in self._cache_dist_vacio:
            idx_arr = np.fromiter(_indices_activos(mascara), dtype=np.int8)
            futuros = self.sia_subsistema.indices_ncubos[idx_arr]
            dist = (
                self.sia_subsistema
                .bipartir(futuros, np.array([], dtype=np.int8))
                .distribucion_marginal()
            )
            self._cache_dist_vacio[mascara] = dist
        return self._cache_dist_vacio[mascara]

    def _dist_parte_efectiva(self, mascara: int) -> np.ndarray:
        """Retorna la distribución correcta según si esa parte usa variante vacía o no."""
        if self._usar_vacio.get(mascara, False):
            return self._dist_parte_vacio(mascara)
        return self._dist_parte(mascara)

    def _costo_parte(self, mascara: int) -> float:
        """
        Contribución Phi de la parte `mascara` (aditiva sobre emd_efecto).

        Si `_permitir_presente_vacio` está activo, evalúa también la variante con
        mecanismo vacío (∅) y usa la que produzca menor costo, registrando la elección
        en `_usar_vacio[mascara]` para la reconstrucción posterior.

        Descomposición exacta:
          costo(parte) = sum_{i in parte} |dist_parte[i] - dist_original[i]|

        La suma de costos de todas las partes iguala emd_efecto(dist_reconstruida,
        dist_original), lo que valida el Branch & Bound acumulativo: en cuanto la
        suma parcial >= mejor_phi, ningún complemento puede mejorar el óptimo.
        """
        if mascara not in self._cache_costo:
            dist = self._dist_parte(mascara)
            idx_arr = np.fromiter(_indices_activos(mascara), dtype=np.int8)
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

    # ── Búsqueda DP + Branch & Bound ───────────────────────────────────────

    def _dp_buscar(
        self,
        restantes: int,
        costo_acum: float,
        partes: list[int],
        k_objetivo: Optional[int] = None,
    ) -> None:
        """
        Búsqueda recursiva sobre todas las k-particiones válidas.

        Si k_objetivo es None, busca el mínimo phi sobre todas las k >= 2.
        Si k_objetivo es un entero, solo acepta particiones con exactamente k partes
        y aplica podas adicionales:
          - Si ya hay k partes pero restan nodos sin asignar → poda inmediata.
          - Si queda solo 1 parte por formar, asigna todos los nodos restantes de
            golpe (por canonicidad, esta asignación es única e irremovible).

        Invariante de ordenación canónica: la próxima parte SIEMPRE incluye el
        nodo de índice mínimo aún sin asignar (bit_min). Esto garantiza que cada
        partición única sea generada exactamente una vez, sin contar permutaciones
        de las mismas partes en distinto orden.

        Branch & Bound: si `costo_acum` ya alcanza o supera `_mejor_phi`, la rama
        completa se abandona, pues agregar más partes no puede reducir el costo total.

        Args:
            restantes  : Máscara de nodos pendientes de asignación.
            costo_acum : Suma de contribuciones Phi de las partes ya formadas.
            partes     : Máscaras de las partes ya asignadas (estado mutable compartido).
            k_objetivo : Número exacto de partes requeridas, o None para buscar todo k>=2.
        """
        if restantes == 0:
            cumple_k = (
                (k_objetivo is None and len(partes) >= 2)
                or (k_objetivo is not None and len(partes) == k_objetivo)
            )
            if cumple_k and costo_acum < self._mejor_phi:
                self._mejor_phi = costo_acum
                self._mejor_particion = list(partes)
            return

        # Poda Branch & Bound: costo parcial ya alcanzó la cota superior
        if costo_acum >= self._mejor_phi:
            return

        # Podas adicionales para k exacto
        if k_objetivo is not None:
            partes_act = len(partes)
            if partes_act >= k_objetivo:
                return  # ya hay k partes pero quedan nodos → imposible
            if partes_act == k_objetivo - 1:
                # La última parte debe ser todos los restantes (ordenación canónica)
                costo_ultima = self._costo_parte(restantes)
                nuevo_costo = costo_acum + costo_ultima
                if nuevo_costo < self._mejor_phi:
                    self._mejor_phi = nuevo_costo
                    self._mejor_particion = list(partes) + [restantes]
                return

        # Nodo de índice mínimo restante → ancla canónica de la próxima parte
        bit_min = restantes & (-restantes)
        otros = restantes ^ bit_min  # nodos restantes sin el nodo ancla

        # Enumerar todos los subconjuntos de `otros` (vacío incluido).
        # La parte formada = submascara | bit_min (siempre contiene bit_min).
        submascara = otros
        while True:
            parte = submascara | bit_min
            costo_parte = self._costo_parte(parte)

            partes.append(parte)
            self._dp_buscar(restantes ^ parte, costo_acum + costo_parte, partes, k_objetivo)
            partes.pop()

            if submascara == 0:
                break
            # Siguiente subconjunto estricto de `otros`
            submascara = (submascara - 1) & otros
